"""The link: everything between a vehicle's true state and its motors spinning.

Every result in this repo up to now was measured through a *perfect* link —
controllers saw exact state and their commands took effect instantly and
completely. No real system does that, and the gap is not a rounding error: for
2-4 tinywhoops flown off a Vicon volume the round trip is

    Vicon capture + solve + network  ~8 ms
    ground station tick (BT + ONNX + shield)
    ELRS uplink @250 Hz              ~6 ms
    FC mixer + motor/prop spin-up    ~20 ms  (first-order, not transport)

which is 30-45 ms of effective lag, plus the failure modes that actually bite in
a mocap lab: marker occlusion, **rigid-body label swaps between two identical
airframes**, uplink packet loss, and thrust falling off as the pack sags.

This module models that link as a wrapper around a controller rather than as a
change to the simulator, for three reasons:

  * the core physics stays untouched, so ``PERFECT`` is bit-identical to every
    number already in the writeup (pinned by ``test_perfect_link_is_a_no_op``);
  * it composes in the *deployment* order. On real hardware the safety filter
    runs on the ground station, downstream of the estimator and upstream of the
    radio, so ``LinkedController(ShieldedController(gate))`` is not merely
    convenient — it is the real topology, and it means the shield degrades with
    the estimate exactly as it would in the volume;
  * the parameters are not study knobs, they are *the spec of the real rig*.
    Tightening ``obs_latency_s`` means buying a faster capture pipeline.

Sign convention: latencies are in seconds and converted to control ticks inside
the buffers, so a profile stays meaningful if the control rate changes. That
matters here — at ``dt`` = 60 ms a whole realistic latency budget is under one
tick, which is itself a finding about the control rate we ship (see
``eval/robustness.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["LinkProfile", "LinkedController", "PERFECT", "VICON_LAB",
           "VICON_BUSY", "FIELD_DEGRADED", "PRESETS"]


@dataclass(frozen=True)
class LinkProfile:
    """One end-to-end link budget. All times in seconds, probabilities per tick."""

    name: str = "perfect"

    # --- perception: truth -> what the ground station believes ---------------
    obs_latency_s: float = 0.0      # capture + solve + network transport
    pos_noise_m: float = 0.0        # Vicon is sub-mm; larger for UWB/onboard
    vel_noise_mps: float = 0.0      # finite-differenced velocity is much noisier
    dropout_p: float = 0.0          # per-body per-tick chance of losing the marker
    dropout_burst: float = 1.0      # mean occlusion length, in ticks
    swap_p: float = 0.0             # chance of a rigid-body IDENTITY swap starting
    swap_burst: float = 1.0         # mean swap length, in ticks

    # --- command: ground station -> ELRS -> FC -> thrust ---------------------
    cmd_latency_s: float = 0.0      # uplink transport (zero-order hold, no interp)
    motor_tau_s: float = 0.0        # first-order thrust lag (spin-up), NOT transport
    packet_loss_p: float = 0.0      # uplink loss; the FC holds its last command
    thrust_scale: float = 1.0       # thrust-to-weight mismatch at full charge
    sag_frac_per_min: float = 0.0   # further thrust lost per minute of flight

    def is_perfect(self) -> bool:
        """True if this link changes nothing, so the wrapper can hard-bypass."""
        return (self.obs_latency_s == 0.0 and self.pos_noise_m == 0.0
                and self.vel_noise_mps == 0.0 and self.dropout_p == 0.0
                and self.swap_p == 0.0 and self.cmd_latency_s == 0.0
                and self.motor_tau_s == 0.0 and self.packet_loss_p == 0.0
                and self.thrust_scale == 1.0 and self.sag_frac_per_min == 0.0)


# A perfect link. Every pre-existing result in the repo was measured here, which
# is exactly the point: it is the top of the robustness curves, not a scenario.
PERFECT = LinkProfile(name="perfect")

# What a well-run Vicon volume with ELRS at 250 Hz should actually deliver.
# Position noise is sub-mm; the velocity figure is finite-difference noise, which
# is the term that matters because every gate predicate is velocity-derived.
VICON_LAB = LinkProfile(
    name="vicon_lab",
    obs_latency_s=0.008, pos_noise_m=0.0005, vel_noise_mps=0.03,
    dropout_p=0.005, dropout_burst=3.0,
    swap_p=0.0005, swap_burst=15.0,
    cmd_latency_s=0.006, motor_tau_s=0.020, packet_loss_p=0.005,
    thrust_scale=1.0, sag_frac_per_min=0.06,
)

# Same rig on a bad day: four identical airframes in a small volume means more
# occlusion and far more identity swapping, which is the interesting one.
VICON_BUSY = replace(
    VICON_LAB, name="vicon_busy",
    obs_latency_s=0.015, vel_noise_mps=0.06,
    dropout_p=0.02, dropout_burst=5.0,
    swap_p=0.004, swap_burst=25.0,
    packet_loss_p=0.02, sag_frac_per_min=0.12,
)

# No mocap: UWB-grade absolute position, longer radio, worse everything. This is
# the "fly it outside the lab" case and it is meant to look bad.
FIELD_DEGRADED = LinkProfile(
    name="field_degraded",
    obs_latency_s=0.040, pos_noise_m=0.15, vel_noise_mps=0.25,
    dropout_p=0.05, dropout_burst=8.0,
    swap_p=0.002, swap_burst=20.0,
    cmd_latency_s=0.020, motor_tau_s=0.030, packet_loss_p=0.05,
    thrust_scale=0.9, sag_frac_per_min=0.15,
)

PRESETS = {p.name: p for p in (PERFECT, VICON_LAB, VICON_BUSY, FIELD_DEGRADED)}


class _Burst:
    """A two-state Markov process: rare onsets, geometrically-distributed length.

    Occlusions and label swaps are *bursty* — that is the whole reason they are
    dangerous. A single corrupted frame is noise any controller shrugs off; the
    same total corruption delivered as one 25-tick identity swap is a controller
    chasing the wrong aircraft for a second and a half.
    """

    def __init__(self, p_on: float, mean_len: float, rng: np.random.Generator):
        self.p_on = float(p_on)
        self.p_stay = 0.0 if mean_len <= 1.0 else 1.0 - 1.0 / float(mean_len)
        self.rng = rng
        self.on = False

    def tick(self) -> bool:
        if self.on:
            self.on = self.rng.random() < self.p_stay
        else:
            self.on = self.rng.random() < self.p_on
        return self.on


class LinkedController:
    """Wraps a controller so it sees a degraded estimate and commands a real link.

    Composition is the deployment order::

        LinkedController(ShieldedController(GatedController(...)), profile)
        \\___ radio + motors ___/ \\__ ground station: shield below BT and policy __/

    so the safety filter is fed the *same* delayed, noisy estimate the tree and
    the policy get. That is deliberate: a shield running on a stale estimate can
    itself command a violation, and pretending otherwise would make the safety
    story look better than the hardware would.
    """

    def __init__(self, inner, profile: LinkProfile = PERFECT, *,
                 rng: np.random.Generator | None = None, seed: int | None = None,
                 dt: float = 0.06, name: str | None = None):
        self.inner = inner
        self.profile = profile
        self.dt = float(dt)
        # Two seeding modes. An explicit ``rng`` is a fixed stream (tests). A
        # ``seed`` re-derives the stream from (seed, episode index) on every
        # reset, so episode k sees the SAME occlusions and swaps no matter which
        # controller is being measured — the corruption is paired across the
        # comparison, not merely the spawns. Without that, a robustness delta at
        # small n is mostly a difference in which dropouts each arm happened to
        # draw.
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed or 0)
        self._fixed_rng = rng is not None
        self._ep = -1
        self.name = name or f"{getattr(inner, 'name', 'ctrl')}@{profile.name}"
        # Counters for the report: corruption you cannot see is corruption you
        # cannot attribute a lost episode to.
        self.n_dropout = 0
        self.n_swap = 0
        self.n_lost_packet = 0
        self.reset()

    @property
    def filter(self):
        """Expose the wrapped shield so violation metrics still find it.

        ``scenarios._violations`` looks for ``.filter`` on the controller it is
        handed. Linked(Shielded(gate)) must therefore forward, or every
        robustness run would silently report zero safety corrections.
        """
        return getattr(self.inner, "filter", None)

    # ------------------------------------------------------------------ setup
    def reset(self):
        self.inner.reset()
        p = self.profile
        self._ep += 1
        if not self._fixed_rng:
            self.rng = np.random.default_rng([self.seed or 0, self._ep])
        self._views: list = []          # history of true views, newest last
        self._cmds: list = []           # history of commanded actions
        self._held = None               # last estimate accepted (dropout hold)
        self._last_cmd = None           # last command the FC received (loss hold)
        self._thrust = None             # first-order motor state
        self._t = 0
        self._self_drop = None
        self._opp_drop = None
        self._swap = _Burst(p.swap_p, p.swap_burst, self.rng)
        self._swap_pair: tuple[int, int] | None = None
        self.last_estimate: dict | None = None
        self.n_dropout = self.n_swap = self.n_lost_packet = 0

    # ------------------------------------------------------------- perception
    def _delayed(self, view):
        """Interpolate the view history back by ``obs_latency_s``.

        Sub-tick interpolation rather than rounding to whole ticks: at dt=60 ms a
        realistic 8-40 ms budget would round to 0 or 1 and the whole latency axis
        would be a step function. Kinematic fields interpolate; discrete ones
        (alive, fuel, time) take the older sample, since you cannot half-observe
        a kill.
        """
        d = self.profile.obs_latency_s / self.dt
        if d <= 0:
            return view
        n = len(self._views)
        lo = min(int(math.floor(d)), n - 1)     # older
        hi = min(int(math.ceil(d)), n - 1)      # newer
        frac = d - math.floor(d)
        a = self._views[n - 1 - lo]
        b = self._views[n - 1 - hi]
        if a is b:
            return a
        # a is older than b; we want the state `d` ticks back, i.e. between them
        w = 1.0 - frac if lo != hi else 1.0
        mix = lambda x, y: w * np.asarray(x) + (1.0 - w) * np.asarray(y)
        return replace(
            a,
            self_pos=mix(a.self_pos, b.self_pos), self_vel=mix(a.self_vel, b.self_vel),
            opp_pos=mix(a.opp_pos, b.opp_pos), opp_vel=mix(a.opp_vel, b.opp_vel),
            asset_pos=None if a.asset_pos is None else mix(a.asset_pos, b.asset_pos),
            asset_vel=None if a.asset_vel is None else mix(a.asset_vel, b.asset_vel),
        )

    def _corrupt(self, view):
        p = self.profile
        n_self, n_opp = len(view.self_pos), len(view.opp_pos)
        if self._self_drop is None:
            self._self_drop = [_Burst(p.dropout_p, p.dropout_burst, self.rng)
                               for _ in range(n_self)]
            self._opp_drop = [_Burst(p.dropout_p, p.dropout_burst, self.rng)
                              for _ in range(n_opp)]

        self_pos = np.array(view.self_pos, dtype=np.float64, copy=True)
        self_vel = np.array(view.self_vel, dtype=np.float64, copy=True)
        opp_pos = np.array(view.opp_pos, dtype=np.float64, copy=True)
        opp_vel = np.array(view.opp_vel, dtype=np.float64, copy=True)

        if p.pos_noise_m > 0:
            self_pos += self.rng.normal(0, p.pos_noise_m, self_pos.shape)
            opp_pos += self.rng.normal(0, p.pos_noise_m, opp_pos.shape)
        if p.vel_noise_mps > 0:
            self_vel += self.rng.normal(0, p.vel_noise_mps, self_vel.shape)
            opp_vel += self.rng.normal(0, p.vel_noise_mps, opp_vel.shape)

        # Occlusion: the solver simply stops publishing that body, so the ground
        # station holds its last pose. Holding (not zeroing, not extrapolating) is
        # what every mocap bridge actually does.
        if self._held is not None:
            for i, b in enumerate(self._self_drop):
                if b.tick():
                    self_pos[i], self_vel[i] = self._held[0][i], self._held[1][i]
                    self.n_dropout += 1
            for j, b in enumerate(self._opp_drop):
                if b.tick():
                    opp_pos[j], opp_vel[j] = self._held[2][j], self._held[3][j]
                    self.n_dropout += 1
        else:
            for b in self._self_drop + self._opp_drop:
                b.tick()

        # Identity swap. Two 65 mm airframes with the same marker pattern crossing
        # inside a metre is the canonical way a Vicon solve mislabels, and it is
        # the failure this whole study exists to look at: the *positions* stay
        # perfectly accurate, only the labels are wrong, so nothing downstream can
        # detect it from residuals. The controller confidently chases the wrong
        # aircraft until the solve recovers.
        if n_opp >= 2:
            was_on = self._swap.on
            if self._swap.tick():
                if not was_on:
                    i, j = self.rng.choice(n_opp, size=2, replace=False)
                    self._swap_pair = (int(i), int(j))
                    self.n_swap += 1
                i, j = self._swap_pair
                opp_pos[[i, j]] = opp_pos[[j, i]]
                opp_vel[[i, j]] = opp_vel[[j, i]]
            else:
                self._swap_pair = None

        self._held = (self_pos.copy(), self_vel.copy(), opp_pos.copy(), opp_vel.copy())
        return replace(view, self_pos=self_pos, self_vel=self_vel,
                       opp_pos=opp_pos, opp_vel=opp_vel)

    # ----------------------------------------------------------------- command
    def _uplink(self, cmd: np.ndarray) -> np.ndarray:
        """Transport delay, packet loss, motor lag and thrust budget."""
        p = self.profile
        self._cmds.append(np.array(cmd, dtype=np.float64, copy=True))

        # Zero-order hold, not interpolation: a radio delivers whole frames.
        if p.cmd_latency_s > 0:
            k = int(round(p.cmd_latency_s / self.dt))
            sent = self._cmds[max(0, len(self._cmds) - 1 - k)]
        else:
            sent = self._cmds[-1]

        # A lost frame means the FC never hears the update and flies the last one.
        if p.packet_loss_p > 0 and self._last_cmd is not None \
                and self.rng.random() < p.packet_loss_p:
            self.n_lost_packet += 1
            sent = self._last_cmd
        self._last_cmd = sent

        # Motors are a first-order lag, not a delay: thrust ramps toward demand.
        if p.motor_tau_s > 0:
            alpha = 1.0 - math.exp(-self.dt / p.motor_tau_s)
            if self._thrust is None:
                self._thrust = np.zeros_like(sent)
            self._thrust = self._thrust + alpha * (sent - self._thrust)
            out = self._thrust
        else:
            out = sent

        scale = p.thrust_scale
        if p.sag_frac_per_min > 0:
            scale *= max(0.0, 1.0 - p.sag_frac_per_min * (self._t * self.dt) / 60.0)
        return np.clip(out * scale, -1.0, 1.0)

    # -------------------------------------------------------------------- API
    def act(self, view) -> np.ndarray:
        if self.profile.is_perfect():
            self.last_estimate = None
            return self.inner.act(view)     # bypass: no RNG draws, bit-identical
        self._views.append(view)
        if len(self._views) > 64:
            self._views.pop(0)
        est = self._corrupt(self._delayed(view))
        # Kept for the trace/explorer: the single most useful thing to render on
        # a degraded run is truth against *belief*, because the failures worth
        # understanding (a sustained label swap, a held pose) look completely
        # normal from inside the controller. It is only visible as a gap.
        self.last_estimate = {
            "self_pos": np.asarray(est.self_pos).tolist(),
            "opp_pos": np.asarray(est.opp_pos).tolist(),
            "swapped": self._swap_pair if self._swap.on else None,
        }
        cmd = np.asarray(self.inner.act(est), dtype=np.float64).reshape(-1, 3)
        self._t += 1
        return self._uplink(cmd).reshape(-1)

    def stats(self) -> dict:
        return {"dropouts": self.n_dropout, "swaps": self.n_swap,
                "lost_packets": self.n_lost_packet, "ticks": self._t}
