"""Core two-team simulator on top of MuJoCo.

This layer is agnostic to *who* controls each team and to *which game* is being
played. It steps the physics given a pursuer-slot action and an evader-slot
action, applies the neutralise mechanic, advances the defended asset (if the
game has one), and produces a ``TeamView`` for each side (structured geometric
state that both scripted controllers and the flat-observation builder consume)
plus a shaped reward.

Slot semantics (fixed) — the *pursuer slot* neutralises, the *evader slot* is
neutralised. Objectives (from ``GameSpec``) vary on top of that:

  * ``tag``     — pursuers must tag BOTH evaders before the time limit; a
                  timeout is an evader win.
  * ``assault`` — attackers (missiles, evader slot) must reach a static asset;
                  defenders (drones, pursuer slot) must kill them first. A
                  timeout is a *defender* win.
  * ``escort``  — same, with the asset transiting to a goal; defenders also win
                  if it arrives.

Two ways an evader-slot agent is neutralised: intercepted (a pursuer-slot agent
within ``capture_radius``) or **spent** (a missile that burned out and decayed
below stall speed). Both freeze it in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .dynamics import DynamicsProfile, apply_dynamics, is_spent
from .games import TAG, AssetSpec, EpisodeConfig, GameSpec, make_game, sample_starts
from .mjcf import ArenaConfig, agent_names, build_mjcf

TEAM_PURSUERS = "pursuers"
TEAM_EVADERS = "evaders"

__all__ = ["TEAM_PURSUERS", "TEAM_EVADERS", "ArenaConfig", "EpisodeConfig",
           "GameSpec", "AssetSpec", "TeamView", "StepResult", "PursuitEvasionCore"]


@dataclass
class TeamView:
    """Egocentric geometric state handed to a team's controller."""

    team: str
    self_pos: np.ndarray             # (n_self, 3)
    self_vel: np.ndarray             # (n_self, 3)
    self_alive: np.ndarray           # (n_self,) bool
    opp_pos: np.ndarray              # (n_opp, 3)
    opp_vel: np.ndarray              # (n_opp, 3)
    opp_alive: np.ndarray            # (n_opp,) bool
    time_frac: float                 # steps_done / max_steps in [0, 1]
    arena: ArenaConfig
    vmax: float                      # this team's max speed
    opp_vmax: float
    # --- objective games (defaults keep `tag` controllers untouched) ---------
    game: str = "tag"
    self_dyn: DynamicsProfile | None = None
    opp_dyn: DynamicsProfile | None = None
    self_fuel: np.ndarray | None = None    # (n_self,) remaining burn fraction in [0,1]
    opp_fuel: np.ndarray | None = None
    asset_pos: np.ndarray | None = None    # (3,) defended asset, None if the game has no asset
    asset_vel: np.ndarray | None = None
    asset_radius: float = 0.0

    @property
    def has_asset(self) -> bool:
        return self.asset_pos is not None


@dataclass
class StepResult:
    pursuer_view: TeamView
    evader_view: TeamView
    pursuer_reward: float
    evader_reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class PursuitEvasionCore:
    def __init__(self, arena: ArenaConfig | None = None, episode: EpisodeConfig | None = None,
                 seed: int | None = None, game: GameSpec | str | None = None):
        if isinstance(game, str):
            game = make_game(game)
        self.game: GameSpec = game or TAG
        # explicit arena/episode override the game's defaults (curricula, sweeps)
        self.arena = arena or self.game.arena
        self.ep = episode or self.game.episode
        self.rng = np.random.default_rng(seed)
        self.pursuers, self.evaders = agent_names(self.arena)
        self.p_dyn = self.arena.pursuer_dyn
        self.e_dyn = self.arena.evader_dyn
        # Build the MuJoCo model ONCE from a static arena (spawn refs = 0). Spawns
        # are applied per-episode by writing qpos in reset(); this keeps a single
        # canonical arena.xml that the C++ runtime loads for identical physics.
        self.model = mujoco.MjModel.from_xml_string(self.static_xml())
        self.data = mujoco.MjData(self.model)
        self._build_addr_cache()
        self.steps = 0
        self.captured = np.zeros(len(self.evaders), dtype=bool)
        self.spent = np.zeros(len(self.evaders), dtype=bool)
        self.burn_left = np.zeros(len(self.evaders), dtype=int)
        self._prev_min_dist = None
        self._prev_asset_dist = None
        self.breached = False
        self.asset_arrived = False

    # ------------------------------------------------------------------ model
    @property
    def dt(self) -> float:
        """Seconds per control step."""
        return self.arena.timestep * self.ep.control_repeat

    def static_xml(self) -> str:
        """Canonical MJCF with all spawn refs at zero (shared with the C++ runtime)."""
        zeros = {n: (0.0, 0.0, 0.0) for n in self.pursuers + self.evaders}
        return build_mjcf(self.arena, zeros, asset=self.game.asset)

    def _sample_starts(self) -> dict[str, tuple[float, float, float]]:
        return sample_starts(self.game, self.rng, self.pursuers, self.evaders)

    # -------------------------------------------------------------- accessors
    def _pos(self, name: str) -> np.ndarray:
        return self.data.qpos[self._qadr[name]].copy()

    def _vel(self, name: str) -> np.ndarray:
        return self.data.qvel[self._qadr[name]].copy()

    def _apply(self, name: str, ctrl: np.ndarray):
        for aid, c in zip(self._act_ids[name], ctrl):
            self.data.ctrl[aid] = float(np.clip(c, -1.0, 1.0))

    # ------------------------------------------------------------------- reset
    def reset(self, seed: int | None = None) -> tuple[TeamView, TeamView]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.reset_with_starts(self._sample_starts())

    def reset_with_starts(self, starts: dict[str, tuple[float, float, float]]
                          ) -> tuple[TeamView, TeamView]:
        """Reset to explicit spawn positions (used for failure replay and C++ parity)."""
        mujoco.mj_resetData(self.model, self.data)
        for name, xyz in starts.items():
            for adr, v in zip(self._qadr[name], xyz):
                self.data.qpos[adr] = float(v)
        self.data.qvel[:] = 0.0
        self.steps = 0
        self.captured = np.zeros(len(self.evaders), dtype=bool)
        self.spent = np.zeros(len(self.evaders), dtype=bool)
        self.burn_left = np.full(len(self.evaders), self.e_dyn.burn_steps, dtype=int)
        self.breached = False
        self.asset_arrived = False
        self._sync_asset()
        mujoco.mj_forward(self.model, self.data)
        self._prev_min_dist = self._nearest_pursuer_evader_dist()
        self._prev_asset_dist = self._nearest_attacker_asset_dist()
        self.last_starts = {k: tuple(map(float, v)) for k, v in starts.items()}
        return self._view(TEAM_PURSUERS), self._view(TEAM_EVADERS)

    def _build_addr_cache(self):
        self._qadr = {}
        self._act_ids = {}
        for name in self.pursuers + self.evaders:
            self._qadr[name] = [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_{ax}")]
                for ax in ("x", "y", "z")
            ]
            self._act_ids[name] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_{ax}")
                for ax in ("x", "y", "z")
            ]

    # --------------------------------------------------------------- the asset
    def asset_position(self) -> np.ndarray | None:
        if self.game.asset is None:
            return None
        return self.game.asset.position_at(self.steps * self.dt)

    def asset_velocity(self) -> np.ndarray | None:
        if self.game.asset is None:
            return None
        return self.game.asset.velocity_at(self.steps * self.dt)

    def _sync_asset(self):
        """Write the asset's kinematic pose into the mocap slot (viz + C++ parity)."""
        p = self.asset_position()
        if p is not None and self.model.nmocap > 0:
            self.data.mocap_pos[0] = p

    # -------------------------------------------------------------------- step
    def step(self, pursuer_action: np.ndarray, evader_action: np.ndarray) -> StepResult:
        pa = np.asarray(pursuer_action, dtype=np.float64).reshape(len(self.pursuers), 3)
        ea = np.asarray(evader_action, dtype=np.float64).reshape(len(self.evaders), 3)

        # Airframe transform: the command a controller emits is a *desire*; what
        # the vehicle can actually push is decided by its DynamicsProfile.
        for i, n in enumerate(self.pursuers):
            self._apply(n, apply_dynamics(self.p_dyn, self._vel(n), pa[i], 1))
        for i, n in enumerate(self.evaders):
            if self.captured[i]:
                self._apply(n, np.zeros(3))       # neutralised: freeze
            else:
                self._apply(n, apply_dynamics(self.e_dyn, self._vel(n), ea[i],
                                              int(self.burn_left[i])))

        newly = 0
        for _ in range(self.ep.control_repeat):
            mujoco.mj_step(self.model, self.data)
            if self.game.capture_substeps:
                # A defender is SLOWER than a missile, so it can never convert to
                # a stern chase — nearly every engagement is a high-offset
                # crossing pass, and the chord cut through the capture sphere is
                # far shorter than its diameter. Those grazing chords fit between
                # two consecutive control-step samples.
                #
                # Note it is NOT simply "missiles are fast": per control step tag
                # closes 1.43 capture radii and assault only 0.79. Tag stays
                # once-per-step because its 40-50% scripted baseline is
                # calibrated on that (CLAUDE.md).
                newly += self._update_captures()

        self.steps += 1
        self.burn_left = np.maximum(self.burn_left - 1, -1)
        self._sync_asset()

        if not self.game.capture_substeps:
            newly += self._update_captures()
        newly += self._update_spent()

        min_dist = self._nearest_pursuer_evader_dist()
        asset_dist = self._nearest_attacker_asset_dist()
        breach = self._check_breach()
        arrived = self._check_arrived()

        p_rew, e_rew = self._rewards(newly, min_dist, asset_dist, breach, arrived)
        self._prev_min_dist = min_dist
        self._prev_asset_dist = asset_dist

        all_captured = bool(self.captured.all())
        timeout = self.steps >= self.ep.max_steps
        terminated = all_captured or breach or arrived
        truncated = timeout and not terminated

        if all_captured:
            p_rew += self.ep.all_captured_bonus
        if truncated and self.game.timeout_winner == TEAM_EVADERS:
            e_rew += self.ep.evade_survive_bonus

        pursuer_win = bool(all_captured or arrived
                           or (truncated and self.game.timeout_winner == TEAM_PURSUERS))

        info = {
            "captured": self.captured.copy(),
            "n_captured": int(self.captured.sum()),
            "n_spent": int(self.spent.sum()),
            "min_dist": float(min_dist),
            "all_captured": all_captured,
            "timeout": timeout,
            "steps": self.steps,
            "game": self.game.key,
            "breach": bool(breach),
            "asset_arrived": bool(arrived),
            "asset_dist": float(asset_dist),
            "pursuer_win": pursuer_win,
        }
        return StepResult(
            self._view(TEAM_PURSUERS), self._view(TEAM_EVADERS),
            p_rew, e_rew, terminated, truncated, info,
        )

    # -------------------------------------------------------------- mechanics
    def _update_captures(self) -> int:
        newly = 0
        p_pos = np.array([self._pos(n) for n in self.pursuers])
        for j, ev in enumerate(self.evaders):
            if self.captured[j]:
                continue
            e_pos = self._pos(ev)
            d = np.linalg.norm(p_pos - e_pos, axis=1)
            if d.min() < self.ep.capture_radius:
                self.captured[j] = True
                newly += 1
        return newly

    def _update_spent(self) -> int:
        """A burned-out missile that has decayed below stall speed is out of the fight."""
        newly = 0
        for j, ev in enumerate(self.evaders):
            if self.captured[j]:
                continue
            if is_spent(self.e_dyn, self._vel(ev), int(self.burn_left[j])):
                self.captured[j] = True
                self.spent[j] = True
                newly += 1
        return newly

    def _check_breach(self) -> bool:
        """Any live attacker inside the asset's breach radius ends it (attacker win)."""
        if self.game.asset is None or self.breached:
            return self.breached
        a = self.asset_position()
        for j, ev in enumerate(self.evaders):
            if self.captured[j]:
                continue
            if float(np.linalg.norm(self._pos(ev) - a)) < self.game.asset.radius:
                self.breached = True
                break
        return self.breached

    def _check_arrived(self) -> bool:
        if self.game.asset is None or self.breached:
            return False
        self.asset_arrived = self.game.asset.arrived_at(self.steps * self.dt)
        return self.asset_arrived

    def _nearest_pursuer_evader_dist(self) -> float:
        p_pos = np.array([self._pos(n) for n in self.pursuers])
        live = [self._pos(self.evaders[j]) for j in range(len(self.evaders)) if not self.captured[j]]
        if not live:
            return 0.0
        e_pos = np.array(live)
        # min over all pursuer-evader pairs
        d = np.linalg.norm(p_pos[:, None, :] - e_pos[None, :, :], axis=2)
        return float(d.min())

    def _nearest_attacker_asset_dist(self) -> float:
        if self.game.asset is None:
            return float("nan")
        a = self.asset_position()
        live = [self._pos(self.evaders[j]) for j in range(len(self.evaders)) if not self.captured[j]]
        if not live:
            return float("inf")
        return float(min(np.linalg.norm(p - a) for p in live))

    def _rewards(self, newly: int, min_dist: float, asset_dist: float,
                 breach: bool, arrived: bool) -> tuple[float, float]:
        ep = self.ep
        # progress shaping: pursuers rewarded for closing the nearest gap
        progress = (self._prev_min_dist - min_dist) if self._prev_min_dist is not None else 0.0
        p_rew = ep.dist_shaping * progress - ep.time_penalty + ep.capture_bonus * newly
        e_rew = -ep.dist_shaping * progress - ep.capture_bonus * newly

        if self.game.asset is not None:
            # attackers are additionally rewarded for closing on the asset
            prev, cur = self._prev_asset_dist, asset_dist
            if prev is not None and np.isfinite(prev) and np.isfinite(cur):
                e_rew += ep.asset_shaping * (prev - cur)
            if breach:
                e_rew += ep.breach_bonus
                p_rew -= ep.breach_penalty
            if arrived:
                p_rew += ep.arrive_bonus

        # mild wall-avoidance so agents don't learn to pin themselves in a corner
        L = self.arena.half_extent
        for n in self.pursuers:
            p = self._pos(n)
            if max(abs(p[0]), abs(p[1])) > 0.9 * L:
                p_rew -= ep.bounds_penalty
        for j, n in enumerate(self.evaders):
            if self.captured[j]:
                continue
            p = self._pos(n)
            if max(abs(p[0]), abs(p[1])) > 0.9 * L:
                e_rew -= ep.bounds_penalty
        return p_rew, e_rew

    # --------------------------------------------------------------- team view
    def _fuel_frac(self) -> np.ndarray:
        n = self.e_dyn.burn_steps
        if n <= 0:
            return np.ones(len(self.evaders))
        return np.clip(self.burn_left / float(n), 0.0, 1.0)

    def _view(self, team: str) -> TeamView:
        fuel = self._fuel_frac()
        if team == TEAM_PURSUERS:
            self_names, opp_names = self.pursuers, self.evaders
            self_alive = np.ones(len(self.pursuers), dtype=bool)
            opp_alive = ~self.captured
            vmax, opp_vmax = self.arena.pursuer_vmax, self.arena.evader_vmax
            self_dyn, opp_dyn = self.p_dyn, self.e_dyn
            self_fuel, opp_fuel = np.ones(len(self.pursuers)), fuel
        else:
            self_names, opp_names = self.evaders, self.pursuers
            self_alive = ~self.captured
            opp_alive = np.ones(len(self.pursuers), dtype=bool)
            vmax, opp_vmax = self.arena.evader_vmax, self.arena.pursuer_vmax
            self_dyn, opp_dyn = self.e_dyn, self.p_dyn
            self_fuel, opp_fuel = fuel, np.ones(len(self.pursuers))
        return TeamView(
            team=team,
            self_pos=np.array([self._pos(n) for n in self_names]),
            self_vel=np.array([self._vel(n) for n in self_names]),
            self_alive=self_alive,
            opp_pos=np.array([self._pos(n) for n in opp_names]),
            opp_vel=np.array([self._vel(n) for n in opp_names]),
            opp_alive=opp_alive,
            time_frac=self.steps / self.ep.max_steps,
            arena=self.arena,
            vmax=vmax,
            opp_vmax=opp_vmax,
            game=self.game.key,
            self_dyn=self_dyn,
            opp_dyn=opp_dyn,
            self_fuel=self_fuel,
            opp_fuel=opp_fuel,
            asset_pos=self.asset_position(),
            asset_vel=self.asset_velocity(),
            asset_radius=self.game.asset.radius if self.game.asset else 0.0,
        )
