"""Per-role flight dynamics: how a *command* becomes actuator forces.

Every agent is still a drag-limited point mass with three slide joints, but the
map from the controller's command (a desired 3D direction in [-1, 1]^3) to the
motor inputs depends on the airframe. That map is where heterogeneity lives:

* ``holonomic`` (**drone**) — the command *is* the thrust vector. A drone can
  hover, reverse instantly, and turn with zero radius. Cheap to control, slow.

* ``missile`` — thrust is fixed along the body axis (taken to be the velocity
  vector) while the motor burns; the command can only be applied *laterally*, and
  only up to ``lat_authority`` of full thrust. That gives a finite turn rate
  ``omega = a_lat / v``, so a fast missile turns wide: radius ``v^2 / a_lat``.
  After ``burn_steps`` the motor cuts out — fins still bite (lateral authority
  persists) but the airframe only decelerates, and below ``v_stall`` it is spent.

``apply_dynamics`` is a **pure function** and is mirrored line-for-line in
``cpp/include/pe/dynamics.hpp``. Changing it means changing both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KIND_HOLONOMIC = "holonomic"
KIND_MISSILE = "missile"


@dataclass(frozen=True)
class DynamicsProfile:
    name: str
    kind: str = KIND_HOLONOMIC
    gear: float = 21.0             # peak thrust (N); v_max = gear / damping
    damping: float = 4.0           # joint damping == aerodynamic drag
    lat_authority: float = 1.0     # missile: max lateral command as a fraction of thrust
    burn_steps: int = 0            # missile: powered control steps (0 => unlimited)
    v_stall: float = 0.0           # missile: below this speed after burnout it is spent
    v_align: float = 0.5           # missile: speed below which there is no body axis yet

    @property
    def vmax(self) -> float:
        return self.gear / self.damping

    @property
    def lat_accel(self) -> float:
        """Peak lateral acceleration (m/s^2) at unit mass — sets the turn radius."""
        return self.lat_authority * self.gear

    def turn_radius(self, speed: float | None = None) -> float:
        v = self.vmax if speed is None else speed
        return float("inf") if self.lat_accel <= 0 else v * v / self.lat_accel


# ---------------------------------------------------------------- presets
# The tag game's original balance, expressed as profiles (gears 21/19, damping 4
# -> v_max 5.25 / 4.75). Do not retune these: existing results depend on them.
DRONE_PURSUER = DynamicsProfile("drone_pursuer", KIND_HOLONOMIC, gear=21.0, damping=4.0)
DRONE_EVADER = DynamicsProfile("drone_evader", KIND_HOLONOMIC, gear=19.0, damping=4.0)

# Air-defence roles. The interceptor is agile but slow; the missile is ~1.5x
# faster and cannot turn inside ~6 m at top speed, which is the entire game.
DRONE_INTERCEPTOR = DynamicsProfile("drone_interceptor", KIND_HOLONOMIC, gear=30.0, damping=4.0)
MISSILE_ATTACKER = DynamicsProfile(
    "missile", KIND_MISSILE, gear=44.0, damping=4.0,
    lat_authority=0.45, burn_steps=120, v_stall=2.0,
)
# Escort attackers fly longer (the convoy transit is a longer episode).
MISSILE_LONG_BURN = DynamicsProfile(
    "missile_long", KIND_MISSILE, gear=44.0, damping=4.0,
    lat_authority=0.45, burn_steps=320, v_stall=2.0,
)

PROFILES = {p.name: p for p in (DRONE_PURSUER, DRONE_EVADER, DRONE_INTERCEPTOR,
                                MISSILE_ATTACKER, MISSILE_LONG_BURN)}


def _clip_norm(v: np.ndarray, lim: float = 1.0) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * (lim / n) if n > lim else v


def apply_dynamics(dyn: DynamicsProfile, vel: np.ndarray, cmd: np.ndarray,
                   burn_left: int) -> np.ndarray:
    """Map a controller command to actuator input for this airframe.

    Returns a vector with norm <= 1 (the actuators are ``ctrlrange="-1 1"`` and
    scaled by ``gear``). ``burn_left`` is the agent's remaining powered steps and
    is ignored for holonomic airframes.
    """
    c = _clip_norm(np.asarray(cmd, dtype=np.float64).reshape(3))
    if dyn.kind == KIND_HOLONOMIC:
        return c

    powered = dyn.burn_steps <= 0 or burn_left > 0
    speed = float(np.linalg.norm(vel))
    if speed < dyn.v_align:
        # Not yet flying: the launcher points it wherever it is commanded.
        return c if powered else np.zeros(3)

    fwd = np.asarray(vel, dtype=np.float64).reshape(3) / speed
    lat = c - float(c @ fwd) * fwd
    lat = _clip_norm(lat, dyn.lat_authority)
    out = (fwd if powered else np.zeros(3)) + lat
    return _clip_norm(out)


def is_spent(dyn: DynamicsProfile, vel: np.ndarray, burn_left: int) -> bool:
    """True once a missile has burned out and decayed below stall speed."""
    if dyn.kind != KIND_MISSILE or dyn.burn_steps <= 0 or dyn.v_stall <= 0:
        return False
    return burn_left <= 0 and float(np.linalg.norm(vel)) < dyn.v_stall
