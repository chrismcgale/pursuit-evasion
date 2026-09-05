"""Programmatic MuJoCo (MJCF) model for the pursuit-evasion / air-defence arena.

Each agent is a point mass that can translate in 3D via three nested slide
joints (x, y, z). Joint damping provides aerodynamic-style drag, so a constant
control force yields a bounded terminal speed (v_max = gear / damping). Which
*direction* an agent may push is decided one layer up, by its
``DynamicsProfile`` (see ``dynamics.py``): a drone commands thrust freely, a
missile only laterally. The MJCF just carries each side's gear and damping.

The world coordinates of an agent are exactly (qpos_x, qpos_y, qpos_z) because
the slide joints lie along the world axes and are nested at the same origin.

Objective games additionally place a **mocap** body for the defended asset: it
has no joints and no dynamics, its pose is written directly each step, so it is
purely kinematic and trivially identical across the two runtimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dynamics import DRONE_EVADER, DRONE_PURSUER, DynamicsProfile


@dataclass(frozen=True)
class ArenaConfig:
    half_extent: float = 12.0      # arena is [-L, L] in x/y
    z_min: float = 0.5             # floor clearance
    z_max: float = 12.0            # ceiling
    agent_radius: float = 0.35
    agent_mass: float = 1.0
    timestep: float = 0.02         # physics dt (s)
    # Per-side airframes. The defaults are the original tag balance (gears
    # 21/19, damping 4 -> v_max 5.25 / 4.75): tuned so scripted pursuers catch
    # ~40% of the time — enough of a speed edge to make capture possible, small
    # enough to leave headroom for a learned or BT-gated policy to visibly beat
    # (or fall short of) the scripted baseline.
    pursuer_dyn: DynamicsProfile = field(default=DRONE_PURSUER)
    evader_dyn: DynamicsProfile = field(default=DRONE_EVADER)
    n_pursuers: int = 2
    n_evaders: int = 2

    @property
    def pursuer_gear(self) -> float:
        return self.pursuer_dyn.gear

    @property
    def evader_gear(self) -> float:
        return self.evader_dyn.gear

    @property
    def damping(self) -> float:
        return self.pursuer_dyn.damping

    @property
    def pursuer_vmax(self) -> float:
        return self.pursuer_dyn.vmax

    @property
    def evader_vmax(self) -> float:
        return self.evader_dyn.vmax


_PURSUER_RGBA = "0.85 0.25 0.20 1"
_EVADER_RGBA = "0.20 0.55 0.85 1"
_ASSET_RGBA = "0.95 0.80 0.20 1"


def _agent_body(name: str, cfg: ArenaConfig, rgba: str, start: tuple[float, float, float],
                dyn: DynamicsProfile) -> str:
    L = cfg.half_extent
    d = dyn.damping
    sx, sy, sz = start
    zr = f"{cfg.z_min} {cfg.z_max}"
    xr = f"{-L} {L}"
    # ref sets the joint's zero so the agent spawns at `start`.
    # Intermediate (x, y) bodies carry a joint so MuJoCo requires them to have
    # mass/inertia; keep it negligible so the real mass lives on the z geom.
    tiny = '<inertial pos="0 0 0" mass="1e-3" diaginertia="1e-5 1e-5 1e-5"/>'
    return f"""
    <body name="{name}_x" pos="0 0 0">
      <joint name="{name}_x" type="slide" axis="1 0 0" range="{xr}" damping="{d}" ref="{sx}"/>
      {tiny}
      <body name="{name}_y" pos="0 0 0">
        <joint name="{name}_y" type="slide" axis="0 1 0" range="{xr}" damping="{d}" ref="{sy}"/>
        {tiny}
        <body name="{name}_z" pos="0 0 0">
          <joint name="{name}_z" type="slide" axis="0 0 1" range="{zr}" damping="{d}" ref="{sz}"/>
          <geom name="{name}" type="sphere" size="{cfg.agent_radius}" mass="{cfg.agent_mass}"
                rgba="{rgba}" contype="0" conaffinity="0"/>
          <site name="{name}_site" pos="0 0 0" size="0.05"/>
        </body>
      </body>
    </body>"""


def _actuators(name: str, gear: float) -> str:
    return "".join(
        f'\n    <motor name="{name}_{ax}" joint="{name}_{ax}" gear="{gear}" ctrlrange="-1 1"/>'
        for ax in ("x", "y", "z")
    )


def agent_names(cfg: ArenaConfig) -> tuple[list[str], list[str]]:
    pursuers = [f"pursuer{i}" for i in range(cfg.n_pursuers)]
    evaders = [f"evader{i}" for i in range(cfg.n_evaders)]
    return pursuers, evaders


def _asset_body(asset, cfg: ArenaConfig) -> str:
    """Kinematic (mocap) body for a defended asset: no joints, pose written directly."""
    x, y, z = asset.start
    return f"""
    <body name="asset" mocap="true" pos="{x} {y} {z}">
      <geom name="asset" type="sphere" size="{asset.draw_radius}" rgba="{_ASSET_RGBA}"
            contype="0" conaffinity="0"/>
      <site name="asset_site" pos="0 0 0" size="0.05"/>
    </body>"""


def build_mjcf(cfg: ArenaConfig, starts: dict[str, tuple[float, float, float]],
               asset=None) -> str:
    """Return the MJCF XML string for the given start positions."""
    pursuers, evaders = agent_names(cfg)
    bodies, acts = [], []
    for n in pursuers:
        bodies.append(_agent_body(n, cfg, _PURSUER_RGBA, starts[n], cfg.pursuer_dyn))
        acts.append(_actuators(n, cfg.pursuer_gear))
    for n in evaders:
        bodies.append(_agent_body(n, cfg, _EVADER_RGBA, starts[n], cfg.evader_dyn))
        acts.append(_actuators(n, cfg.evader_gear))
    if asset is not None:
        bodies.append(_asset_body(asset, cfg))

    L = cfg.half_extent
    return f"""<mujoco model="pursuit_evasion">
  <compiler angle="radian"/>
  <option timestep="{cfg.timestep}" integrator="RK4" gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="{L} {L} 0.1" material="grid" pos="0 0 0"/>
    <light pos="0 0 {cfg.z_max}" dir="0 0 -1" directional="true"/>
    {"".join(bodies)}
  </worldbody>
  <actuator>{"".join(acts)}
  </actuator>
</mujoco>
"""
