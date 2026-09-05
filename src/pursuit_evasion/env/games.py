"""Game modes: what the two teams are actually trying to do.

The core simulator is deliberately generic — two teams, one of which can
*neutralise* members of the other by getting close enough. A ``GameSpec`` layers
an objective on top of that mechanic, so the same physics, observation encoding,
BT gate, safety filter and instrumentation serve several different problems:

``tag``      the original 2v2 pursuit-evasion. Homogeneous drones. Pursuers must
             tag both evaders before the clock; a timeout is an evader win.

``assault``  **capture the flag / point defence.** Two *missiles* (fast, fixed
             forward thrust, limited lateral authority, finite burn) run in at a
             static asset from long range. Two *interceptor drones* (slower but
             holonomic) must kill them first. A timeout is a *defender* win —
             the attackers are the ones on a clock. Because the missiles are
             ~1.5x faster than the drones, a tail chase is impossible: the
             defenders have to fly the geometry and be on the path.

``escort``   the same matchup with a **moving** asset — a convoy transiting the
             arena at constant speed. Defenders win if it arrives. Now the
             defenders must trade "stay with the asset" against "push out and
             intercept", which the static-asset version never asks.

The pursuer/evader *slots* are reused for defenders/attackers: the neutralise
mechanic points the same way (a defender kills a missile the way a pursuer tags
an evader), so every controller, gate and metric carries over unchanged and only
the objective, dynamics and spawn geometry differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .dynamics import (DRONE_EVADER, DRONE_INTERCEPTOR, DRONE_PURSUER,
                       MISSILE_ATTACKER, MISSILE_LONG_BURN, DynamicsProfile)
from .mjcf import ArenaConfig


@dataclass
class EpisodeConfig:
    max_steps: int = 500
    control_repeat: int = 5          # physics substeps per env step
    capture_radius: float = 0.7
    spawn_radius: float = 10.0       # agents spawn within this radius (xy)
    spawn_z_lo: float = 1.5
    spawn_z_hi: float = 6.0
    # reward shaping
    capture_bonus: float = 15.0      # pursuer slot, per opponent neutralised
    all_captured_bonus: float = 30.0
    evade_survive_bonus: float = 30.0
    time_penalty: float = 0.01       # pursuer slot pays per step
    dist_shaping: float = 0.05       # coefficient on progress toward nearest opponent
    bounds_penalty: float = 0.02     # discourage hugging the walls
    # objective games only
    breach_bonus: float = 60.0       # attacker slot, for reaching the asset
    breach_penalty: float = 60.0     # defender slot, same event
    asset_shaping: float = 0.05      # attacker progress toward the asset
    arrive_bonus: float = 30.0       # defender slot, escort asset reaches its goal


@dataclass(frozen=True)
class AssetSpec:
    """The defended object. Static if ``goal is None``, otherwise it transits."""

    radius: float = 2.5                             # breach radius
    start: tuple[float, float, float] = (0.0, 0.0, 0.8)
    goal: tuple[float, float, float] | None = None
    speed: float = 0.0                              # m/s along start -> goal
    draw_radius: float = 1.2                        # visual geom size

    @property
    def moving(self) -> bool:
        return self.goal is not None and self.speed > 0.0

    def position_at(self, t: float) -> np.ndarray:
        """Asset position ``t`` seconds into the episode (constant-speed transit)."""
        p0 = np.array(self.start, dtype=np.float64)
        if not self.moving:
            return p0
        p1 = np.array(self.goal, dtype=np.float64)
        leg = p1 - p0
        total = float(np.linalg.norm(leg))
        s = min(self.speed * t, total)
        return p0 + leg * (s / total if total > 1e-9 else 0.0)

    def velocity_at(self, t: float) -> np.ndarray:
        if not self.moving or self.arrived_at(t):
            return np.zeros(3)
        p0 = np.array(self.start, dtype=np.float64)
        p1 = np.array(self.goal, dtype=np.float64)
        leg = p1 - p0
        return leg / max(float(np.linalg.norm(leg)), 1e-9) * self.speed

    def arrived_at(self, t: float) -> bool:
        if not self.moving:
            return False
        p0 = np.array(self.start, dtype=np.float64)
        p1 = np.array(self.goal, dtype=np.float64)
        return self.speed * t >= float(np.linalg.norm(p1 - p0)) - 1e-9


@dataclass(frozen=True)
class GameSpec:
    key: str
    pursuer_role: str                 # display name for the neutralising team
    evader_role: str                  # display name for the neutralised team
    arena: ArenaConfig
    episode: EpisodeConfig
    asset: AssetSpec | None = None
    timeout_winner: str = "evaders"   # who a full-length episode favours
    spawn_kind: str = "opposed"
    capture_substeps: bool = False    # check capture every physics substep (fast closers)

    @property
    def has_asset(self) -> bool:
        return self.asset is not None


# --------------------------------------------------------------------- specs

def _tag() -> GameSpec:
    return GameSpec(
        key="tag",
        pursuer_role="pursuers",
        evader_role="evaders",
        arena=ArenaConfig(),                       # drones 21/19, damping 4
        episode=EpisodeConfig(),
        asset=None,
        timeout_winner="evaders",
        spawn_kind="opposed",
        capture_substeps=False,
    )


_DEFENCE_ARENA = ArenaConfig(
    half_extent=40.0, z_min=0.5, z_max=20.0, agent_radius=0.5,
    pursuer_dyn=DRONE_INTERCEPTOR, evader_dyn=MISSILE_ATTACKER,
)


def _assault() -> GameSpec:
    return GameSpec(
        key="assault",
        pursuer_role="defenders",
        evader_role="attackers",
        arena=_DEFENCE_ARENA,
        episode=EpisodeConfig(
            max_steps=200, control_repeat=3, capture_radius=1.4,
            spawn_radius=34.0, spawn_z_lo=5.0, spawn_z_hi=15.0,
            time_penalty=0.0,
        ),
        asset=AssetSpec(radius=2.5, start=(0.0, 0.0, 0.8)),
        timeout_winner="pursuers",     # the missiles are the ones on a clock
        spawn_kind="air_defence",
        capture_substeps=True,         # closing speeds reach ~18 m/s
    )


def _escort() -> GameSpec:
    L = _DEFENCE_ARENA.half_extent
    return GameSpec(
        key="escort",
        pursuer_role="escorts",
        evader_role="attackers",
        arena=replace(_DEFENCE_ARENA, evader_dyn=MISSILE_LONG_BURN),
        episode=EpisodeConfig(
            max_steps=320, control_repeat=3, capture_radius=1.4,
            spawn_radius=0.8 * L, spawn_z_lo=5.0, spawn_z_hi=15.0,
            time_penalty=0.0,
        ),
        asset=AssetSpec(radius=2.0, start=(-0.8 * L, 0.0, 1.5),
                        goal=(0.8 * L, 0.0, 1.5), speed=4.0, draw_radius=1.6),
        timeout_winner="pursuers",
        spawn_kind="ambush",
        capture_substeps=True,
    )


_BUILDERS = {"tag": _tag, "assault": _assault, "escort": _escort}
GAME_KEYS = tuple(_BUILDERS)


def make_game(key: str = "tag") -> GameSpec:
    if key not in _BUILDERS:
        raise KeyError(f"unknown game {key!r}; known: {GAME_KEYS}")
    return _BUILDERS[key]()


TAG = _tag()


# -------------------------------------------------------------------- spawns

def sample_starts(game: GameSpec, rng: np.random.Generator,
                  pursuers: list[str], evaders: list[str]) -> dict[str, tuple[float, float, float]]:
    """Sample spawn positions for one episode of ``game``."""
    ep, arena = game.episode, game.arena
    starts: dict[str, tuple[float, float, float]] = {}

    if game.spawn_kind == "opposed":
        # tag: pursuers cluster on one side, evaders on the other (legible chase)
        for n in pursuers:
            ang = rng.uniform(-0.6, 0.6)
            r = rng.uniform(ep.spawn_radius * 0.6, ep.spawn_radius)
            z = rng.uniform(ep.spawn_z_lo, ep.spawn_z_hi)
            starts[n] = (-r * np.cos(ang), r * np.sin(ang), z)
        for n in evaders:
            ang = rng.uniform(-0.6, 0.6)
            r = rng.uniform(ep.spawn_radius * 0.6, ep.spawn_radius)
            z = rng.uniform(ep.spawn_z_lo, ep.spawn_z_hi)
            starts[n] = (r * np.cos(ang), r * np.sin(ang), z)
        return starts

    # objective games: attackers run in from long range on separated bearings,
    # defenders hold a combat air patrol near the thing they are protecting.
    asset = np.array(game.asset.start, dtype=np.float64)
    n_att = max(len(evaders), 1)
    if game.spawn_kind == "ambush":
        # escort: the raid lies in wait *downrange*, inside a forward arc about
        # the convoy's transit heading, so it is a head-on merge rather than a
        # stern chase the missiles could never be caught in.
        leg = np.array(game.asset.goal, dtype=np.float64) - asset
        base = float(np.arctan2(leg[1], leg[0]))
        arc = np.deg2rad(70.0)
    else:
        base, arc = rng.uniform(-np.pi, np.pi), None   # threats from any bearing
    for i, n in enumerate(evaders):
        # spread the raid across bearings so one defender cannot cover both
        if arc is None:
            off = 2 * np.pi * i / n_att
        else:
            off = arc * ((2 * i / (n_att - 1) - 1.0) if n_att > 1 else 0.0)
        ang = base + off + rng.uniform(-0.35, 0.35)
        r = rng.uniform(ep.spawn_radius * 0.85, ep.spawn_radius)
        z = rng.uniform(ep.spawn_z_lo, ep.spawn_z_hi)
        starts[n] = (float(asset[0] + r * np.cos(ang)),
                     float(asset[1] + r * np.sin(ang)), float(z))
    for i, n in enumerate(pursuers):
        ang = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(5.0, 9.0)
        z = rng.uniform(2.0, 8.0)
        starts[n] = (float(asset[0] + r * np.cos(ang)),
                     float(asset[1] + r * np.sin(ang)), float(z))

    # keep everything inside the joint ranges
    L, zlo, zhi = arena.half_extent, arena.z_min, arena.z_max
    for k, (x, y, z) in starts.items():
        starts[k] = (float(np.clip(x, -L + 1.0, L - 1.0)),
                     float(np.clip(y, -L + 1.0, L - 1.0)),
                     float(np.clip(z, zlo + 0.5, zhi - 0.5)))
    return starts
