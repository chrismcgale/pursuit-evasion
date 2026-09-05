#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "pe/config.hpp"
#include "pe/dynamics.hpp"
#include "pe/vec3.hpp"

// Exact mirror of src/pursuit_evasion/env/games.py.
//
// The simulator is generic: two teams, one of which can *neutralise* members of
// the other by getting close enough. A GameSpec layers an objective on top, so
// the same physics, observation encoding, BT gate, safety filter and metrics
// serve several problems:
//
//   tag      — the original 2v2 chase. Homogeneous drones; timeout = evader win.
//   assault  — point defence. Two missiles run in at a static asset; two slower
//              interceptor drones must kill them first. Timeout = defender win.
//   escort   — same matchup, moving asset (a convoy in transit).
//
// The pursuer/evader slots are reused for defenders/attackers, so every
// controller and metric carries over and only the objective, dynamics and spawn
// geometry differ.
namespace pe {

// Mirrors games.py:AssetSpec. Static if `moving` is false, else a constant-speed
// transit from `start` to `goal`.
struct AssetSpec {
  double radius = 2.5;               // breach radius
  Vec3 start{0.0, 0.0, 0.8};
  Vec3 goal{0.0, 0.0, 0.8};
  double speed = 0.0;                // m/s along start -> goal
  double draw_radius = 1.2;          // visual geom size
  bool has_goal = false;

  bool moving() const { return has_goal && speed > 0.0; }

  double leg_length() const { return norm(goal - start); }

  Vec3 position_at(double t) const {
    if (!moving()) return start;
    const double total = leg_length();
    if (total <= 1e-9) return start;
    const double s = std::min(speed * t, total);
    return start + (goal - start) * (s / total);
  }

  Vec3 velocity_at(double t) const {
    if (!moving() || arrived_at(t)) return {0, 0, 0};
    return unit(goal - start) * speed;
  }

  bool arrived_at(double t) const {
    if (!moving()) return false;
    return speed * t >= leg_length() - 1e-9;
  }
};

// Mirrors games.py:GameSpec
struct GameSpec {
  std::string key = "tag";
  std::string pursuer_role = "pursuers";   // display name for the neutralising team
  std::string evader_role = "evaders";
  ArenaConfig arena{};
  EpisodeConfig episode{};
  bool has_asset = false;
  AssetSpec asset{};
  Team timeout_winner = Team::Evaders;     // who a full-length episode favours
  std::string spawn_kind = "opposed";
  bool capture_substeps = false;           // check capture every physics substep
};

// --------------------------------------------------------------------- specs

inline GameSpec game_tag() {
  GameSpec g;
  g.key = "tag";
  g.arena = ArenaConfig{};                 // drones 21/19, damping 4
  g.episode = EpisodeConfig{};
  g.has_asset = false;
  g.timeout_winner = Team::Evaders;
  g.spawn_kind = "opposed";
  g.capture_substeps = false;
  return g;
}

// Mirrors games.py:_DEFENCE_ARENA
inline ArenaConfig defence_arena() {
  ArenaConfig a;
  a.half_extent = 40.0;
  a.z_min = 0.5;
  a.z_max = 20.0;
  a.agent_radius = 0.5;
  a.pursuer_dyn = drone_interceptor();
  a.evader_dyn = missile_attacker();
  return a;
}

inline GameSpec game_assault() {
  GameSpec g;
  g.key = "assault";
  g.pursuer_role = "defenders";
  g.evader_role = "attackers";
  g.arena = defence_arena();
  g.episode.max_steps = 200;
  g.episode.control_repeat = 3;
  g.episode.capture_radius = 1.4;
  g.episode.spawn_radius = 34.0;
  g.episode.spawn_z_lo = 5.0;
  g.episode.spawn_z_hi = 15.0;
  g.has_asset = true;
  g.asset = AssetSpec{2.5, Vec3{0.0, 0.0, 0.8}, Vec3{0.0, 0.0, 0.8}, 0.0, 1.2, false};
  g.timeout_winner = Team::Pursuers;       // the missiles are the ones on a clock
  g.spawn_kind = "air_defence";
  g.capture_substeps = true;               // crossing passes, see sim.cpp:step
  return g;
}

inline GameSpec game_escort() {
  const double L = 40.0;
  GameSpec g;
  g.key = "escort";
  g.pursuer_role = "escorts";
  g.evader_role = "attackers";
  g.arena = defence_arena();
  g.arena.evader_dyn = missile_long_burn();
  g.episode.max_steps = 320;
  g.episode.control_repeat = 3;
  g.episode.capture_radius = 1.4;
  g.episode.spawn_radius = 0.8 * L;
  g.episode.spawn_z_lo = 5.0;
  g.episode.spawn_z_hi = 15.0;
  g.has_asset = true;
  g.asset = AssetSpec{2.0, Vec3{-0.8 * L, 0.0, 1.5}, Vec3{0.8 * L, 0.0, 1.5},
                      4.0, 1.6, true};
  g.timeout_winner = Team::Pursuers;
  g.spawn_kind = "ambush";
  g.capture_substeps = true;
  return g;
}

inline std::vector<std::string> game_keys() { return {"tag", "assault", "escort"}; }

inline GameSpec make_game(const std::string& key) {
  if (key == "tag") return game_tag();
  if (key == "assault") return game_assault();
  if (key == "escort") return game_escort();
  throw std::runtime_error("unknown game '" + key + "'; known: tag, assault, escort");
}

// Which arena.xml a game loads (generated from Python; see CLAUDE.md).
inline std::string arena_asset_name(const std::string& key) {
  return key == "tag" ? "arena.xml" : "arena_" + key + ".xml";
}

}  // namespace pe
