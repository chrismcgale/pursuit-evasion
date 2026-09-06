#pragma once
#include <string>
#include <vector>

#include "pe/dynamics.hpp"

namespace pe {

enum class Team { Pursuers, Evaders };
enum class Mode { Scripted, RL };

// Mirrors src/pursuit_evasion/env/mjcf.py:ArenaConfig.
// Gear/damping/vmax are *derived* from the per-role dynamics profiles, exactly
// as on the Python side, so the default arena is byte-identical to arena.xml.
struct ArenaConfig {
  double half_extent = 12.0;
  double z_min = 0.5;
  double z_max = 12.0;
  double agent_radius = 0.35;
  double timestep = 0.02;
  DynamicsProfile pursuer_dyn = drone_pursuer();
  DynamicsProfile evader_dyn = drone_evader();
  int n_pursuers = 2;
  int n_evaders = 2;
  double pursuer_gear() const { return pursuer_dyn.gear; }
  double evader_gear() const { return evader_dyn.gear; }
  double damping() const { return pursuer_dyn.damping; }
  double pursuer_vmax() const { return pursuer_dyn.vmax(); }
  double evader_vmax() const { return evader_dyn.vmax(); }
};

// Mirrors src/pursuit_evasion/env/games.py:EpisodeConfig (dynamics-relevant fields)
struct EpisodeConfig {
  int max_steps = 500;
  int control_repeat = 5;
  double capture_radius = 0.7;
  double spawn_radius = 10.0;
  double spawn_z_lo = 1.5;
  double spawn_z_hi = 6.0;
};

// Mirrors src/pursuit_evasion/safety.py:SafetyConfig
struct SafetyConfig {
  double geofence_margin = 0.75;
  double z_margin = 0.75;
  double speed_limit_frac = 0.98;
  double correction_gain = 2.0;
};

// Mirrors src/pursuit_evasion/bt/gating.py:GateThresholds
struct GateThresholds {
  double close_quarters = 3.0;
  double juke_lateral = 2.2;
  double juke_min_dist = 6.0;
  double closing_fast = 3.0;
  double evader_danger = 4.0;
  // objective games
  double defender_close = 6.0;    // defender: engagement range (bigger arena)
  double threat_imminent = 1.6;   // defender: seconds-to-asset that means "now"
  double committed_range = 8.0;   // attacker: inside this it is a ballistic run-in
};

inline std::vector<std::string> pursuer_names(const ArenaConfig& c) {
  std::vector<std::string> v;
  for (int i = 0; i < c.n_pursuers; ++i) v.push_back("pursuer" + std::to_string(i));
  return v;
}
inline std::vector<std::string> evader_names(const ArenaConfig& c) {
  std::vector<std::string> v;
  for (int i = 0; i < c.n_evaders; ++i) v.push_back("evader" + std::to_string(i));
  return v;
}

}  // namespace pe
