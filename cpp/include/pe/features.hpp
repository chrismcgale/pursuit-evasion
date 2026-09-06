#pragma once
#include <algorithm>
#include <limits>
#include <vector>

#include "pe/scripted.hpp"
#include "pe/sim.hpp"
#include "pe/vec3.hpp"

namespace pe {

// Mirrors src/pursuit_evasion/bt/features.py:AgentFeatures
struct AgentFeatures {
  bool alive = false;
  double dist_nearest = std::numeric_limits<double>::infinity();
  double dist_second = std::numeric_limits<double>::infinity();
  int n_live_others = 0;
  double closing_rate = 0.0;
  double target_lateral = 0.0;
  bool intercept_ahead = false;
  bool contested = false;
  // --- objective games; inert (inf / true / 1.0) when the game has no asset ---
  double asset_dist = std::numeric_limits<double>::infinity();   // own dist to the asset
  double threat_time = std::numeric_limits<double>::infinity();  // secs until it is reached
  bool intercept_feasible = true;   // a positive lead-intercept solution exists
  double fuel = 1.0;                // own remaining burn fraction
  // Consecutive gate ticks intercept_feasible has been false. Not computable
  // from one view: GatedController maintains the per-agent streak and stamps it
  // in before ticking, so a one-tick infeasibility glitch can be debounced.
  int infeasible_ticks = 0;
};

// Exact mirror of features.py:_pairwise / agent_features (others == opponents).
inline AgentFeatures agent_features(const TeamView& v, int i) {
  AgentFeatures f;
  f.alive = v.self_alive[i];
  Vec3 pos = v.self_pos[i], vel = v.self_vel[i];

  std::vector<int> live;
  for (int j = 0; j < v.n_opp(); ++j)
    if (v.opp_alive[j]) live.push_back(j);
  if (live.empty()) return f;

  // sort live opponents by distance
  std::sort(live.begin(), live.end(), [&](int a, int b) {
    return norm(v.opp_pos[a] - pos) < norm(v.opp_pos[b] - pos);
  });
  int j0 = live[0];
  double dist0 = norm(v.opp_pos[j0] - pos);
  double dist1 = live.size() > 1 ? norm(v.opp_pos[live[1]] - pos)
                                 : std::numeric_limits<double>::infinity();

  Vec3 los = unit(v.opp_pos[j0] - pos);
  Vec3 rel_v = v.opp_vel[j0] - vel;
  double closing = -dot(rel_v, los);
  Vec3 tv = v.opp_vel[j0];
  Vec3 lateral_vec = tv - dot(tv, los) * los;
  double lateral = norm(lateral_vec);
  bool ahead = ((dot(los, unit(vel)) > 0.3) || norm(vel) < 1e-3) && dist0 > 1.2;
  bool contested = std::isfinite(dist1) && (dist1 - dist0) < 0.35 * std::max(dist0, 1e-6);

  f.dist_nearest = dist0;
  f.dist_second = dist1;
  f.n_live_others = (int)live.size();
  f.closing_rate = closing;
  f.target_lateral = lateral;
  f.intercept_ahead = ahead;
  f.contested = contested;

  // --- objective-game extras ------------------------------------------------
  if (v.has_asset) {
    f.asset_dist = norm(v.asset_pos - pos);
    f.threat_time = time_to_asset(v.opp_pos[j0], v.opp_vel[j0], v.asset_pos, v.opp_vmax);
    f.intercept_feasible =
        lead_intercept_time(v.opp_pos[j0] - pos, v.opp_vel[j0], v.vmax).has_value();
  }
  if (i < (int)v.self_fuel.size()) f.fuel = v.self_fuel[i];
  return f;
}

}  // namespace pe
