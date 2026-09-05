#pragma once
#include <vector>

#include "pe/sim.hpp"
#include "pe/vec3.hpp"

namespace pe {

// Exact mirror of src/pursuit_evasion/env/observations.py:build_team_obs.
// Layout: [time] + per-self[pos3,vel3,alive1] + per-(self,opp)[relpos3,relvel3,alive1]
// and, for games with a defended asset, an objective block:
//         + per-self[rel_asset3, rel_asset_vel3, fuel1] + [asset_live1].
// Games without an asset (tag) stay byte-identical to the original 43-dim layout,
// so previously trained policies still load.
inline int obs_dim(int n_self, int n_opp, bool has_asset = false) {
  const int base = 1 + n_self * 7 + n_self * n_opp * 7;
  return base + (has_asset ? n_self * 7 + 1 : 0);
}
inline int act_dim(int n_self) { return n_self * 3; }

inline std::vector<float> build_team_obs(const TeamView& v) {
  const double L = v.arena->half_extent;
  const double vmax = std::max(v.vmax, 1e-6);
  const double ovmax = std::max(v.opp_vmax, 1e-6);
  const int n_self = v.n_self(), n_opp = v.n_opp();
  std::vector<float> o;
  o.reserve(obs_dim(n_self, n_opp, v.has_asset));

  auto push = [&](double val) { o.push_back((float)clamp(val, -5.0, 5.0)); };
  auto push3 = [&](Vec3 a, double s) { push(a.x * s); push(a.y * s); push(a.z * s); };

  push(v.time_frac * 2.0 - 1.0);
  for (int i = 0; i < n_self; ++i) {
    push3(v.self_pos[i], 1.0 / L);
    push3(v.self_vel[i], 1.0 / vmax);
    push(v.self_alive[i] ? 1.0 : -1.0);
  }
  for (int i = 0; i < n_self; ++i) {
    for (int j = 0; j < n_opp; ++j) {
      push3(v.opp_pos[j] - v.self_pos[i], 1.0 / (2.0 * L));
      push3(v.opp_vel[j] - v.self_vel[i], 1.0 / ovmax);
      push(v.opp_alive[j] ? 1.0 : -1.0);
    }
  }
  if (v.has_asset) {
    for (int i = 0; i < n_self; ++i) {
      push3(v.asset_pos - v.self_pos[i], 1.0 / (2.0 * L));
      push3(v.asset_vel - v.self_vel[i], 1.0 / vmax);
      const double fuel = i < (int)v.self_fuel.size() ? v.self_fuel[i] : 1.0;
      push(fuel * 2.0 - 1.0);
    }
    push(1.0);   // asset intact this tick
  }
  return o;
}

}  // namespace pe
