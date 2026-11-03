#pragma once
#include <cmath>
#include <vector>

#include "pe/config.hpp"
#include "pe/controller.hpp"
#include "pe/sim.hpp"
#include "pe/vec3.hpp"

namespace pe {

struct Violation {
  bool geofence = false;
  bool speed = false;
  bool separation = false;
  bool any() const { return geofence || speed || separation; }
};

struct Mate {
  Vec3 pos, vel;
  int index;
};

// Exact mirror of safety.py:separate. Own-team separation, brake-aware: acts on
// the predicted relative position as well as the current one, removes the
// command component toward the teammate, pushes away in proportion to the
// shortfall, and keeps that push through the downstream norm clip.
inline bool separate(Vec3 pos, Vec3 vel, Vec3& a, const std::vector<Mate>& mates,
                     const SafetyConfig& cfg, int rank) {
  const double h = cfg.brake_horizon_s;
  const double sep = cfg.separation_m;
  bool fired = false;
  for (const Mate& m : mates) {
    const Vec3 r = (pos + h * vel) - (m.pos + h * m.vel);
    const Vec3 r0 = pos - m.pos;
    const double d = norm(r);
    const double d0 = norm(r0);
    const double dmin = std::min(d, d0);
    if (dmin >= sep) continue;
    Vec3 n;
    if (d > 1e-9) n = r / d;
    else if (d0 > 1e-9) n = r0 / d0;
    else n = Vec3{0.0, 0.0, rank < m.index ? 1.0 : -1.0};
    const double along = dot(a, n);
    if (along < 0) a = a - along * n;
    const double c = cfg.correction_gain * (sep - dmin);
    a = a + c * n;
    const double n2 = dot(a, a);
    if (n2 > 1.0) {
      const double p = dot(a, n);
      if (p >= 1.0) {
        a = n;
      } else {
        const Vec3 perp = a - p * n;
        const double pn = norm(perp);
        if (pn > 1e-12) {
          const double kk = std::sqrt(1.0 - p * p) / pn;
          a = p * n + kk * perp;
        }
      }
    }
    fired = true;
  }
  return fired;
}

// Exact mirror of src/pursuit_evasion/safety.py:filter_action. Corrects `action`
// in place to respect the geofence and speed cap; returns which fired.
inline Violation filter_action(Vec3 pos, Vec3 vel, Vec3& action, double vmax,
                               double half_extent, double z_min, double z_max,
                               const SafetyConfig& cfg,
                               const std::vector<Mate>& mates = {}, int rank = 0) {
  Violation v;
  for (int k = 0; k < 3; ++k) action[k] = clamp(action[k], -1.0, 1.0);
  if (cfg.separation_m > 0 && !mates.empty())
    v.separation = separate(pos, vel, action, mates, cfg, rank);

  double lo_xy = -(half_extent - cfg.geofence_margin);
  double hi_xy = half_extent - cfg.geofence_margin;
  double lo[3] = {lo_xy, lo_xy, z_min + cfg.z_margin};
  double hi[3] = {hi_xy, hi_xy, z_max - cfg.z_margin};

  // Acts on the predicted position too: a position-only fence fires a whole
  // control period late and then still needs braking distance (see safety.py).
  bool fenced[3] = {false, false, false};
  for (int ax = 0; ax < 3; ++ax) {
    const double pred = pos[ax] + cfg.brake_horizon_s * vel[ax];
    if (pos[ax] >= hi[ax]) {
      double over = pos[ax] - hi[ax];
      action[ax] = std::min(action[ax], 0.0) - cfg.correction_gain * over;
      fenced[ax] = true;
    } else if (pos[ax] <= lo[ax]) {
      double under = lo[ax] - pos[ax];
      action[ax] = std::max(action[ax], 0.0) + cfg.correction_gain * under;
      fenced[ax] = true;
    } else if (pred > hi[ax]) {
      action[ax] = std::min(action[ax], 0.0) - cfg.correction_gain * (pred - hi[ax]);
      fenced[ax] = true;
    } else if (pred < lo[ax]) {
      action[ax] = std::max(action[ax], 0.0) + cfg.correction_gain * (lo[ax] - pred);
      fenced[ax] = true;
    }
  }
  v.geofence = fenced[0] || fenced[1] || fenced[2];

  double speed = norm(vel);
  double limit = cfg.speed_limit_frac * vmax;
  if (speed > limit && speed > 1e-6) {
    Vec3 vhat = vel / speed;
    double along = dot(action, vhat);
    if (along > 0) {
      action = action - along * vhat;
      v.speed = true;
    }
  }
  for (int k = 0; k < 3; ++k) action[k] = clamp(action[k], -1.0, 1.0);
  // Keep the fence's authority through apply_dynamics' unit-norm clip: shrink
  // the unfenced axes, never the inward correction.
  if (cfg.keep_fence_authority && v.geofence) {
    const double n2 = action[0] * action[0] + action[1] * action[1] + action[2] * action[2];
    if (n2 > 1.0) {
      double f2 = 0.0;
      for (int ax = 0; ax < 3; ++ax)
        if (fenced[ax]) f2 += action[ax] * action[ax];
      if (f2 >= 1.0) {
        const double f = std::sqrt(f2);
        for (int ax = 0; ax < 3; ++ax) action[ax] = fenced[ax] ? action[ax] / f : 0.0;
      } else {
        const double k = std::sqrt((1.0 - f2) / (n2 - f2));
        for (int ax = 0; ax < 3; ++ax)
          if (!fenced[ax]) action[ax] = action[ax] * k;
      }
    }
  }
  for (int k = 0; k < 3; ++k) action[k] = clamp(action[k], -1.0, 1.0);
  return v;
}

class TeamSafetyFilter {
 public:
  int n_geofence = 0, n_speed = 0, n_separation = 0;
  explicit TeamSafetyFilter(SafetyConfig cfg = {}) : cfg_(cfg) {}
  void reset() { n_geofence = n_speed = n_separation = 0; }
  const SafetyConfig& cfg() const { return cfg_; }

  std::vector<double> apply(const TeamView& v, const std::vector<double>& team_action) {
    const auto& a = v.arena;
    std::vector<double> out(team_action.size(), 0.0);
    for (int i = 0; i < v.n_self(); ++i) {
      if (!v.self_alive[i]) continue;
      Vec3 act{team_action[i * 3], team_action[i * 3 + 1], team_action[i * 3 + 2]};
      std::vector<Mate> mates;
      if (cfg_.separation_m > 0)
        for (int j = 0; j < v.n_self(); ++j)
          if (j != i && v.self_alive[j]) mates.push_back({v.self_pos[j], v.self_vel[j], j});
      Violation viol = filter_action(v.self_pos[i], v.self_vel[i], act, v.vmax,
                                     a->half_extent, a->z_min, a->z_max, cfg_, mates, i);
      out[i * 3 + 0] = act.x;
      out[i * 3 + 1] = act.y;
      out[i * 3 + 2] = act.z;
      n_geofence += viol.geofence ? 1 : 0;
      n_speed += viol.speed ? 1 : 0;
      n_separation += viol.separation ? 1 : 0;
    }
    return out;
  }

 private:
  SafetyConfig cfg_;
};

// Wraps any controller so its output always passes through the safety filter.
struct ShieldedController : Controller {
  Controller* inner;
  TeamSafetyFilter filter;
  explicit ShieldedController(Controller* c, SafetyConfig cfg = {})
      : inner(c), filter(cfg) {
    name = "shielded_" + c->name;
  }
  void reset() override {
    inner->reset();
    filter.reset();
  }
  std::vector<double> act(const TeamView& v) override {
    return filter.apply(v, inner->act(v));
  }
};

}  // namespace pe
