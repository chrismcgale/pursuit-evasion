#include "pe/scripted.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pe {

std::optional<double> lead_intercept_time(Vec3 rel_pos, Vec3 evader_vel, double vp) {
  // Solves |rel_pos + evader_vel * t| = vp * t for the smallest positive t.
  double a = dot(evader_vel, evader_vel) - vp * vp;
  double b = 2.0 * dot(rel_pos, evader_vel);
  double c = dot(rel_pos, rel_pos);
  if (std::fabs(a) < 1e-6) {
    if (std::fabs(b) > 1e-6) {
      double cand = -c / b;
      if (cand > 0) return cand;
    }
    return std::nullopt;
  }
  double disc = b * b - 4 * a * c;
  if (disc < 0) return std::nullopt;
  double sq = std::sqrt(disc);
  double r0 = (-b - sq) / (2 * a), r1 = (-b + sq) / (2 * a);
  double best = std::numeric_limits<double>::infinity();
  for (double r : {r0, r1})
    if (r > 1e-6) best = std::min(best, r);
  if (!std::isfinite(best)) return std::nullopt;
  return best;
}

Vec3 lead_intercept_dir(Vec3 rel_pos, Vec3 evader_vel, double vp) {
  auto t = lead_intercept_time(rel_pos, evader_vel, vp);
  if (!t) return unit(rel_pos);   // pure pursuit fallback
  return unit(rel_pos + evader_vel * (*t));
}

double time_to_asset(Vec3 pos, Vec3 vel, Vec3 asset, double vmax) {
  double d = norm(asset - pos);
  double speed = norm(vel);
  double closing = speed > 1e-6 ? dot(vel, unit(asset - pos)) : 0.0;
  // use the better of its current closure and its top speed: a missile that is
  // currently turning is still a threat on its full-speed clock
  double rate = std::max({closing, 0.35 * vmax, 1e-6});
  return d / rate;
}

std::vector<int> assign_targets(const std::vector<Vec3>& p_pos,
                                const std::vector<Vec3>& e_pos,
                                const std::vector<char>& live) {
  int n_p = (int)p_pos.size();
  std::vector<int> live_idx;
  for (int j = 0; j < (int)live.size(); ++j)
    if (live[j]) live_idx.push_back(j);
  std::vector<int> assign(n_p, -1);
  if (live_idx.empty()) return assign;
  if (live_idx.size() == 1) {
    std::fill(assign.begin(), assign.end(), live_idx[0]);
    return assign;
  }
  // pairs (cost, pi, evader_global_index) generated pi-outer, ei-inner; stable
  // sort by cost so ties keep generation order (matches Python's sorted()).
  struct Pair { double cost; int pi; int ej; };
  std::vector<Pair> pairs;
  for (int pi = 0; pi < n_p; ++pi)
    for (int ei = 0; ei < (int)live_idx.size(); ++ei)
      pairs.push_back({norm(p_pos[pi] - e_pos[live_idx[ei]]), pi, live_idx[ei]});
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const Pair& a, const Pair& b) { return a.cost < b.cost; });

  std::vector<char> used_p(n_p, 0), used_e(e_pos.size(), 0);
  int n_used_e = 0;
  for (auto& pr : pairs) {
    if (used_p[pr.pi] || used_e[pr.ej]) continue;
    assign[pr.pi] = pr.ej;
    used_p[pr.pi] = 1;
    used_e[pr.ej] = 1;
    if (++n_used_e == (int)live_idx.size()) break;
  }
  for (int pi = 0; pi < n_p; ++pi) {
    if (assign[pi] != -1) continue;
    double best = std::numeric_limits<double>::infinity();
    int bj = live_idx[0];
    for (int j : live_idx) {
      double d = norm(e_pos[j] - p_pos[pi]);
      if (d < best) { best = d; bj = j; }
    }
    assign[pi] = bj;
  }
  return assign;
}

std::vector<double> InterceptPursuers::act(const TeamView& v) {
  double vp = v.vmax;
  const auto& p_pos = v.self_pos;
  const auto& e_pos = v.opp_pos;
  const auto& e_vel = v.opp_vel;
  std::vector<int> assign = assign_targets(p_pos, e_pos, v.opp_alive);
  int n_live = 0;
  for (char a : v.opp_alive) n_live += a ? 1 : 0;

  int n = v.n_self();
  std::vector<double> out(n * 3, 0.0);
  for (int pi = 0; pi < n; ++pi) {
    int ej = assign[pi];
    if (ej < 0) continue;
    Vec3 rel = e_pos[ej] - p_pos[pi];
    Vec3 d = lead_intercept_dir(rel, e_vel[ej], vp);
    if (n_live == 1) {
      double side = (pi % 2 == 0) ? 1.0 : -1.0;
      Vec3 perp = unit(cross(d, {0, 0, 1}));
      if (norm(perp) < 1e-6) perp = {1, 0, 0};
      double close = norm(rel);
      double flank = clamp(close / 4.0, 0.0, 1.0);
      d = unit(d + side * 0.6 * flank * perp);
    }
    out[pi * 3 + 0] = clamp(d.x, -1, 1);
    out[pi * 3 + 1] = clamp(d.y, -1, 1);
    out[pi * 3 + 2] = clamp(d.z, -1, 1);
  }
  return out;
}

std::vector<double> FieldEvaders::act(const TeamView& v) {
  const double L = v.arena->half_extent;
  const double z_lo = v.arena->z_min, z_hi = v.arena->z_max;
  const auto& e_pos = v.self_pos;
  const auto& p_pos = v.opp_pos;
  int n = v.n_self();
  std::vector<double> out(n * 3, 0.0);

  for (int i = 0; i < n; ++i) {
    if (!v.self_alive[i]) continue;
    Vec3 pos = e_pos[i];
    Vec3 force{0, 0, 0};

    for (int pj = 0; pj < v.n_opp(); ++pj) {
      Vec3 to_me = pos - p_pos[pj];
      double d = norm(to_me) + 1e-6;
      double w = 1.0 / (d * d);
      force = force + w * unit(to_me);
      if (d < 5.0) {
        Vec3 perp = unit(cross(to_me, {0, 0, 1}));
        if (norm(perp) < 1e-6) perp = {0, 0, 1};
        double sign = ((i + pj) % 2 == 0) ? 1.0 : -1.0;
        force = force + juke * w * sign * perp;
      }
    }

    double margin = 2.5;
    for (int ax = 0; ax < 2; ++ax) {
      if (pos[ax] > L - margin)
        force[ax] -= (pos[ax] - (L - margin)) / margin;
      else if (pos[ax] < -(L - margin))
        force[ax] += ((-(L - margin)) - pos[ax]) / margin;
    }
    if (pos[2] > z_hi - margin)
      force[2] -= (pos[2] - (z_hi - margin)) / margin;
    else if (pos[2] < z_lo + margin)
      force[2] += ((z_lo + margin) - pos[2]) / margin;

    for (int k = 0; k < n; ++k) {
      if (k == i || !v.self_alive[k]) continue;
      Vec3 to_me = pos - e_pos[k];
      double d = norm(to_me) + 1e-6;
      if (d < 3.0) force = force + 0.4 * unit(to_me) * (1.0 / d);
    }

    Vec3 a = unit(force);
    out[i * 3 + 0] = clamp(a.x, -1, 1);
    out[i * 3 + 1] = clamp(a.y, -1, 1);
    out[i * 3 + 2] = clamp(a.z, -1, 1);
  }
  return out;
}

// ------------------------------------------------------- GuardDefenders
// Exact mirror of scripted/defenders.py.

static constexpr double kBig = 1e9;

Vec3 GuardDefenders::gate_point(Vec3 att_pos, Vec3 asset) const {
  // A point on the attacker's inbound bearing, a standoff out from the asset.
  Vec3 to_att = att_pos - asset;
  double r = norm(to_att);
  if (r < 1e-6) return asset;
  double reach = clamp(r * 0.5, min_standoff, standoff);
  return asset + unit(to_att) * reach;
}

std::vector<int> GuardDefenders::assign(const TeamView& v, Vec3 asset) const {
  const int n_self = v.n_self();
  std::vector<int> live;
  for (int j = 0; j < v.n_opp(); ++j)
    if (v.opp_alive[j]) live.push_back(j);
  std::vector<int> out(n_self, -1);
  if (live.empty()) return out;

  // order threats by time-to-asset (most urgent first). Distance-greedy
  // assignment is actively wrong here: the nearest missile is often the least
  // urgent. stable_sort keeps ties in index order, matching Python's sorted().
  std::stable_sort(live.begin(), live.end(), [&](int a, int b) {
    return time_to_asset(v.opp_pos[a], v.opp_vel[a], asset, v.opp_vmax) <
           time_to_asset(v.opp_pos[b], v.opp_vel[b], asset, v.opp_vmax);
  });

  std::vector<char> free(n_self, 1);
  int n_free = n_self;
  for (int j : live) {
    if (n_free == 0) break;
    // give this threat whichever free defender can meet it soonest; if none can
    // intercept, whichever is closest to its gate point
    int best_i = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_self; ++i) {
      if (!free[i]) continue;
      Vec3 rel = v.opp_pos[j] - v.self_pos[i];
      auto t = lead_intercept_time(rel, v.opp_vel[j], v.vmax);
      double cost = t ? *t : kBig + norm(gate_point(v.opp_pos[j], asset) - v.self_pos[i]);
      if (cost < best_cost) { best_cost = cost; best_i = i; }
    }
    if (best_i < 0) break;
    out[best_i] = j;
    free[best_i] = 0;
    --n_free;
  }
  // spare defenders back up the most urgent threat
  for (int i = 0; i < n_self; ++i)
    if (free[i]) out[i] = live[0];
  return out;
}

std::vector<double> GuardDefenders::act(const TeamView& v) {
  const int n_self = v.n_self();
  std::vector<double> out(n_self * 3, 0.0);
  if (!v.has_asset) return out;
  const Vec3 asset = v.asset_pos;
  std::vector<int> a = assign(v, asset);

  for (int i = 0; i < n_self; ++i) {
    if (!v.self_alive[i]) continue;
    const Vec3 pos = v.self_pos[i];
    Vec3 d{0, 0, 0};
    const int j = a[i];
    if (j < 0) {
      // nothing left to kill: settle back onto the asset's patrol ring
      Vec3 to_asset = asset - pos;
      double r = norm(to_asset);
      d = unit(to_asset) * (r > patrol_radius ? 1.0 : -0.3);
    } else {
      Vec3 rel = v.opp_pos[j] - pos;
      auto t_int = lead_intercept_time(rel, v.opp_vel[j], v.vmax);
      double t_asset = time_to_asset(v.opp_pos[j], v.opp_vel[j], asset, v.opp_vmax);
      if (t_int && *t_int <= t_asset) {
        d = lead_intercept_dir(rel, v.opp_vel[j], v.vmax);
      } else {
        // can't catch it — be where it has to come through
        Vec3 to_gate = gate_point(v.opp_pos[j], asset) - pos;
        d = norm(to_gate) < 0.8 ? unit(rel)   // on station: hold and face the threat
                                : unit(to_gate);
      }
    }
    out[i * 3 + 0] = clamp(d.x, -1, 1);
    out[i * 3 + 1] = clamp(d.y, -1, 1);
    out[i * 3 + 2] = clamp(d.z, -1, 1);
  }
  return out;
}

// ------------------------------------------------------ MissileAttackers
// Exact mirror of scripted/attackers.py.

// Any unit vector perpendicular to v (prefers the horizontal plane).
static Vec3 perp_of(Vec3 v) {
  Vec3 p = cross(v, {0.0, 0.0, 1.0});
  if (norm(p) < 1e-6) p = cross(v, {1.0, 0.0, 0.0});
  return unit(p);
}

std::vector<double> MissileAttackers::act(const TeamView& v) {
  const int n_self = v.n_self();
  std::vector<double> out(n_self * 3, 0.0);
  if (!v.has_asset) return out;

  const Vec3 asset_p = v.asset_pos, asset_v = v.asset_vel;
  for (int i = 0; i < n_self; ++i) {
    if (!v.self_alive[i]) continue;
    const Vec3 pos = v.self_pos[i];
    const Vec3 rel = asset_p - pos;
    const double rng = norm(rel);
    // 1. lead guidance: aim where the asset WILL be (matters for the convoy)
    Vec3 d = lead_intercept_dir(rel, asset_v, v.vmax);

    // 2. bearing split, fading to zero as the range closes
    if (rng > 1e-6) {
      double bias = split * clamp((rng - 6.0) / split_range, 0.0, 1.0);
      double side = (i % 2 == 0) ? 1.0 : -1.0;
      d = unit(d + side * bias * perp_of(d));
    }

    // 3. terminal break against the nearest interceptor on a collision bearing
    int best_j = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for (int j = 0; j < v.n_opp(); ++j) {
      if (!v.opp_alive[j]) continue;
      double dj = norm(v.opp_pos[j] - pos);
      if (dj < best_d) { best_d = dj; best_j = j; }
    }
    if (best_j >= 0 && best_d < dodge_range) {
      Vec3 los = unit(v.opp_pos[best_j] - pos);
      // only break if the threat is roughly ahead; one behind us is beaten
      if (dot(los, d) > 0.1) {
        double urgency = 1.0 - best_d / dodge_range;
        double side = ((i + best_j) % 2 == 0) ? 1.0 : -1.0;
        d = unit(d + side * dodge_gain * urgency * perp_of(los));
      }
    }
    out[i * 3 + 0] = clamp(d.x, -1, 1);
    out[i * 3 + 1] = clamp(d.y, -1, 1);
    out[i * 3 + 2] = clamp(d.z, -1, 1);
  }
  return out;
}

ControllerPair default_controllers(const std::string& game_key) {
  if (game_key == "tag")
    return {std::make_unique<InterceptPursuers>(), std::make_unique<FieldEvaders>()};
  return {std::make_unique<GuardDefenders>(), std::make_unique<MissileAttackers>()};
}

}  // namespace pe
