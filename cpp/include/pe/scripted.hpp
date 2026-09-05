#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pe/controller.hpp"
#include "pe/vec3.hpp"

namespace pe {

// Free functions mirrored from scripted/pursuers.py for reuse and unit testing.
// lead_intercept_time returns nullopt when no positive root exists: the target is
// faster and opening, so NO constant-speed pursuit closes it. That case is the
// whole air-defence game (see GuardDefenders).
std::optional<double> lead_intercept_time(Vec3 rel_pos, Vec3 evader_vel, double vp);
Vec3 lead_intercept_dir(Vec3 rel_pos, Vec3 evader_vel, double vp);
std::vector<int> assign_targets(const std::vector<Vec3>& p_pos,
                                const std::vector<Vec3>& e_pos,
                                const std::vector<char>& live);

// Mirrors scripted/defenders.py:time_to_asset — optimistic seconds until this
// attacker reaches the asset.
double time_to_asset(Vec3 pos, Vec3 vel, Vec3 asset, double vmax);

// Mirrors scripted/pursuers.py:InterceptPursuers
struct InterceptPursuers : Controller {
  InterceptPursuers() { name = "scripted_pursuers"; }
  std::vector<double> act(const TeamView& v) override;
};

// Mirrors scripted/evaders.py:FieldEvaders
struct FieldEvaders : Controller {
  double juke = 0.7;
  FieldEvaders() { name = "scripted_evaders"; }
  std::vector<double> act(const TeamView& v) override;
};

// Mirrors scripted/defenders.py:GuardDefenders — the interceptor-drone side of
// the objective games. Interceptors are ~40% slower than the missiles, so a tail
// chase is not a strategy; this runs two regimes (fly the lead bearing when an
// intercept is feasible, otherwise hold a gate point on the inbound bearing) and
// assigns by threat time rather than distance.
struct GuardDefenders : Controller {
  double standoff = 9.0;         // gate distance from the asset (m)
  double min_standoff = 3.5;
  double patrol_radius = 6.0;
  GuardDefenders() { name = "scripted_defenders"; }
  std::vector<double> act(const TeamView& v) override;

  Vec3 gate_point(Vec3 att_pos, Vec3 asset) const;
  std::vector<int> assign(const TeamView& v, Vec3 asset) const;
};

// Mirrors scripted/attackers.py:MissileAttackers — lead guidance to the asset,
// a range-fading bearing split so the raid arrives on separated bearings, and a
// terminal break against a nearby interceptor on a collision bearing.
struct MissileAttackers : Controller {
  double split = 0.55;
  double split_range = 25.0;
  double dodge_range = 11.0;
  double dodge_gain = 1.5;
  MissileAttackers() { name = "scripted_attackers"; }
  std::vector<double> act(const TeamView& v) override;
};

// Mirrors scripted/__init__.py:default_controllers — the scripted matchup for a
// game, as {pursuer-slot, evader-slot}.
struct ControllerPair {
  std::unique_ptr<Controller> pursuer;
  std::unique_ptr<Controller> evader;
};
ControllerPair default_controllers(const std::string& game_key);

}  // namespace pe
