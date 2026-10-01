#include <doctest/doctest.h>

#include "pe/config.hpp"
#include "pe/safety.hpp"
#include "pe/vec3.hpp"

using namespace pe;

namespace {
constexpr double L = 12.0, ZMIN = 0.5, ZMAX = 12.0;
SafetyConfig cfg;
double VMAX = 5.25;  // pursuer v_max = 21/4
}  // namespace

TEST_CASE("geofence: outward push at +x face is cancelled and steered inward") {
  Vec3 pos{L, 0, 3}, vel{0, 0, 0}, action{1, 0, 0};
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK(v.geofence);
  CHECK(action.x < 0.0);  // pushed back inside, not out
}

TEST_CASE("geofence: -x face pushes back toward +x") {
  Vec3 pos{-L, 0, 3}, vel{0, 0, 0}, action{-1, 0, 0};
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK(v.geofence);
  CHECK(action.x > 0.0);
}

TEST_CASE("geofence: ceiling clamps upward z command") {
  Vec3 pos{0, 0, ZMAX}, vel{0, 0, 0}, action{0, 0, 1};
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK(v.geofence);
  CHECK(action.z < 0.0);
}

TEST_CASE("no violation well inside the fence at low speed") {
  Vec3 pos{0, 0, 3}, vel{0.1, 0, 0}, action{1, 0, 0};
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK_FALSE(v.any());
  CHECK(action.x == doctest::Approx(1.0));
}

TEST_CASE("speed cap removes the speed-increasing component") {
  Vec3 pos{0, 0, 3}, vel{6.0, 0, 0}, action{1, 0, 0};  // speed 6 > 0.98*5.25
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK(v.speed);
  CHECK(action.x == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("speed cap allows braking (component opposing velocity)") {
  Vec3 pos{0, 0, 3}, vel{6.0, 0, 0}, action{-1, 0, 0};
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK_FALSE(v.speed);       // opposing velocity is allowed
  CHECK(action.x < 0.0);
}

TEST_CASE("brake-aware fence: fast approach is braked BEFORE the face") {
  // inside the keep-in box (hi = 11.25) but crossing it within the horizon
  Vec3 pos{10.9, 0, 3}, vel{5.0, 0, 0}, action{1, 0, 0};
  auto v = filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK(v.geofence);
  CHECK(action.x < 0.0);
  SafetyConfig legacy;
  legacy.brake_horizon_s = 0.0;
  legacy.keep_fence_authority = false;
  Vec3 a2{1, 0, 0};
  CHECK_FALSE(filter_action(pos, vel, a2, VMAX, L, ZMIN, ZMAX, legacy).geofence);
}

TEST_CASE("brake-aware fence: a slow drift toward the face is left alone") {
  Vec3 pos{10.9, 0, 3}, vel{0.5, 0, 0}, action{1, 0, 0};
  CHECK_FALSE(filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg).geofence);
}

TEST_CASE("fence keeps its authority through the unit-norm clip") {
  // a diagonal command at the +x face: the inward x correction must survive
  // apply_dynamics' norm clip intact; the tangential axes give way instead
  Vec3 pos{L, 0, 3}, vel{0, 0, 0}, action{1, 1, 1};
  filter_action(pos, vel, action, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK(norm(action) <= 1.0 + 1e-12);
  CHECK(action.x == doctest::Approx(-1.0));
}
