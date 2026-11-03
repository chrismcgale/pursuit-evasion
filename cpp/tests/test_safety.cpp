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

// ---- own-team separation (mirrors tests/test_separation.py) ------------------
namespace {
SafetyConfig sep_on() {
  SafetyConfig c;
  c.separation_m = 1.5;
  return c;
}
const Vec3 Z{0, 0, 0};
}  // namespace

TEST_CASE("separation: off by default, mates change nothing") {
  CHECK(SafetyConfig{}.separation_m == 0.0);
  Vec3 a{1, 0, 0}, b{1, 0, 0};
  auto v = filter_action({0, 0, 3}, Z, a, VMAX, L, ZMIN, ZMAX, cfg, {{{0.5, 0, 3}, Z, 1}}, 0);
  filter_action({0, 0, 3}, Z, b, VMAX, L, ZMIN, ZMAX, cfg);
  CHECK_FALSE(v.separation);
  CHECK(a.x == b.x);
  CHECK(a.y == b.y);
  CHECK(a.z == b.z);
}

TEST_CASE("separation: a converging command is turned away") {
  Vec3 a{1, 0, 0};
  auto v = filter_action({0, 0, 3}, Z, a, VMAX, L, ZMIN, ZMAX, sep_on(), {{{1, 0, 3}, Z, 1}}, 0);
  CHECK(v.separation);
  CHECK(a.x < 0.0);
  CHECK(norm(a) <= 1.0 + 1e-12);
}

TEST_CASE("separation: brake-aware (acts on the predicted gap)") {
  Vec3 a{1, 0, 0};
  auto v = filter_action({0, 0, 3}, {4, 0, 0}, a, VMAX, L, ZMIN, ZMAX, sep_on(),
                         {{{2, 0, 3}, {-4, 0, 0}, 1}}, 0);
  CHECK(v.separation);
  CHECK(a.x < 0.0);
  Vec3 b{0, 1, 0};
  v = filter_action({0, 0, 3}, Z, b, VMAX, L, ZMIN, ZMAX, sep_on(), {{{2, 0, 3}, Z, 1}}, 0);
  CHECK_FALSE(v.separation);
}

TEST_CASE("separation: the push keeps its authority through the norm clip") {
  Vec3 a{0, 1, 0};
  auto v = filter_action({0, 0, 3}, Z, a, VMAX, L, ZMIN, ZMAX, sep_on(), {{{-0.5, 0, 3}, Z, 1}}, 0);
  CHECK(v.separation);
  CHECK(std::abs(norm(a) - 1.0) < 1e-12);
  CHECK(std::abs(a.x - 1.0) < 1e-12);
}

TEST_CASE("separation: coincident agents split in opposite directions") {
  Vec3 a0{0, 0, 0}, a1{0, 0, 0};
  separate({0, 0, 3}, Z, a0, {{{0, 0, 3}, Z, 1}}, sep_on(), 0);
  separate({0, 0, 3}, Z, a1, {{{0, 0, 3}, Z, 0}}, sep_on(), 1);
  CHECK(a0.z > 0.0);
  CHECK(a1.z < 0.0);
}

TEST_CASE("separation: the fence wins at the wall") {
  Vec3 a{0, 0, 0};
  auto v = filter_action({L, 0, 3}, Z, a, VMAX, L, ZMIN, ZMAX, sep_on(), {{{L - 1, 0, 3}, Z, 1}}, 0);
  CHECK(v.separation);
  CHECK(v.geofence);
  CHECK(a.x < 0.0);
}

TEST_CASE("separation: the team filter ignores dead teammates") {
  ArenaConfig arena;
  TeamView tv;
  tv.arena = &arena;
  tv.vmax = VMAX;
  tv.self_pos = {{0, 0, 3}, {0.5, 0, 3}};
  tv.self_vel = {Z, Z};
  tv.self_alive = {1, 1};
  TeamSafetyFilter f(sep_on());
  f.apply(tv, std::vector<double>(6, 0.0));
  CHECK(f.n_separation == 2);
  tv.self_alive[1] = 0;
  f.reset();
  f.apply(tv, std::vector<double>(6, 0.0));
  CHECK(f.n_separation == 0);
}
