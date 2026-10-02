// Unit tests for the objective games: the airframe transform (missile vs drone)
// and the two new gate profiles. The gate tests load the *shipped* XML from
// trees/ rather than an inline string, so a hand-edit in Groot2 that breaks a
// profile fails CI instead of silently changing behaviour.
#include <doctest/doctest.h>

#include <behaviortree_cpp/bt_factory.h>

#include <cmath>
#include <string>

#include "pe/bt_nodes.hpp"
#include "pe/config.hpp"
#include "pe/dynamics.hpp"
#include "pe/features.hpp"
#include "pe/games.hpp"
#include "pe/gate.hpp"

using namespace pe;

namespace {

AgentFeatures obj_base() {
  AgentFeatures f;
  f.alive = true;
  f.dist_nearest = 20;
  f.dist_second = 25;
  f.n_live_others = 2;
  f.asset_dist = 30;
  f.threat_time = 10.0;
  f.intercept_feasible = true;
  f.intercept_ahead = false;
  f.fuel = 1.0;
  return f;
}

// Tick a shipped tree file and report which controller it selected.
std::string route(const std::string& tree_file, const AgentFeatures& f,
                  const GateThresholds& thr = {}) {
  BT::BehaviorTreeFactory factory;
  register_gate_nodes(factory);
  auto tree = factory.createTreeFromFile(std::string(PE_TREES_DIR) + "/" + tree_file);
  tree.rootBlackboard()->set("features", f);
  tree.rootBlackboard()->set("thresholds", thr);
  tree.rootBlackboard()->set("mode", std::string("scripted"));
  tree.tickOnce();
  return tree.rootBlackboard()->get<std::string>("mode");
}

}  // namespace

// --------------------------------------------------------------------- dynamics

TEST_CASE("holonomic airframes pass the command straight through") {
  const auto d = drone_interceptor();
  Vec3 vel{5, 0, 0}, cmd{0, -1, 0};
  Vec3 u = apply_dynamics(d, vel, cmd, 1);
  CHECK(u.x == doctest::Approx(0.0));
  CHECK(u.y == doctest::Approx(-1.0));
  CHECK(u.z == doctest::Approx(0.0));
}

// Regression: the pre-multi-game code clipped commands per COMPONENT, so a drone
// commanded (1,1,1) pulled sqrt(3)*gear and topped out at 7.42 m/s against a
// documented v_max of 5.25. Scripted controllers emit unit-norm directions and
// never saw it; learned policies fill the action cube and exploited it, which is
// why every tag policy number moved when the norm clip landed. Thrust must be
// isotropic or `v_max = gear/damping` is only true along the axes.
TEST_CASE("a drone's thrust is isotropic: no command direction beats unit thrust") {
  const auto d = drone_interceptor();
  const Vec3 vel{5, 0, 0};
  for (const Vec3& cmd : {Vec3{1, 1, 1}, Vec3{1, 1, 0}, Vec3{-1, 1, -1}, Vec3{1, 0, 0}})
    CHECK(norm(apply_dynamics(d, vel, cmd, 1)) <= doctest::Approx(1.0));
  // and the direction is preserved while being scaled down
  const Vec3 u = apply_dynamics(d, vel, Vec3{1, 1, 1}, 1);
  CHECK(u.x == doctest::Approx(u.y));
  CHECK(u.y == doctest::Approx(u.z));
}

TEST_CASE("a missile cannot turn around: thrust stays on the velocity axis") {
  const auto m = missile_attacker();
  Vec3 vel{10, 0, 0};
  // command a full reversal; the airframe must refuse to point backwards
  Vec3 u = apply_dynamics(m, vel, Vec3{-1, 0, 0}, 100);
  CHECK(u.x > 0.0);
}

TEST_CASE("lateral authority binds as a ratio of forward thrust") {
  const auto m = missile_attacker();
  Vec3 vel{10, 0, 0};
  // hard lateral demand; the final unit-clip rescales both components, so the
  // invariant is the lateral:forward RATIO, not the absolute lateral magnitude
  Vec3 u = apply_dynamics(m, vel, Vec3{0, 1, 0}, 100);
  const double ratio = std::abs(u.y) / std::abs(u.x);
  CHECK(ratio <= doctest::Approx(m.lat_authority).epsilon(1e-9));
}

TEST_CASE("a burnt-out missile coasts: no forward thrust, lateral only") {
  const auto m = missile_attacker();
  Vec3 vel{10, 0, 0};
  Vec3 u = apply_dynamics(m, vel, Vec3{1, 0, 0}, 0);  // burn_left exhausted
  CHECK(u.x == doctest::Approx(0.0));
}

TEST_CASE("below v_align a missile has no body axis to thrust along") {
  const auto m = missile_attacker();
  Vec3 slow{0.1, 0, 0};  // < v_align (0.5)
  Vec3 powered = apply_dynamics(m, slow, Vec3{1, 0, 0}, 100);
  CHECK(norm(powered) == doctest::Approx(1.0));  // free to accelerate off the rail
  Vec3 coasting = apply_dynamics(m, slow, Vec3{1, 0, 0}, 0);
  CHECK(norm(coasting) == doctest::Approx(0.0));
}

TEST_CASE("turn radius grows with the square of speed") {
  const auto m = missile_attacker();
  const double r1 = m.turn_radius(5.0), r2 = m.turn_radius(10.0);
  CHECK(r2 == doctest::Approx(4.0 * r1));
}

TEST_CASE("is_spent only once burnt out AND decayed below stall") {
  const auto m = missile_attacker();
  CHECK_FALSE(is_spent(m, Vec3{10, 0, 0}, 100));  // still burning
  CHECK_FALSE(is_spent(m, Vec3{10, 0, 0}, 0));    // coasting but still fast
  CHECK(is_spent(m, Vec3{1, 0, 0}, 0));           // coasting and below v_stall (2)
}

TEST_CASE("drones never go spent — they have no finite burn") {
  const auto d = drone_interceptor();
  CHECK_FALSE(is_spent(d, Vec3{0, 0, 0}, 0));
}

// ------------------------------------------------------------------ gate nodes

TEST_CASE("ScriptedFallback hands over exactly on the law's own fallback flag") {
  auto f = obj_base();
  f.scripted_fallback = true;
  CHECK(route("gate_defenders.xml", f) == "rl");
  f.scripted_fallback = false;
  CHECK(route("gate_defenders.xml", f) == "scripted");
  // the retired proxy no longer routes anything on its own
  f.intercept_feasible = false;
  CHECK(route("gate_defenders.xml", f) == "scripted");
}

TEST_CASE("sphere intercept: radius 0 is the point solve; >0 only ever helps") {
  Vec3 rel{-2, 1, 0}, v{-11, 0, 0};
  CHECK(lead_intercept_time_sphere(rel, v, 7.5, 0.0) == lead_intercept_time(rel, v, 7.5));
  auto t = lead_intercept_time_sphere(Vec3{15, 2, 0}, Vec3{-11, 0, 0}, 7.5, 1.4);
  REQUIRE(t.has_value());
  CHECK(norm(Vec3{15, 2, 0} + Vec3{-11, 0, 0} * (*t)) <= 7.5 * (*t) + 1.4 + 1e-9);
}

// Pins the negative result of the debounce experiment (Python mirror:
// test_games.py::test_defender_handover_is_immediate): requiring the fallback
// to persist k ticks before handover was a dose-response LOSS on the assault
// latency axis (bt_gated 0.71/0.59/0.49 at perfect link for k=1/2/3, nothing
// recovered under latency), because genuine fallback regimes are 20-30-tick
// streaks that do not flicker — a debounce is pure delay against an 11 m/s
// missile. The first fallback tick must hand over immediately. A missile
// receding at 11 m/s from a 7.5 m/s defender is unreachable even to the sphere.
namespace {
struct ConstCtl : Controller {
  double v;
  explicit ConstCtl(double x) : v(x) {}
  std::vector<double> act(const TeamView& view) override {
    return std::vector<double>(view.n_self() * 3, v);
  }
};

TeamView defender_view(double missile_vel_x) {
  TeamView v;
  v.team = Team::Pursuers;
  v.self_pos = {{0, 0, 3}};
  v.self_vel = {{0, 0, 0}};
  v.self_alive = {1};
  v.opp_pos = {{20, 0, 3}};
  v.opp_vel = {{missile_vel_x, 0, 0}};
  v.opp_alive = {1};
  v.vmax = 7.5;
  v.opp_vmax = 11.0;
  v.game = "assault";
  v.has_asset = true;
  v.asset_pos = {-5, 0, 3};
  return v;
}
}  // namespace

TEST_CASE("defender handover is immediate on the first fallback tick") {
  GuardDefenders law;
  ConstCtl rl(-0.75);
  GatedController gc(Team::Pursuers, &law, &rl,
                     std::string(PE_TREES_DIR) + "/gate_defenders.xml");
  TeamView receding = defender_view(11.0), inbound = defender_view(-11.0);
  std::vector<char> fb;
  law.plan(receding, fb);
  REQUIRE(fb[0]);
  law.plan(inbound, fb);
  REQUIRE_FALSE(fb[0]);

  CHECK(gc.act(inbound)[0] == law.act(inbound)[0]);  // intercept: scripted
  CHECK(gc.act(receding)[0] == -0.75);               // first fallback tick: RL
  CHECK(gc.mode_counts["rl"] == 1);
}

// This is the finding, pinned as a test: the tag instinct "close quarters is
// messy, hand it to the policy" costs 42 points in air defence (0.70 -> 0.12).
// The terminal endgame against a missile is a precise geometry problem the
// lead-intercept law owns, so a close-in defender with a feasible solution must
// stay scripted. See the ablation table in trees/gate_defenders.xml.
TEST_CASE("defender gate does NOT hand close quarters to the policy") {
  auto f = obj_base();
  f.dist_nearest = 1.0;      // point blank
  f.threat_time = 0.2;       // and the asset is about to be hit
  f.scripted_fallback = false;
  CHECK(route("gate_defenders.xml", f) == "scripted");
}

TEST_CASE("defender gate falls through to scripted by default") {
  CHECK(route("gate_defenders.xml", obj_base()) == "scripted");
}

TEST_CASE("attacker gate: a committed missile flies the guidance law regardless") {
  auto f = obj_base();
  f.asset_dist = 3.0;    // inside committed_range (8)
  f.dist_nearest = 1.0;  // an interceptor is right there — irrelevant now
  f.closing_rate = 9.0;
  CHECK(route("gate_attackers.xml", f) == "scripted");
}

TEST_CASE("attacker gate never routes to the policy — the ablation said so") {
  // Pins the measured result recorded in gate_attackers.xml and in
  // gating.py:_attacker_predicates. Both RL branches this profile used to carry
  // LOST breach rate once there was a real attacker policy to gate with:
  // 0.46/0.57 scripted -> 0.37/0.38 fully gated -> 0.33/0.35 policy only.
  // The missile side therefore hands over nothing. Python mirror:
  // test_games.py::test_attacker_gate_hands_over_nothing.
  auto f = obj_base();
  f.asset_dist = 30.0;   // not committed...
  f.dist_nearest = 1.0;  // ...an interceptor is right on top of us...
  f.closing_rate = 9.0;  // ...and closing hard. Still scripted.
  CHECK(route("gate_attackers.xml", f) == "scripted");
}

// ----------------------------------------------------------------- game specs

TEST_CASE("game specs mirror games.py: who wins a timeout") {
  // tag: the evaders survived the clock, so they win
  CHECK(game_tag().timeout_winner == Team::Evaders);
  // air defence: the *missiles* are the ones on a clock, so a timeout defends
  CHECK(game_assault().timeout_winner == Team::Pursuers);
  CHECK(game_escort().timeout_winner == Team::Pursuers);
}

TEST_CASE("only the objective games carry an asset, and only escort moves it") {
  CHECK_FALSE(game_tag().has_asset);
  CHECK(game_assault().has_asset);
  CHECK_FALSE(game_assault().asset.moving());
  CHECK(game_escort().asset.moving());
}

TEST_CASE("the escort asset transits its leg and then stops") {
  const auto g = game_escort();
  const double t_end = g.asset.leg_length() / g.asset.speed;
  CHECK(norm(g.asset.position_at(0.0) - g.asset.start) == doctest::Approx(0.0));
  CHECK(norm(g.asset.position_at(t_end) - g.asset.goal) == doctest::Approx(0.0));
  CHECK_FALSE(g.asset.arrived_at(t_end * 0.5));
  CHECK(g.asset.arrived_at(t_end));
  // clamped, not overshooting, once arrived
  CHECK(norm(g.asset.position_at(t_end * 2) - g.asset.goal) == doctest::Approx(0.0));
  CHECK(norm(g.asset.velocity_at(t_end * 2)) == doctest::Approx(0.0));
}

TEST_CASE("defenders are slower than the missiles — a tail chase is not a plan") {
  const auto g = game_assault();
  CHECK(g.arena.pursuer_dyn.vmax() < g.arena.evader_dyn.vmax());
}

// Pins the substep-capture policy AND the honest reason for it. Naively you'd
// say "missiles are fast, so they tunnel and tag doesn't" — that is false: per
// control step, tag closes 1.43 capture radii and assault only 0.79. What
// actually differs is the geometry. A defender cannot tail-chase a missile
// (it is slower, see the test above), so essentially every engagement is a
// high-offset CROSSING pass, and the chord cut through the capture sphere is far
// shorter than its diameter — those are the samples a once-per-step test misses.
// Tag's terminal phase is a low-offset stern chase where the chord is near
// maximal. Tag also stays once-per-step because its 40–50% scripted baseline is
// calibrated on that; see CLAUDE.md.
TEST_CASE("substep capture is on for the crossing-pass games, off for tag") {
  auto travel_per_step = [](const GameSpec& g) {
    const double closing = g.arena.pursuer_dyn.vmax() + g.arena.evader_dyn.vmax();
    return closing * g.arena.timestep * g.episode.control_repeat;
  };
  const auto assault = game_assault();
  const auto tag = game_tag();

  CHECK(assault.capture_substeps);
  CHECK(game_escort().capture_substeps);
  CHECK_FALSE(tag.capture_substeps);

  // the defender is the slower airframe, so it cannot convert to a stern chase
  CHECK(assault.arena.pursuer_dyn.vmax() < assault.arena.evader_dyn.vmax());
  CHECK(tag.arena.pursuer_dyn.vmax() > tag.arena.evader_dyn.vmax());

  // and a step of closing travel is a large enough slice of the sphere that a
  // grazing chord fits between two consecutive control-step samples
  CHECK(travel_per_step(assault) > 0.5 * assault.episode.capture_radius);
}

TEST_CASE("make_game rejects an unknown key rather than silently defaulting") {
  CHECK_THROWS_AS(make_game("nonsense"), std::runtime_error);
  CHECK(arena_asset_name("tag") == "arena.xml");
  CHECK(arena_asset_name("assault") == "arena_assault.xml");
}
