#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <behaviortree_cpp/bt_factory.h>

#include <string>

#include "pe/bt_nodes.hpp"
#include "pe/config.hpp"
#include "pe/features.hpp"

using namespace pe;

namespace {

// Tick a single condition inside `Fallback[ Sequence[COND, SetMode rl], SetMode scripted ]`
// so the resulting mode reports whether the condition succeeded.
std::string ticked_mode(const std::string& cond, const AgentFeatures& f,
                        const GateThresholds& thr = {}) {
  BT::BehaviorTreeFactory factory;
  register_gate_nodes(factory);
  std::string xml =
      "<root BTCPP_format=\"4\"><BehaviorTree ID=\"T\"><Fallback>"
      "<Sequence><" + cond + "/><SetMode mode=\"rl\"/></Sequence>"
      "<SetMode mode=\"scripted\"/></Fallback></BehaviorTree></root>";
  auto tree = factory.createTreeFromText(xml);
  tree.rootBlackboard()->set("features", f);
  tree.rootBlackboard()->set("thresholds", thr);
  tree.rootBlackboard()->set("mode", std::string("scripted"));
  tree.tickOnce();
  return tree.rootBlackboard()->get<std::string>("mode");
}

AgentFeatures base() {
  AgentFeatures f;
  f.alive = true;
  f.dist_nearest = 20;
  f.dist_second = 25;
  f.n_live_others = 2;
  f.closing_rate = 0;
  f.target_lateral = 0;
  f.intercept_ahead = false;
  f.contested = false;
  return f;
}

}  // namespace

TEST_CASE("CloseQuarters fires below threshold, not above") {
  auto f = base();
  f.dist_nearest = 2.0;
  CHECK(ticked_mode("CloseQuarters", f) == "rl");
  f.dist_nearest = 5.0;
  CHECK(ticked_mode("CloseQuarters", f) == "scripted");
}

TEST_CASE("TargetJuking needs high lateral AND close range") {
  auto f = base();
  f.target_lateral = 3.0;
  f.dist_nearest = 5.0;
  CHECK(ticked_mode("TargetJuking", f) == "rl");
  f.target_lateral = 1.0;  // not juking
  CHECK(ticked_mode("TargetJuking", f) == "scripted");
  f.target_lateral = 3.0;
  f.dist_nearest = 9.0;    // too far to matter
  CHECK(ticked_mode("TargetJuking", f) == "scripted");
}

TEST_CASE("Contested requires ambiguity flag and >=2 live") {
  auto f = base();
  f.contested = true;
  f.n_live_others = 2;
  CHECK(ticked_mode("Contested", f) == "rl");
  f.contested = false;
  CHECK(ticked_mode("Contested", f) == "scripted");
  f.contested = true;
  f.n_live_others = 1;
  CHECK(ticked_mode("Contested", f) == "scripted");
}

TEST_CASE("CleanIntercept mirrors intercept_ahead") {
  auto f = base();
  f.intercept_ahead = true;
  CHECK(ticked_mode("CleanIntercept", f) == "rl");  // condition succeeded
  f.intercept_ahead = false;
  CHECK(ticked_mode("CleanIntercept", f) == "scripted");
}

TEST_CASE("Evader conditions: ThreatClose and ThreatClosingFast") {
  auto f = base();
  f.dist_nearest = 3.0;  // < evader_danger (4)
  CHECK(ticked_mode("ThreatClose", f) == "rl");
  f.dist_nearest = 10.0;
  CHECK(ticked_mode("ThreatClose", f) == "scripted");
  f.closing_rate = 4.0;  // > closing_fast (3)
  CHECK(ticked_mode("ThreatClosingFast", f) == "rl");
  f.closing_rate = 1.0;
  CHECK(ticked_mode("ThreatClosingFast", f) == "scripted");
}

TEST_CASE("full pursuer gate routes each regime correctly") {
  BT::BehaviorTreeFactory factory;
  register_gate_nodes(factory);
  std::string xml =
      "<root BTCPP_format=\"4\"><BehaviorTree ID=\"T\"><Fallback>"
      "<Sequence><CloseQuarters/><SetMode mode=\"rl\"/></Sequence>"
      "<Sequence><TargetJuking/><SetMode mode=\"rl\"/></Sequence>"
      "<Sequence><Contested/><SetMode mode=\"rl\"/></Sequence>"
      "<Sequence><CleanIntercept/><SetMode mode=\"scripted\"/></Sequence>"
      "<SetMode mode=\"scripted\"/></Fallback></BehaviorTree></root>";
  auto tree = factory.createTreeFromText(xml);
  auto run = [&](const AgentFeatures& f) {
    tree.rootBlackboard()->set("features", f);
    tree.rootBlackboard()->set("thresholds", GateThresholds{});
    tree.rootBlackboard()->set("mode", std::string("scripted"));
    tree.tickOnce();
    return tree.rootBlackboard()->get<std::string>("mode");
  };
  auto f = base();
  f.dist_nearest = 1.5;  // close quarters
  CHECK(run(f) == "rl");
  f = base();
  f.intercept_ahead = true;  // clean, open geometry
  CHECK(run(f) == "scripted");
  f = base();  // far, nothing special
  CHECK(run(f) == "scripted");
}
