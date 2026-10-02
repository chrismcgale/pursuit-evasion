#include "pe/gate.hpp"

#include <behaviortree_cpp/loggers/groot2_publisher.h>

#include "pe/bt_nodes.hpp"
#include "pe/features.hpp"
#include "pe/scripted.hpp"

namespace pe {

GatedController::GatedController(Team team, Controller* scripted, Controller* rl,
                                const std::string& xml_path, GateThresholds thr,
                                bool enable_groot, unsigned groot_port)
    : team_(team), scripted_(scripted), rl_(rl), thr_(thr) {
  name = "bt_gated";
  register_gate_nodes(factory_);
  tree_ = std::make_unique<BT::Tree>(factory_.createTreeFromFile(xml_path));
  tree_->rootBlackboard()->set("thresholds", thr_);
  tree_->rootBlackboard()->set("mode", std::string("scripted"));
  if (enable_groot)
    publisher_ = std::make_unique<BT::Groot2Publisher>(*tree_, groot_port);
}

GatedController::~GatedController() = default;

void GatedController::reset() {
  scripted_->reset();
  rl_->reset();
  mode_counts = {{"scripted", 0}, {"rl", 0}};
}

std::vector<double> GatedController::act(const TeamView& v) {
  // A scripted law that reports its own regime (GuardDefenders::plan) is asked
  // once per tick; its flags feed `scripted_fallback` (mirrors gating.py).
  std::vector<char> fallback;
  auto* law = dynamic_cast<GuardDefenders*>(scripted_);
  std::vector<double> s = law ? law->plan(v, fallback) : scripted_->act(v);
  std::vector<double> r = rl_->act(v);
  int n = v.n_self();
  std::vector<double> out(n * 3, 0.0);

  auto bb = tree_->rootBlackboard();
  bb->set("thresholds", thr_);
  for (int i = 0; i < n; ++i) {
    if (!v.self_alive[i]) continue;
    AgentFeatures f = agent_features(v, i);
    if (law) f.scripted_fallback = fallback[i] != 0;
    bb->set("features", f);
    bb->set("mode", std::string("scripted"));
    tree_->tickOnce();
    std::string mode = bb->get<std::string>("mode");
    mode_counts[mode] += 1;
    const std::vector<double>& src = (mode == "rl") ? r : s;
    for (int k = 0; k < 3; ++k) out[i * 3 + k] = clamp(src[i * 3 + k], -1.0, 1.0);
  }
  return out;
}

}  // namespace pe
