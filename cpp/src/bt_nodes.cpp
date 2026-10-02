#include "pe/bt_nodes.hpp"

#include <string>

#include "pe/config.hpp"
#include "pe/features.hpp"

using BT::NodeStatus;

namespace pe {

namespace {

// Small helper base: pulls the current AgentFeatures + GateThresholds that the
// GatedController writes to the blackboard before each per-agent tick.
struct GateCond : BT::ConditionNode {
  GateCond(const std::string& n, const BT::NodeConfig& c) : BT::ConditionNode(n, c) {}
  static BT::PortsList providedPorts() { return {}; }
  AgentFeatures feat() { return config().blackboard->get<AgentFeatures>("features"); }
  GateThresholds thr() { return config().blackboard->get<GateThresholds>("thresholds"); }
};

#define GATE_CONDITION(NAME, EXPR)                                        \
  struct NAME : GateCond {                                                \
    NAME(const std::string& n, const BT::NodeConfig& c) : GateCond(n, c) {} \
    NodeStatus tick() override {                                          \
      const AgentFeatures f = feat();                                     \
      const GateThresholds t = thr();                                     \
      (void)f; (void)t;                                                   \
      return (EXPR) ? NodeStatus::SUCCESS : NodeStatus::FAILURE;          \
    }                                                                     \
  };

// Pursuer-side predicates (mirror gating.py:_pursuer_predicates)
GATE_CONDITION(CloseQuarters, f.dist_nearest < t.close_quarters)
GATE_CONDITION(TargetJuking,
               f.target_lateral > t.juke_lateral && f.dist_nearest < t.juke_min_dist)
GATE_CONDITION(Contested, f.contested && f.n_live_others >= 2)
GATE_CONDITION(CleanIntercept, f.intercept_ahead)
// Evader-side predicates (mirror gating.py:_evader_predicates)
GATE_CONDITION(ThreatClose, f.dist_nearest < t.evader_danger)
GATE_CONDITION(ThreatClosingFast, f.closing_rate > t.closing_fast)
// Defender-side predicates (mirror gating.py:_defender_predicates). The scripted
// law has no answer when no intercept exists — it parks on a static gate point —
// so that is where the learned branch earns its keep. The handover is immediate
// by measurement: debouncing it k ticks was a dose-response loss (0.71/0.59/0.49
// at perfect link for k=1/2/3) recovering nothing under latency — see
// _defender_predicates' docstring in gating.py for the full table.
// Shipped: hand over exactly when the defender law itself is in its gate-point
// fallback (GuardDefenders::plan). InterceptInfeasible below was the proxy it
// replaced — point solve, nearest threat — and over the sphere-solving law it
// LOSES (-0.13 assault); kept registered so that ablation reproduces from Groot2.
GATE_CONDITION(ScriptedFallback, f.scripted_fallback)
GATE_CONDITION(CleanInterceptScripted, !f.scripted_fallback && f.intercept_ahead)
GATE_CONDITION(InterceptInfeasible, !f.intercept_feasible)
GATE_CONDITION(ThreatImminent, f.threat_time < t.threat_imminent)
GATE_CONDITION(DefenderCloseQuarters, f.dist_nearest < t.defender_close)
GATE_CONDITION(CleanInterceptFeasible, f.intercept_feasible && f.intercept_ahead)
// Attacker-side predicate (mirror gating.py:_attacker_predicates). Inside the
// terminal run-in there is no lateral authority left to spend on cleverness, so
// the guidance law owns it regardless of threats.
GATE_CONDITION(Committed, f.asset_dist < t.committed_range)

struct SetMode : BT::SyncActionNode {
  SetMode(const std::string& n, const BT::NodeConfig& c) : BT::SyncActionNode(n, c) {}
  static BT::PortsList providedPorts() { return {BT::InputPort<std::string>("mode")}; }
  NodeStatus tick() override {
    std::string m = "scripted";
    getInput("mode", m);
    config().blackboard->set("mode", m);
    return NodeStatus::SUCCESS;
  }
};

}  // namespace

void register_gate_nodes(BT::BehaviorTreeFactory& f) {
  f.registerNodeType<CloseQuarters>("CloseQuarters");
  f.registerNodeType<TargetJuking>("TargetJuking");
  f.registerNodeType<Contested>("Contested");
  f.registerNodeType<CleanIntercept>("CleanIntercept");
  f.registerNodeType<ThreatClose>("ThreatClose");
  f.registerNodeType<ThreatClosingFast>("ThreatClosingFast");
  f.registerNodeType<ScriptedFallback>("ScriptedFallback");
  f.registerNodeType<CleanInterceptScripted>("CleanInterceptScripted");
  f.registerNodeType<InterceptInfeasible>("InterceptInfeasible");
  f.registerNodeType<ThreatImminent>("ThreatImminent");
  f.registerNodeType<DefenderCloseQuarters>("DefenderCloseQuarters");
  f.registerNodeType<CleanInterceptFeasible>("CleanInterceptFeasible");
  f.registerNodeType<Committed>("Committed");
  f.registerNodeType<SetMode>("SetMode");
}

}  // namespace pe
