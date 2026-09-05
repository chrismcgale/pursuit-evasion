#pragma once
#include <behaviortree_cpp/bt_factory.h>

namespace pe {

// Registers the gate's condition + action nodes so a tree XML (authored/edited
// in Groot2) can be loaded. Nodes read `features` (AgentFeatures) and
// `thresholds` (GateThresholds) from the blackboard and write `mode`
// ("scripted"/"rl"). See src/pursuit_evasion/bt/gating.py for the reference.
void register_gate_nodes(BT::BehaviorTreeFactory& factory);

}  // namespace pe
