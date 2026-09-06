#pragma once
#include <behaviortree_cpp/bt_factory.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pe/config.hpp"
#include "pe/controller.hpp"

namespace BT {
class Groot2Publisher;
}

namespace pe {

// Mirrors src/pursuit_evasion/bt/gating.py:GatedController, but the tree is a
// real BehaviorTree.CPP tree loaded from XML (Groot2-editable). Each tick it
// evaluates both controllers, then ticks the tree per agent to choose which
// sub-action to use, recording per-mode usage for the writeup.
class GatedController : public Controller {
 public:
  GatedController(Team team, Controller* scripted, Controller* rl,
                  const std::string& xml_path, GateThresholds thr = {},
                  bool enable_groot = false, unsigned groot_port = 1667);
  ~GatedController();

  void reset() override;
  std::vector<double> act(const TeamView& v) override;

  std::map<std::string, int> mode_counts{{"scripted", 0}, {"rl", 0}};

 private:
  Team team_;
  Controller* scripted_;
  Controller* rl_;
  GateThresholds thr_;
  BT::BehaviorTreeFactory factory_;
  std::unique_ptr<BT::Tree> tree_;
  std::unique_ptr<BT::Groot2Publisher> publisher_;
};

}  // namespace pe
