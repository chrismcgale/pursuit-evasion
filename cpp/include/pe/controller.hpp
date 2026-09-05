#pragma once
#include <string>
#include <vector>

#include "pe/sim.hpp"

namespace pe {

// Uniform interface: map a team view to a flat action (n_self*3), each in [-1,1].
// Mirrors src/pursuit_evasion/scripted/base.py:Controller.
struct Controller {
  std::string name = "controller";
  virtual std::vector<double> act(const TeamView& view) = 0;
  virtual void reset() {}
  virtual ~Controller() = default;
};

}  // namespace pe
