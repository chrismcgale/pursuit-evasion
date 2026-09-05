#pragma once
#include <memory>
#include <string>
#include <vector>

#include "pe/controller.hpp"
#include "pe/observations.hpp"

namespace Ort {
class Env;
class Session;
}  // namespace Ort

namespace pe {

// Loads a policy exported by train/export_onnx.py and runs it through ONNX
// Runtime. Input "obs" (float32, [1, obs_dim]) -> output "action" ([1, act_dim]).
class OnnxPolicy {
 public:
  explicit OnnxPolicy(const std::string& model_path);
  ~OnnxPolicy();
  std::vector<double> action(const std::vector<float>& obs);  // clipped to [-1,1]
  int obs_dim() const { return obs_dim_; }
  int act_dim() const { return act_dim_; }

 private:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  int obs_dim_ = 0, act_dim_ = 0;
};

// Mirrors scripted/base.py:RLController (deterministic action from the ONNX head).
struct RLController : Controller {
  OnnxPolicy* policy;
  explicit RLController(OnnxPolicy* p, std::string nm = "rl") : policy(p) { name = nm; }
  std::vector<double> act(const TeamView& v) override {
    return policy->action(build_team_obs(v));
  }
};

}  // namespace pe
