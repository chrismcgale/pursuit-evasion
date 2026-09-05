#include "pe/onnx_policy.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <stdexcept>

#include "pe/vec3.hpp"

namespace pe {

OnnxPolicy::OnnxPolicy(const std::string& model_path) {
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "pe_policy");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);

  // read the fixed feature dims from the graph (batch axis is dynamic).
  // Keep the TypeInfo objects alive: GetTensorTypeAndShapeInfo() returns an
  // unowned view into them, so binding to a temporary would dangle.
  Ort::TypeInfo in_type = session_->GetInputTypeInfo(0);
  Ort::TypeInfo out_type = session_->GetOutputTypeInfo(0);
  auto in_shape = in_type.GetTensorTypeAndShapeInfo().GetShape();
  auto out_shape = out_type.GetTensorTypeAndShapeInfo().GetShape();
  obs_dim_ = (int)in_shape.back();
  act_dim_ = (int)out_shape.back();
}

OnnxPolicy::~OnnxPolicy() = default;

std::vector<double> OnnxPolicy::action(const std::vector<float>& obs) {
  if ((int)obs.size() != obs_dim_)
    throw std::runtime_error("obs size mismatch for ONNX policy");
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<int64_t, 2> shape{1, obs_dim_};
  Ort::Value in = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(obs.data()), obs.size(), shape.data(), shape.size());

  const char* in_names[] = {"obs"};
  const char* out_names[] = {"action"};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                               out_names, 1);
  const float* a = outputs[0].GetTensorData<float>();
  std::vector<double> action(act_dim_);
  for (int i = 0; i < act_dim_; ++i) action[i] = clamp((double)a[i], -1.0, 1.0);
  return action;
}

}  // namespace pe
