"""Export a trained SB3 policy to ONNX for the C++ runtime.

The C++ tick loop runs the policy through ONNX Runtime, so the network must be
serialised in a framework-neutral format. We export the *deterministic* action
head (obs -> mean action); the C++ side clips to [-1, 1] exactly as SB3's
``predict`` does. ``verify_parity`` checks the ONNX outputs match SB3 on random
observations before we trust it downstream.

    uv run python -m pursuit_evasion.train.export_onnx models/pursuer.zip models/pursuer.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


class _DeterministicActor(torch.nn.Module):
    """Deterministic action head of an SB3 ActorCritic policy as pure tensor ops.

    Computing the Gaussian mean directly (features -> policy MLP -> action_net)
    avoids the distribution/sampling code paths that the ONNX exporters cannot
    trace, and it IS the deterministic action for the DiagGaussian policy PPO uses.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def export_policy(model_path: str | Path, out_path: str | Path,
                  opset: int = 17) -> Path:
    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy.to("cpu").eval()
    obs_dim = int(model.observation_space.shape[0])

    wrapper = _DeterministicActor(policy)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    out_path = Path(out_path)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    ok, err = verify_parity(model, out_path)
    print(f"[onnx] exported {out_path}  obs_dim={obs_dim}  parity_max_err={err:.2e}  ok={ok}")
    return out_path


def verify_parity(model: PPO, onnx_path: str | Path, n: int = 256,
                  tol: float = 1e-4) -> tuple[bool, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    obs_dim = int(model.observation_space.shape[0])
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (n, obs_dim)).astype(np.float32)
    sb3_actions, _ = model.predict(obs, deterministic=True)
    onnx_actions = sess.run(["action"], {"obs": obs})[0]
    # SB3 predict clips to the action space; mirror that before comparing
    onnx_clipped = np.clip(onnx_actions, -1.0, 1.0)
    max_err = float(np.max(np.abs(sb3_actions - onnx_clipped)))
    return max_err < tol, max_err


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print("usage: export_onnx <model.zip> <out.onnx>")
        return 1
    export_policy(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
