"""
export_policy_to_json.py
=========================

Exports a PolicyNetwork checkpoint (.pt) to a plain JSON file the
browser can fetch/embed, for use by policy.js's pure-JS forward
pass. No PyTorch needed at inference time in the browser.

Usage
-----
    python3 export_policy_to_json.py \
        --checkpoint phase2_policy_best.pt \
        --out phase2_policy_best.json
"""

import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu")

    expected_keys = [
        "trunk.0.weight", "trunk.0.bias",
        "trunk.2.weight", "trunk.2.bias",
        "policy_head.weight", "policy_head.bias",
        "value_head.weight", "value_head.bias",
    ]
    missing = [k for k in expected_keys if k not in sd]
    if missing:
        raise ValueError(f"Checkpoint missing expected keys: {missing}")

    payload = {
        "obs_dim": sd["trunk.0.weight"].shape[1],
        "hidden_dim": sd["trunk.0.weight"].shape[0],
        "action_dim": sd["policy_head.weight"].shape[0],
        "activation": "relu",
        # Weight matrices are stored row-major as [out_features][in_features],
        # matching nn.Linear's weight layout exactly (y = W @ x + b).
        "trunk0_weight": sd["trunk.0.weight"].tolist(),
        "trunk0_bias": sd["trunk.0.bias"].tolist(),
        "trunk2_weight": sd["trunk.2.weight"].tolist(),
        "trunk2_bias": sd["trunk.2.bias"].tolist(),
        "policy_head_weight": sd["policy_head.weight"].tolist(),
        "policy_head_bias": sd["policy_head.bias"].tolist(),
        "value_head_weight": sd["value_head.weight"].tolist(),
        "value_head_bias": sd["value_head.bias"].tolist(),
    }

    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"Exported {args.checkpoint} -> {args.out}")
    print(f"  obs_dim={payload['obs_dim']} hidden_dim={payload['hidden_dim']} "
          f"action_dim={payload['action_dim']}")


if __name__ == "__main__":
    main()
