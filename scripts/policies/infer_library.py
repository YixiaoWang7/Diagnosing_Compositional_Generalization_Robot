"""Predict raw-scale action trajectories from an NPZ of canonical raw tensors."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from robopolicy import InferenceConfig, load_policy


def main() -> None:
    """Parse filenames and invoke the public inference reconstruction path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", choices=("auto", "enabled", "disabled"), default="auto")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        batch = {key: torch.from_numpy(source[key]) for key in source.files}
    config = InferenceConfig.model_validate(
        {"compile": args.compile, "mode": args.compile_mode, "precision": args.precision}
    )
    inference = load_policy(args.checkpoint, device=args.device, inference_config=config)
    actions = inference.predict(batch)
    np.save(args.output, actions.to("cpu").numpy())


if __name__ == "__main__":
    main()
