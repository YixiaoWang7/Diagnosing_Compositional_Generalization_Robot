"""Verify a wheel in an isolated interpreter with only the requested extras."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def check_core(wheel: Path | None = None) -> None:
    """Exercise typed inputs, loss, gradients, and prediction without integrations."""
    import robopolicy

    forbidden = {"torch", "cv2", "torchcodec", "h5py", "wandb", "transformers", "torchvision"}
    assert not forbidden.intersection(sys.modules), "Cold import initialized an integration"
    location = str(Path(robopolicy.__file__).resolve())
    assert location.startswith(str(wheel)) if wheel else "site-packages" in location
    import torch

    from robopolicy import PolicyConfig, PolicyInput, build_policy

    config = PolicyConfig(
        state_dim=3,
        action_dim=2,
        obs_steps=2,
        action_steps=4,
        fusion={"dim": 16, "heads": 2, "layers": 1, "ff_dim": 32},
        head={
            "type": "transformer",
            "dim": 16,
            "heads": 2,
            "layers": 1,
            "ff_dim": 32,
            "time_dim": 8,
        },
    )
    policy = build_policy(config)
    inputs = PolicyInput(state=torch.zeros(2, 2, 3), actions=torch.zeros(2, 4, 2))
    torch.autograd.backward(policy.loss(inputs))
    assert any(parameter.grad is not None for parameter in policy.parameters())
    prediction = policy.predict(inputs)
    assert prediction.shape == (2, 4, 2) and torch.isfinite(prediction).all()
    assert not (forbidden - {"torch"}).intersection(sys.modules)


def check_decoder(mode: str) -> None:
    """Decode known real FFV1 frames with each sampler using one installed backend."""
    import h5py
    import numpy as np
    import torch

    from robopolicy import DatasetConfig, FeatureSchema, build_dataset, build_loader

    other = "torchcodec" if mode == "opencv" else "cv2"
    assert importlib.util.find_spec(other) is None, f"Unexpected second decoder: {other}"
    with tempfile.TemporaryDirectory(prefix="robopolicy-install-") as directory:
        root = Path(directory)
        frames = np.empty((6, 16, 16, 3), dtype=np.uint8)
        frames[:, :, :, 0] = np.arange(6)[:, None, None]
        frames[:, :, :, 1] = 80
        frames[:, :, :, 2] = 200
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "16x16",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                str(root / "camera.avi"),
            ],
            input=frames.tobytes(),
            check=True,
        )
        with h5py.File(root / "episode.h5", "w") as handle:
            handle["data/demo/state"] = np.arange(6, dtype=np.float32)[:, None]
            handle["data/demo/actions"] = np.arange(6, dtype=np.float32)[:, None]
        (root / "camera.json").write_text(
            json.dumps(
                {
                    "frames": 6,
                    "segments": [{"demo_key": "demo", "frame_start": 0, "frame_count": 6}],
                }
            )
        )
        (root / "task.json").write_text(
            json.dumps(
                {
                    "shards": [
                        {
                            "shard_id": 0,
                            "hdf5": "episode.h5",
                            "demos": [{"demo_key": "demo", "length": 6}],
                            "videos": {"camera": {"mp4": "camera.avi", "sidecar": "camera.json"}},
                        }
                    ]
                }
            )
        )
        (root / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "tasks": [{"task_slug": "opaque", "manifest": "task.json"}],
                }
            )
        )
        schema = FeatureSchema(
            features=(
                {"key": "state", "shape": [1], "paths": ["state"], "history": 1},
                {"key": "actions", "shape": [1], "paths": ["actions"], "future": 1},
                {
                    "key": "images",
                    "shape": [3, 16, 16],
                    "source": "video",
                    "cameras": ["camera"],
                    "dtype": "uint8",
                    "history": 1,
                },
            )
        )
        for strategy in ("window_major", "global_pass"):
            dataset = build_dataset(
                DatasetConfig(
                    root=str(root),
                    video={"decoder": mode},
                    selection={"val_fraction": 0},
                    sampling={"strategy": strategy, "k_passes": 8},
                ),
                schema,
            )
            with build_loader(dataset) as loader:
                batch = next(iter(loader))
            assert batch["images"].shape == (5, 2, 1, 3, 16, 16)
            assert torch.equal(batch["images"][:, :, 0, 0, 0, 0].float(), batch["state"][:, :, 0])
            assert torch.all(batch["images"][:, :, 0, 1] == 80)
            assert torch.all(batch["images"][:, :, 0, 2] == 200)
            assert dataset.resources["live_handles"] == 0


def main() -> None:
    """Run the selected isolated wheel gate and print machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("core", "opencv", "torchcodec"), required=True)
    parser.add_argument("--wheel", type=Path, help="Check a rebuilt wheel without reinstalling")
    args = parser.parse_args()
    wheel = args.wheel.resolve() if args.wheel else None
    if wheel is not None:
        sys.path.insert(0, str(wheel))
    check_core(wheel)
    if args.mode == "core":
        assert all(
            importlib.util.find_spec(name) is None
            for name in (
                "cv2",
                "h5py",
                "torchcodec",
                "wandb",
                "transformers",
                "torchvision",
            )
        )
    else:
        check_decoder(args.mode)
    print(
        json.dumps(
            {
                "status": "passed",
                "wheel": str(wheel) if wheel else None,
                "mode": args.mode,
                "python": sys.version,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("robopolicy", "torch", "numpy")
                },
                "executable": sys.executable,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
