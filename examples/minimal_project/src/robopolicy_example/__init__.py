"""An independently packaged application consuming only public RoboPolicy APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from robopolicy import Trainer, load_config


def make_dataset(root: Path) -> None:
    """Generate aligned state/action trajectories for a small deterministic example."""
    root.mkdir(parents=True, exist_ok=True)
    demos = []
    with h5py.File(root / "episodes.h5", "w") as handle:
        for index in range(20):
            key = f"episode-{index}"
            demos.append({"demo_key": key, "length": 32})
            state = np.full((32, 1), index / 20, dtype=np.float32)
            handle[f"data/{key}/state"] = state
            handle[f"data/{key}/actions"] = state * 0.5
    (root / "task.json").write_text(
        json.dumps(
            {
                "task": "example trajectories",
                "shards": [{"shard_id": 0, "hdf5": "episodes.h5", "demos": demos}],
            }
        )
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps({"tasks": [{"task_slug": "example", "manifest": "task.json"}]})
    )


def main() -> None:
    """Generate toy data, then delegate training to the public library trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    make_dataset(Path(config.dataset.root))
    Trainer(config).fit()
