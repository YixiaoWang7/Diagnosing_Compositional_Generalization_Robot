"""Measure checkpoint inference with isolated GPU workers and verified W&B logging."""

from __future__ import annotations

import argparse
from pathlib import Path

from robopolicy.training.inference_benchmark import InferenceBenchmarkConfig, run_benchmark_suite


def main() -> None:
    """Parse a benchmark specification and invoke the library evaluation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = InferenceBenchmarkConfig.model_validate_json(args.config.read_text())
    print(run_benchmark_suite(config))


if __name__ == "__main__":
    main()
