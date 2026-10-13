"""Run immutable job specifications in separate, explicitly configured processes."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def run_stage(job_path: Path, stage: str) -> None:
    """Launch one stage, retaining its output and exit status in the job directory."""
    job: dict[str, Any] = json.loads(job_path.read_text())
    role = "train" if stage == "train" else "eval"
    runtime = job["execution"]
    environment = os.environ.copy()
    if runtime[f"{role}_gpus"] is not None:
        environment["CUDA_VISIBLE_DEVICES"] = runtime[f"{role}_gpus"]
    command = [
        runtime[f"{role}_python"],
        "-m",
        "corl2026_cg",
        stage,
        "--job",
        str(job_path.resolve()),
    ]
    status_path = job_path.parent / f"{stage}_status.json"
    status: dict[str, Any] = {"stage": stage, "command": command, "status": "running"}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        with (job_path.parent / f"{stage}.log").open("a") as output:
            subprocess.run(
                command, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True
            )
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    else:
        status["status"] = "complete"
    finally:
        status_path.write_text(json.dumps(status, indent=2) + "\n")


def run_jobs(
    paths: list[Path], *, overlap: bool = False, eval_only: bool = False, skip_eval: bool = False
) -> None:
    """Train sequentially, optionally overlapping with one evaluation worker.

    Failed evaluations are observed before subsequent training launches. A failure
    cancels unstarted evaluations; any already running process is joined before exit.
    """
    if eval_only and skip_eval:
        raise ValueError("eval_only and skip_eval are mutually exclusive")
    if not overlap or eval_only or skip_eval:
        for path in paths:
            if not eval_only:
                run_stage(path, "train")
            if not skip_eval:
                run_stage(path, "evaluate")
        return
    pending: list[concurrent.futures.Future[None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            for path in paths:
                for future in pending:
                    if future.done():
                        future.result()
                run_stage(path, "train")
                # Bound backlog to one evaluation waiting behind the running one.
                if len(pending) >= 2:
                    pending.pop(0).result()
                for future in pending:
                    if future.done():
                        future.result()
                pending.append(executor.submit(run_stage, path, "evaluate"))
            for future in pending:
                future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
