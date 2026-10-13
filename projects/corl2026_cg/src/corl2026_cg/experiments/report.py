"""Aggregate explicit experiment identities without parsing seed suffixes."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _comparison(
    kind: str, directory: Path, job: dict[str, Any], result: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Identify comparable measurements independently of per-seed output paths.

    Actual rollout settings take precedence over the original job specification.
    Older results without those settings remain readable, but their declared-only
    provenance is explicit and they never merge with verified measurements.
    """
    if kind == "rollout":
        config_path = directory / "evaluation/eval_config_used.json"
        recorded = config_path.is_file()
        source = json.loads(config_path.read_text()) if recorded else job
        evaluation = {
            key: value
            for key, value in source.get("evaluation", {}).items()
            if key
            not in {
                "logging",
                "simulator_root",
                "device",
                "max_videos",
                "max_attention_videos",
                "fps",
            }
        }
        step = source.get("checkpoint_step", result.get("checkpoint_step"))
        if recorded and result.get("checkpoint_step", step) != step:
            raise ValueError(f"Checkpoint step differs between rollout artifacts: {directory}")
        settings = {
            "family": source.get("family"),
            "instruction": source.get("instruction"),
            "task_set": source.get("task_set"),
            "evaluation": evaluation or None,
            "checkpoint_step": step,
        }
        if "conditioning" in source:
            settings["conditioning"] = source["conditioning"]
            settings["active_profile"] = source.get("active_profile")
            settings["active_profile_digest"] = source.get("active_profile_digest")
        provenance_source = str(config_path) if recorded else "experiment.json fallback"
        status = "recorded" if recorded else "declared_only"
    else:
        fields = (
            "metric_seed",
            "sampling_seed",
            "requested_samples",
            "batch_size",
            "family",
            "instruction",
            "checkpoint_step",
        )
        settings = {key: result.get(key) for key in fields}
        # These fields further distinguish task membership and algorithm settings
        # when present, without requiring older summaries to invent provenance.
        settings.update(
            {
                key: result[key]
                for key in (
                    "algorithm",
                    "task_indices_paths",
                    "vocab_sizes",
                    "training_task_slugs",
                    "membership",
                    "task_universe",
                    "conditioning",
                    "active_profile",
                    "active_profile_digest",
                )
                if key in result
            }
        )
        if result.get("dataset_fingerprints"):
            settings["dataset_fingerprints"] = result["dataset_fingerprints"]
        elif "dataset_root" in result:
            settings["dataset_root"] = result["dataset_root"]
        provenance_source = str(directory / "losses.json")
        status = "recorded"
    missing = sorted(
        key
        for key, value in settings.items()
        if value is None and key not in {"active_profile", "active_profile_digest"}
    )
    if missing and status == "recorded":
        status = "incomplete"
    comparable = {"kind": kind, "settings": settings, "provenance": status}
    digest = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()[:12]
    return f"{kind}-{digest}", {
        "status": status,
        "source": provenance_source,
        "missing_fields": missing,
        "settings": settings,
    }


def _metrics(values: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten numeric summary metrics, excluding episode and task detail arrays."""
    flattened: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[name] = float(value)
    return flattened


def write_report(directories: list[Path], output: Path) -> dict[str, Any]:
    """Write per-run CSV, grouped JSON, and a readable Markdown result table."""
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in directories:
        job = json.loads((directory / "experiment.json").read_text())
        if job["name"] in seen:
            raise ValueError(f"Duplicate report input: {job['name']}")
        seen.add(job["name"])
        for kind, path in (
            ("rollout", directory / "evaluation/eval_summary.json"),
            ("loss", directory / "losses.json"),
        ):
            record = {
                "name": job["name"],
                "group": job["group"],
                "seed": job["seed"],
                "kind": kind,
                "path": str(path),
                "status": "missing",
            }
            results.append(record)
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            record["status"] = "complete"
            evaluation_identity, provenance = _comparison(kind, directory, job, result)
            record["evaluation_identity"] = evaluation_identity
            record["provenance"] = provenance
            design_metrics = result.get("design_metrics", job.get("design_metrics"))
            if design_metrics:
                record["design_metrics"] = design_metrics
            if "splits" in result:
                record["splits"] = {
                    key: {
                        "status": value.get(
                            "status",
                            "not_applicable" if value.get("n_episodes") == 0 else "complete",
                        ),
                        "num_samples": value.get("num_samples"),
                        "n_episodes": value.get("n_episodes"),
                        "missing_task_slugs": value.get("missing_task_slugs", []),
                    }
                    for key, value in result["splits"].items()
                }
            selected = {
                key: result[key]
                for key in ("aggregated", "aggregated_by_split", "splits", "per_task")
                if key in result
            }
            if kind == "rollout" and design_metrics:
                selected["design"] = design_metrics
            for metric, value in _metrics(selected).items():
                rows.append(
                    {
                        "name": job["name"],
                        "group": job["group"],
                        "evaluation_identity": evaluation_identity,
                        "seed": job["seed"],
                        "kind": kind,
                        "metric": metric,
                        "value": value,
                    }
                )
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        key = (row["group"], row["kind"], row["evaluation_identity"], row["metric"])
        groups.setdefault(key, []).append(row["value"])
    summaries = [
        {
            "group": group,
            "kind": kind,
            "evaluation_identity": evaluation_identity,
            "metric": metric,
            "seeds": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for (group, kind, evaluation_identity, metric), values in sorted(groups.items())
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_run.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "group", "seed", "kind", "evaluation_identity", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {"runs": sorted(seen), "results": results, "summaries": summaries}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CG experiment results",
        "",
        "Standard deviations are across training seeds.",
        "Different evaluation settings or checkpoint steps form separate comparison groups.",
        "",
        "| Experiment | Evaluation | Kind | Metric | Seeds | Mean | Std |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['group']} | {r['evaluation_identity']} | {r['kind']} | {r['metric']} | {r['seeds']} | "
        f"{r['mean']:.6g} | {r['std']:.6g} |"
        for r in summaries
    )
    if not summaries:
        lines.extend(["", "No completed evaluation metrics are available."])
    unavailable = [
        f"- {record['name']}: {split}: {', '.join(values['missing_task_slugs'])}"
        for record in results
        for split, values in record.get("splits", {}).items()
        if values.get("missing_task_slugs")
    ]
    if unavailable:
        lines.extend(["", "Tasks unavailable in the offline dataset:", "", *unavailable])
    missing = [
        f"- {record['name']}: {record['kind']}"
        for record in results
        if record["status"] == "missing"
    ]
    if missing:
        lines.extend(["", "Missing results:", "", *missing])
    incomplete = [
        f"- {record['name']}: {record['kind']} ({record['provenance']['status']})"
        for record in results
        if record.get("provenance", {}).get("status") in {"declared_only", "incomplete"}
    ]
    if incomplete:
        lines.extend(["", "Incomplete evaluation provenance:", "", *incomplete])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
