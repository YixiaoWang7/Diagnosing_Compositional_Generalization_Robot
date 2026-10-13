"""Training entry points enforce active profile contracts before trainer creation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from robopolicy import RunConfig
from test_active_experiments import active_spec

from corl2026_cg.cli import main
from corl2026_cg.experiments.config import resolve_jobs
from corl2026_cg.instructions.contracts import validate_finetune_conditioning


@pytest.mark.parametrize("source_is_active", [False, True])
def test_static_active_finetuning_requires_explicit_semantic_compatibility(
    tmp_path: Path, source_is_active: bool
) -> None:
    """A training restart cannot silently reinterpret static and reactive queries."""
    spec = active_spec(tmp_path)
    active = RunConfig.model_validate(resolve_jobs(spec)[0]["run"])
    static = RunConfig.model_validate(
        resolve_jobs(spec.model_copy(update={"conditioning": "static"}))[0]["run"]
    )
    target, source = (static, active) if source_is_active else (active, static)
    with pytest.raises(ValueError, match="matching active instruction profiles"):
        validate_finetune_conditioning(target, source)


def test_resume_revalidates_dataset_profile_before_loading_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume path must enforce the same prepared-label contract as fresh training."""
    config = resolve_jobs(active_spec(tmp_path))[0]["run"]
    monkeypatch.setattr("robopolicy.read_checkpoint", lambda path: {"run_config": config})

    def reject_profile(*args: object, **kwargs: object) -> None:
        """Simulate changed label provenance in a selected demonstration."""
        raise ValueError("Active instruction profile digest mismatch")

    def forbidden_trainer(*args: object, **kwargs: object) -> None:
        """Trainer construction must not precede semantic validation."""
        pytest.fail("Trainer resumed before profile validation")

    monkeypatch.setattr("corl2026_cg.instructions.validate_dataset_profile", reject_profile)
    monkeypatch.setattr("robopolicy.Trainer.resume", forbidden_trainer)
    with pytest.raises(ValueError, match="profile digest mismatch"):
        main(["train", "--resume", str(tmp_path / "checkpoint.pt")])


def test_cli_finetuning_checks_source_profile_before_constructing_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both supported active profiles have 12 tokens but different token meanings."""
    target = resolve_jobs(active_spec(tmp_path, family="l4_2s"))[0]["run"]
    source = resolve_jobs(active_spec(tmp_path, family="l5"))[0]["run"]
    config_path = tmp_path / "target.json"
    config_path.write_text(json.dumps(target))
    monkeypatch.setattr("robopolicy.read_checkpoint", lambda path: {"run_config": source})
    monkeypatch.setattr(
        "corl2026_cg.instructions.validate_dataset_profile", lambda *args, **kwargs: None
    )

    def forbidden_trainer(*args: object, **kwargs: object) -> None:
        """Matching native vocabulary width is insufficient for fine-tuning."""
        pytest.fail("Trainer constructed before validating source profile")

    monkeypatch.setattr("robopolicy.Trainer.finetune", forbidden_trainer)
    with pytest.raises(ValueError, match="matching active instruction profiles"):
        main(
            ["train", "--config", str(config_path), "--finetune-from", str(tmp_path / "source.pt")]
        )
