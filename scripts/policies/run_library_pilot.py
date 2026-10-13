"""Run checkpoint-stop-resume-validation-inference for a supplied pilot config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from robopolicy.checkpoints.inference import load_policy
from robopolicy.config.run import load_config
from robopolicy.datasets.factory import build_loader
from robopolicy.training.logging import log_evaluation
from robopolicy.training.trainer import Trainer, validate


def main() -> None:
    """Write observable acceptance results after exercising the full training path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.fit(until_step=args.checkpoint_step)
    identity = trainer.logger.identity
    checkpoint = Path(config.logging.out_dir) / "latest.pt"
    resumed = Trainer.resume(checkpoint)
    assert resumed.completed_updates == args.checkpoint_step
    assert resumed.logger.identity == identity
    resumed.fit()
    with build_loader(resumed.validation_dataset, config.loader) as loader:
        metrics = validate(resumed.policy, loader, resumed.packer, config.training.device)
    with build_loader(resumed.dataset, config.loader) as loader:
        batch = next(iter(loader))
    raw = {key: value for key, value in batch.items() if key != "actions"}
    noise = torch.zeros(
        batch["state"].shape[0],
        config.policy.action_steps,
        config.policy.action_dim,
        device=config.training.device,
    )
    resumed.policy.eval()
    expected = resumed.packer.actions_to_raw(
        resumed.policy.predict(resumed.packer(raw, device=config.training.device), noise)
    )
    inference = load_policy(checkpoint, device=config.training.device)
    actual = inference.predict(raw, noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    result = {
        "completed_updates": resumed.completed_updates,
        "checkpoint_step": args.checkpoint_step,
        "data_epoch": resumed.data_epoch,
        "wandb": identity.model_dump() if identity else None,
        "prediction_shape": list(actual.shape),
        "prediction_parity": "exact",
        "validation": metrics,
        "dataset_fingerprint": resumed.dataset.fingerprint,
        "selection": resumed.dataset.selection,
        "config": config.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    evaluation_identity = log_evaluation(
        config.logging,
        config.model_dump(mode="json"),
        metrics,
        args.output,
        training_identity=identity,
        checkpoint_step=resumed.completed_updates,
    )
    result["evaluation_wandb"] = evaluation_identity.model_dump() if evaluation_identity else None
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"completed_updates": resumed.completed_updates, "validation": metrics}),
        flush=True,
    )


if __name__ == "__main__":
    main()
