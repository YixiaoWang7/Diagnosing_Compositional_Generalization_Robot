"""One training loop with finite data epochs and recoverable optimizer updates."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from robopolicy.checkpoints.serialization import (
    SCHEMA_VERSION,
    capture_rng,
    read_checkpoint,
    restore_rng,
    save_checkpoint,
)
from robopolicy.config.run import RunConfig
from robopolicy.datasets.factory import PolicyLoader, build_dataset, build_loader
from robopolicy.datasets.statistics import fit_statistics
from robopolicy.policies.flow import FlowPolicy, build_policy
from robopolicy.processing.normalization import Statistics
from robopolicy.processing.packing import InputPacker
from robopolicy.training.logging import MetricsLogger, RunIdentity


def validate(
    policy: FlowPolicy,
    batches: Iterable[dict[str, Any]],
    packer: InputPacker,
    device: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Average validation loss by sample count and restore mode even on exceptions."""
    previous_mode = policy.training
    # Validation randomness is fixed and does not perturb subsequent training.
    rng = capture_rng(device)
    total_loss, total_samples = 0.0, 0
    policy.eval()
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            for index, batch in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                inputs = packer(batch, device=device, require_actions=True)
                count = inputs.state.shape[0]
                total_loss += float(policy.loss(inputs).item()) * count
                total_samples += count
    finally:
        policy.train(previous_mode)
        restore_rng(rng, device)
    if total_samples == 0:
        raise ValueError("Validation has no eligible samples")
    return {"val/loss": total_loss / total_samples, "val/samples": float(total_samples)}


class Trainer:
    """Own training state, data epochs, validation, metrics, and resource shutdown.

    Recovery resumes the saved training configuration at the next optimizer
    update, starting data at the next logical epoch. Prefetched batches and the
    exact uninterrupted trajectory are intentionally not replayed.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        _checkpoint: dict[str, Any] | None = None,
        _initialize_pretrained: bool = True,
    ) -> None:
        """Set up one CPU or single-GPU run, optionally restoring a validated state."""
        self.config = config
        self.device = config.training.device
        device = torch.device(self.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported training devices are CPU and a single CUDA GPU")
        if device.type == "cuda" and (
            not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()
        ):
            raise ValueError(f"Requested CUDA device is unavailable: {device}")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.dataset = build_dataset(config.dataset, config.features, "train")
        self.validation_dataset = build_dataset(config.dataset, config.features, "val")
        if len(self.dataset) == 0:
            raise ValueError("Training selection has no eligible samples")
        if _checkpoint is not None:
            if self.dataset.fingerprint != _checkpoint[
                "dataset_fingerprint"
            ] or self.dataset.selection != tuple(tuple(item) for item in _checkpoint["selection"]):
                raise ValueError(
                    "Dataset identity or resolved training selection differs from checkpoint"
                )
            # read_checkpoint already checked configuration and normalization provenance.
            statistics = Statistics.model_validate(_checkpoint["statistics"])
        else:
            statistics = fit_statistics(
                self.dataset, Path(config.logging.out_dir) / "normalization.json"
            )
        self.packer = InputPacker(config.features, statistics, config.processing)
        self.policy = build_policy(
            config.policy, initialize_pretrained=_checkpoint is None and _initialize_pretrained
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_multiplier)
        self.scaler = GradScaler("cuda", enabled=config.training.amp and device.type == "cuda")
        self.completed_updates = 0
        self.data_epoch = 0
        self.skipped_updates = 0
        self.loader: PolicyLoader | None = None
        self.validation_loader: PolicyLoader | None = None
        self._closed = False
        identity = None
        if _checkpoint is not None:
            self.policy.load_state_dict(_checkpoint["model"], strict=True)
            self.optimizer.load_state_dict(_checkpoint["optimizer"])
            self.scheduler.load_state_dict(_checkpoint["scheduler"])
            if _checkpoint["scaler"] and self.scaler.is_enabled():
                self.scaler.load_state_dict(_checkpoint["scaler"])
            self.completed_updates = int(_checkpoint["completed_updates"])
            self.data_epoch = int(_checkpoint["data_epoch"]) + 1
            self.skipped_updates = int(_checkpoint["skipped_updates"])
            if _checkpoint["wandb"] is not None:
                identity = RunIdentity.model_validate(_checkpoint["wandb"])
        self.logger = MetricsLogger(
            config.logging,
            config.model_dump(mode="json"),
            identity,
            self.completed_updates if _checkpoint is not None else None,
            provenance={
                "dataset_fingerprint": self.dataset.fingerprint,
                "selection": [list(item) for item in self.dataset.selection],
                "feature_schema": config.features.model_dump(mode="json"),
            },
        )
        if _checkpoint is not None:
            restore_rng(_checkpoint["rng"], self.device)

    def _lr_multiplier(self, update: int) -> float:
        """Compute the next update's learning rate from completed update count."""
        config = self.config.training
        if config.warmup_updates and update < config.warmup_updates:
            return (update + 1) / config.warmup_updates
        if config.schedule == "constant":
            return 1.0
        progress = min(
            1.0,
            max(
                0.0,
                (update - config.warmup_updates)
                / max(1, config.max_updates - config.warmup_updates),
            ),
        )
        return (
            config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )

    def update(self, batch: dict[str, Any]) -> tuple[float, bool]:
        """Attempt one update; advance scheduling only when AMP actually steps."""
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        inputs = self.packer(batch, device=self.device, training=True, require_actions=True)
        device_type = torch.device(self.device).type
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=self.config.training.amp):
            loss = self.policy.loss(inputs)
        if not self.scaler.is_enabled() and not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        scaled_loss: Tensor = self.scaler.scale(loss)
        torch.autograd.backward(scaled_loss)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.training.grad_clip_norm,
            error_if_nonfinite=not self.scaler.is_enabled(),
        )
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= previous_scale
        if updated:
            self.completed_updates += 1
            self.scheduler.step()
        else:
            self.skipped_updates += 1
        return float(loss.detach().item()), updated

    def fit(self, *, until_step: int | None = None) -> None:
        """Train through a completed-update limit and release all resources on exit.

        ``until_step`` allows an intentional stop without changing the saved
        training plan, such as checkpointing at 250 of a 500-update pilot.
        """
        target = self.config.training.max_updates if until_step is None else until_step
        if self._closed:
            raise RuntimeError("Trainer is closed; resume a checkpoint to continue")
        if not self.completed_updates < target <= self.config.training.max_updates:
            raise ValueError("Target must be after the current update and within max_updates")
        self.loader = build_loader(self.dataset, self.config.loader)
        if len(self.validation_dataset):
            self.validation_loader = build_loader(self.validation_dataset, self.config.loader)
        consecutive_skips = 0
        try:
            while self.completed_updates < target:
                self.dataset.set_epoch(self.data_epoch)
                batches_seen = 0
                for batch in self.loader:
                    batches_seen += 1
                    loss, updated = self.update(batch)
                    if not updated:
                        consecutive_skips += 1
                        self.logger.log(
                            {
                                "event": "amp_skip",
                                "train/step": self.completed_updates,
                                "train/skipped_updates": self.skipped_updates,
                            }
                        )
                        if consecutive_skips >= self.config.training.max_consecutive_skips:
                            raise FloatingPointError(
                                "AMP repeatedly skipped updates; check inputs and model stability"
                            )
                        continue
                    consecutive_skips = 0
                    step = self.completed_updates
                    if step % self.config.training.log_every == 0:
                        self.logger.log(
                            {
                                "event": "train",
                                "train/step": step,
                                "train/loss": loss,
                                "train/lr": self.optimizer.param_groups[0]["lr"],
                            }
                        )
                    if (
                        self.validation_loader is not None
                        and step % self.config.training.validate_every == 0
                    ):
                        metrics = validate(
                            self.policy,
                            self.validation_loader,
                            self.packer,
                            self.device,
                            self.config.training.validation_batches,
                        )
                        self.logger.log({"event": "validation", "train/step": step, **metrics})
                    if step % self.config.training.checkpoint_every == 0:
                        self.save(Path(self.config.logging.out_dir) / f"step-{step}.pt")
                    if step >= target:
                        break
                if batches_seen == 0:
                    raise RuntimeError("Finite training loader unexpectedly emitted no batches")
                if self.completed_updates < target:
                    self.data_epoch += 1
            self.save(Path(self.config.logging.out_dir) / "latest.pt")
        finally:
            self.close()

    def save(self, path: str | Path) -> None:
        """Atomically persist architecture, state, preprocessing, and run identity."""
        payload: dict[str, Any] = {
            "format": "robopolicy.training",
            "schema_version": SCHEMA_VERSION,
            "policy_config": self.config.policy.model_dump(mode="json"),
            "model": self.policy.state_dict(),
            "statistics": self.packer.normalizer.statistics.model_dump(mode="json"),
            "feature_schema": self.config.features.model_dump(mode="json"),
            "processing": self.config.processing.model_dump(mode="json"),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "completed_updates": self.completed_updates,
            "data_epoch": self.data_epoch,
            "skipped_updates": self.skipped_updates,
            "rng": capture_rng(self.device),
            "dataset_fingerprint": self.dataset.fingerprint,
            "selection": [list(item) for item in self.dataset.selection],
            "run_config": self.config.model_dump(mode="json"),
            "wandb": self.logger.identity.model_dump() if self.logger.identity else None,
            "segment": self.logger.segment,
        }
        save_checkpoint(path, payload)
        self.logger.log_checkpoint(path, self.completed_updates)

    @classmethod
    def resume(
        cls,
        path: str | Path,
        *,
        dataset_root: str | None = None,
        device: str | None = None,
        out_dir: str | None = None,
    ) -> Trainer:
        """Recover using saved settings; allow data relocation and device changes."""
        payload = read_checkpoint(path)
        values = dict(payload["run_config"])
        if dataset_root is not None:
            values["dataset"] = {**values["dataset"], "root": dataset_root}
        if device is not None:
            values["training"] = {**values["training"], "device": device}
        if out_dir is not None:
            values["logging"] = {**values["logging"], "out_dir": out_dir}
        return cls(RunConfig.model_validate(values), _checkpoint=payload)

    @classmethod
    def finetune(cls, config: RunConfig, checkpoint: str | Path) -> Trainer:
        """Start fresh optimizer, RNG, statistics, and logging using saved weights."""
        payload = read_checkpoint(checkpoint)
        if config.policy.model_dump(mode="json") != payload["policy_config"]:
            raise ValueError("Fine-tuning requires a matching architecture")
        trainer = cls(config, _initialize_pretrained=False)
        trainer.policy.load_state_dict(payload["model"], strict=True)
        return trainer

    def close(self) -> None:
        """Release loaders and logging idempotently after success or exceptions."""
        if self._closed:
            return
        for loader in (self.loader, self.validation_loader):
            if loader is not None:
                loader.close()
        self.dataset.close()
        self.validation_dataset.close()
        self.logger.close()
        self._closed = True
