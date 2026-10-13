"""Release checks against real vision, decoder, and multiprocessing runtimes."""

from __future__ import annotations

import builtins
import shutil
from pathlib import Path

import pytest
import torch

from robopolicy import DatasetConfig, LoaderConfig, RunConfig, Trainer, build_dataset, build_loader
from robopolicy.checkpoints.serialization import read_checkpoint, save_checkpoint
from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.config import VideoConfig
from robopolicy.datasets.video import DecoderHandle, require_decoder
from tests.core.test_dataset import write_dataset
from tests.core.test_training import run_config


@pytest.mark.parametrize("decoder", ["opencv", "torchcodec"])
def test_explicit_decoder_failure_is_actionable(
    decoder: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def unavailable(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in {"cv2", "torchcodec"}:
            raise ImportError("deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match=rf"Install robopolicy\[{decoder}\]"):
        require_decoder(VideoConfig(decoder=decoder))


@pytest.mark.parametrize("decoder", ["opencv", "torchcodec"])
def test_real_decoder_repeated_requests_and_frame_bounds(tmp_path: Path, decoder: str) -> None:
    write_dataset(tmp_path, video=True)
    config = VideoConfig(decoder=decoder)
    path = tmp_path / "task/0-left.avi"
    with pytest.raises(ValueError, match="length mismatch"):
        DecoderHandle(path, config, 12)
    handle = DecoderHandle(path, config, 11)
    try:
        indices = [4, 0, 4, 2, 0]
        frames = handle.read(indices)
        assert frames.shape == (5, 3, 16, 16) and frames.dtype == torch.uint8
        assert frames[:, 0, 0, 0].tolist() == indices
        assert frames[:, 1, 0, 0].tolist() == [0] * 5
        assert frames[:, 2, 0, 0].tolist() == [200] * 5
        assert torch.equal(handle.read(indices), frames)
        for invalid in ([-1], [11], []):
            with pytest.raises(ValueError):
                handle.read(invalid)
    finally:
        handle.close()
    assert handle.native is None


@pytest.mark.parametrize("decoder", ["opencv", "torchcodec"])
def test_video_workers_reset_and_terminate(tmp_path: Path, decoder: str) -> None:
    schema = write_dataset(tmp_path, video=True)
    dataset = build_dataset(
        DatasetConfig(
            root=str(tmp_path),
            video={"decoder": decoder, "max_handles": 2},
            selection={"val_fraction": 0},
            sampling={"k_passes": 4, "block_size": 2},
        ),
        schema,
    )
    with build_loader(dataset, LoaderConfig(batch_size=2, num_workers=2)) as loader:
        first = next(iter(loader))
        assert first["images"].dtype == torch.uint8
        workers = list(loader._iterator._workers)
        dataset.set_epoch(1)
        values = [value for batch in loader for value in batch["state"][:, -1, 0].tolist()]
        assert len(values) == len(set(values)) == len(dataset)
        next(iter(loader))
    assert all(not worker.is_alive() for worker in workers)
    assert 0 < dataset.resources["peak_handles"] <= 4
    assert dataset.resources["live_handles"] == 0


@pytest.mark.parametrize("decoder", ["opencv", "torchcodec"])
def test_decode_error_in_worker_stops_all_workers(tmp_path: Path, decoder: str) -> None:
    schema = write_dataset(tmp_path, video=True)
    dataset = build_dataset(
        DatasetConfig(
            root=str(tmp_path),
            video={"decoder": decoder},
            selection={"val_fraction": 0},
        ),
        schema,
    )
    # Keep valid manifests but corrupt actual encoded sources after index validation.
    for path in (tmp_path / "task").glob("*.avi"):
        path.write_bytes(b"broken encoded video")
    loader = build_loader(dataset, LoaderConfig(batch_size=2, num_workers=2))
    try:
        iterator = iter(loader)
        workers = list(loader._iterator._workers)
        with pytest.raises(RuntimeError, match="[Vv]ideo|[Oo]pen|[Dd]ecod"):
            next(iterator)
    finally:
        loader.close()
    assert all(not worker.is_alive() for worker in workers)
    assert dataset.resources["live_handles"] == 0


@pytest.mark.vision
@pytest.mark.parametrize("representation", ["rgb", "current_and_signed_difference"])
def test_dinov2_checkpoint_inference_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, representation: str
) -> None:
    from transformers import Dinov2Model

    from robopolicy import load_policy

    config = run_config(tmp_path).model_dump(mode="json")
    schema = write_dataset(tmp_path / "data", video=True)
    config["features"] = FeatureSchema(
        features=tuple(feature for feature in schema.features if feature.key != "task_indices")
    ).model_dump(mode="json")
    config["policy"].update(
        cameras=2,
        vision={
            "type": "dinov2",
            "dim": 24,
            "depth": 1,
            "heads": 3,
            "patch_size": 8,
            "image_size": 16,
            "freeze": True,
        },
    )
    config["processing"]["image_representation"] = representation
    if representation == "current_and_signed_difference":
        config["policy"]["obs_steps"] = 2
        for feature in config["features"]["features"]:
            if feature["key"] in {"state", "images"}:
                feature["history"] = 1
    trainer = Trainer(RunConfig.model_validate(config))
    with build_loader(trainer.dataset) as loader:
        raw = next(iter(loader))
    _, updated = trainer.update(raw)
    assert updated
    vision = trainer.policy.encoder.vision
    assert not vision.backbone.training
    assert all(not parameter.requires_grad for parameter in vision.backbone.parameters())
    checkpoint = tmp_path / "vision.pt"
    trainer.save(checkpoint)
    inputs = trainer.packer(raw)
    noise = torch.zeros(inputs.state.shape[0], 3, 1)
    trainer.policy.eval()
    expected = trainer.packer.actions_to_raw(trainer.policy.predict(inputs, noise))
    payload = read_checkpoint(checkpoint)
    assert payload["processing"]["image_representation"] == representation
    if representation == "rgb":
        # Checkpoints from before this opt-in field must retain identical RGB predictions.
        del payload["processing"]["image_representation"]
        del payload["run_config"]["processing"]["image_representation"]
    assert any("backbone" in key for key in payload["model"])
    payload["policy_config"]["vision"]["pretrained"] = "unavailable/pretrained-backbone"
    payload["run_config"]["policy"]["vision"]["pretrained"] = "unavailable/pretrained-backbone"
    save_checkpoint(checkpoint, payload)
    trainer.close()
    shutil.rmtree(tmp_path / "data")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Inference attempted a pretrained download")

    monkeypatch.setattr(Dinov2Model, "from_pretrained", forbidden)
    inference = load_policy(checkpoint, device="cpu")
    observations = {key: value for key, value in raw.items() if key != "actions"}
    torch.testing.assert_close(inference.predict(observations, noise), expected, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["policy_config", "feature_schema"])
def test_resume_rejects_inconsistent_architecture_schema(tmp_path: Path, field: str) -> None:
    trainer = Trainer(run_config(tmp_path))
    trainer.fit(until_step=3)
    checkpoint = tmp_path / "run/latest.pt"
    payload = read_checkpoint(checkpoint)
    if field == "policy_config":
        payload[field]["state_dim"] += 1
    else:
        payload[field]["features"][0]["shape"] = [2]
    save_checkpoint(checkpoint, payload)
    with pytest.raises(ValueError, match=f"{field} conflicts with run_config"):
        Trainer.resume(checkpoint)


@pytest.mark.cuda
def test_cuda_rng_scaler_recovery_and_cpu_relocation(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "Run CUDA acceptance tests on a CUDA runner"
    trainer = Trainer(run_config(tmp_path, device="cuda:0", amp=True))
    trainer.fit(until_step=3)
    checkpoint = tmp_path / "run/latest.pt"
    saved = read_checkpoint(checkpoint)
    torch.cuda.set_rng_state(saved["rng"]["cuda"], "cuda:0")
    expected = torch.rand(4, device="cuda:0")
    resumed = Trainer.resume(checkpoint)
    assert resumed.scaler.state_dict() == saved["scaler"]
    torch.testing.assert_close(torch.rand(4, device="cuda:0"), expected, rtol=0, atol=0)
    resumed.close()
    cpu = Trainer.resume(checkpoint, device="cpu", out_dir=str(tmp_path / "cpu-run"))
    assert cpu.completed_updates == 3 and next(cpu.policy.parameters()).device.type == "cpu"
    cpu.fit()
    assert cpu.completed_updates == 6
