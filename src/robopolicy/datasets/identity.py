"""Content verification without a loader, decoder, or execution resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from robopolicy.contracts.schema import FeatureSchema
from robopolicy.datasets.config import DatasetConfig
from robopolicy.datasets.index import fingerprint, read_manifest, validate_entries
from robopolicy.datasets.selection import select_entries


@dataclass(frozen=True)
class DatasetIdentity:
    """Location-independent identity of selected dataset sources.

    Attributes:
        fingerprint: SHA-256 of source bytes, relative names, and episode identities.
        selection: Ordered (task, shard, episode) identifiers.
    """

    fingerprint: str
    selection: tuple[tuple[str, int, str], ...]


def dataset_identity(
    config: DatasetConfig, schema: FeatureSchema, split: Literal["train", "val", "all"] = "train"
) -> DatasetIdentity:
    """Verify selected metadata and hash source files once with bounded memory.

    Args:
        config: Dataset root and selection settings; execution options are unused.
        schema: Required numeric shapes and video sidecars.
        split: Selection role, with the same ordering as ``build_dataset``.

    Returns:
        Fingerprint and resolved selection, unchanged by relocation. HDF5 is
        required; video files are hashed as bytes and never decoded.
    """
    root = Path(config.root).expanduser().resolve()
    entries = select_entries(read_manifest(root), config.selection, split)
    episodes = validate_entries(entries, schema)
    return DatasetIdentity(
        fingerprint(root, entries, episodes), tuple(ep.identity for ep in episodes)
    )
