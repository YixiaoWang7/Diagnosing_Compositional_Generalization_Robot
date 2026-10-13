"""Prepare immutable CG datasets for the public RoboPolicy data contract."""

from corl2026_cg.data.inspect import inspect_instructions
from corl2026_cg.data.prepare import prepare_dataset

__all__ = ["prepare_dataset", "inspect_instructions"]
