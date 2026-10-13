"""Algorithm selection shared by training and checkpoint inference."""

from robopolicy.algorithms.flow_matching import FlowMatchingAlgorithm, FlowMatchingConfig
from robopolicy.algorithms.improved_mean_flow import (
    ImprovedMeanFlowAlgorithm,
    ImprovedMeanFlowConfig,
)

AlgorithmConfig = FlowMatchingConfig | ImprovedMeanFlowConfig


def build_algorithm(config: AlgorithmConfig) -> FlowMatchingAlgorithm | ImprovedMeanFlowAlgorithm:
    """Construct the configured action-generation algorithm without neural parameters."""
    if isinstance(config, ImprovedMeanFlowConfig):
        return ImprovedMeanFlowAlgorithm(config)
    return FlowMatchingAlgorithm(config)
