"""A single policy composes observation encoding, a head, and flow matching."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from robopolicy.algorithms import build_algorithm
from robopolicy.algorithms.improved_mean_flow import ImprovedMeanFlowConfig
from robopolicy.contracts.batch import PolicyInput
from robopolicy.modules.observation import ObservationEncoder
from robopolicy.modules.policy_head.config import ConvHeadConfig
from robopolicy.modules.policy_head.transformer import (
    TransformerActionDecoder,
    TransformerActionDecoderConfig,
)
from robopolicy.modules.policy_head.unet import FiLMConvPolicyHead
from robopolicy.policies.config import PolicyConfig


class ActionField(nn.Module):
    """Adapt two distinct head computations to the same flow-field signature."""

    def __init__(self, config: PolicyConfig, tokens: int) -> None:
        """Construct a head from settings owned by that component."""
        super().__init__()
        self.config = config
        head = config.head
        self.network: nn.Module
        if isinstance(head, ConvHeadConfig):
            self.network = FiLMConvPolicyHead(
                action_dim=config.action_dim,
                horizon=config.action_steps,
                global_cond_dim=tokens * config.fusion.dim,
                down_dims=head.down_dims,
                time_embed_dim=head.time_dim,
                kernel_size=head.kernel_size,
                n_groups=head.groups,
            )
        else:
            self.network = TransformerActionDecoder(
                action_dim=config.action_dim,
                horizon=config.action_steps,
                context_dim=config.fusion.dim,
                robot_state_dim=config.fusion.dim,
                config=TransformerActionDecoderConfig(
                    d_model=head.dim,
                    n_layers=head.layers,
                    n_heads=head.heads,
                    ff_dim=head.ff_dim,
                    time_embed_dim=head.time_dim,
                    dropout=head.dropout,
                    attention_dropout=head.attention_dropout,
                    interval_conditioning=isinstance(config.algorithm, ImprovedMeanFlowConfig),
                    state_conditioning=head.state_conditioning,
                ),
            )

    def forward(
        self,
        actions: Tensor,
        timestep: Tensor,
        global_cond: Tensor,
        *,
        time_delta: Tensor | None = None,
    ) -> Tensor:
        """Predict velocities (B,T_action,D_action), optionally conditioned on intervals (B,)."""
        if isinstance(self.config.head, ConvHeadConfig):
            return cast(Tensor, self.network(actions, timestep, global_cond=global_cond))
        state = global_cond[:, : self.config.obs_steps]  # (B, T_obs, D_context)
        context = global_cond[:, self.config.obs_steps :]
        return cast(
            Tensor,
            self.network(
                actions, timestep, context=context, robot_state=state, time_delta=time_delta
            ),
        )


class FlowPolicy(nn.Module):
    """Public loss, prediction, and attention APIs sharing one observation encoder."""

    def __init__(self, config: PolicyConfig, initialize_pretrained: bool = False) -> None:
        """Construct the configured architecture, with no implicit downloads."""
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(
            config.state_dim,
            config.obs_steps,
            config.cameras,
            config.entity_dim,
            config.selected_entities,
            config.conditioning,
            config.fusion,
            config.vision,
            initialize_pretrained,
        )
        self.field = ActionField(config, self.encoder.token_count)
        self.algorithm = build_algorithm(config.algorithm)

    def encode(
        self, inputs: PolicyInput, return_attention: bool = False
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Validate and encode canonical observations into ``(B, L_context, D)``."""
        return cast(
            tuple[Tensor, dict[str, Tensor]],
            self.encoder(inputs, return_attention=return_attention),
        )

    def loss(self, inputs: PolicyInput) -> Tensor:
        """Return scalar flow loss using actions ``(B, T_action, D_action)``."""
        actions = inputs.actions
        if actions is None or actions.shape != (
            inputs.state.shape[0],
            self.config.action_steps,
            self.config.action_dim,
        ):
            raise ValueError("Training requires actions matching (B, T_action, D_action)")
        if not actions.is_floating_point():
            raise ValueError("Actions must be floating point")
        context, _ = self.encode(inputs)
        return self.algorithm.compute_loss(self.field, actions, context)

    @torch.no_grad()
    def predict(self, inputs: PolicyInput, noise: Tensor | None = None) -> Tensor:
        """Generate normalized actions ``(B, T_action, D_action)`` from observations.

        Optional fixed noise makes prediction parity directly testable. Callers
        choose model mode; inference checkpoint loading returns an eval model.
        """
        self.encoder.validate(inputs)
        shape = (inputs.state.shape[0], self.config.action_steps, self.config.action_dim)
        if noise is None:
            noise = torch.randn(shape, device=inputs.state.device, dtype=inputs.state.dtype)
        if (
            noise.shape != shape
            or noise.device != inputs.state.device
            or noise.dtype != inputs.state.dtype
        ):
            raise ValueError("Initial noise must match action shape, input device, and dtype")
        return self.predict_tensors(inputs, noise)

    def predict_tensors(self, inputs: PolicyInput, noise: Tensor) -> Tensor:
        """Run the shared prediction computation after boundary validation.

        Args:
            inputs: Validated canonical observations, already on the model device.
            noise: Initial ``(B, T_action, D_action)`` actions in the solver dtype.

        Returns:
            Normalized actions ``(B, T_action, D_action)``. Keeping the solver
            state in the noise dtype preserves FP32 accumulation under autocast.
        """
        context, _ = self.encoder.encode_tensors(inputs)
        return self.algorithm.generate_samples(self.field, noise, context)

    def attention_maps(self, inputs: PolicyInput) -> dict[str, Tensor]:
        """Return maps from the same encoding path used by loss and prediction."""
        return self.encode(inputs, return_attention=True)[1]


def build_policy(config: PolicyConfig, *, initialize_pretrained: bool = False) -> FlowPolicy:
    """Build any supported policy for training or inference from the same factory."""
    return FlowPolicy(config, initialize_pretrained)
