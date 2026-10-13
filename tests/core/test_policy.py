"""Generic conditioning and shared observation/action computations."""

from dataclasses import replace

import pytest
import torch

from robopolicy.contracts.batch import PolicyInput
from robopolicy.modules.queries import IndexedConditioning, QueryTokens
from robopolicy.policies.config import PolicyConfig
from robopolicy.policies.flow import build_policy


def policy_config(head="transformer", images=False, **kwargs):
    """Create a tiny architecture using production components."""
    return PolicyConfig(
        state_dim=2,
        action_dim=1,
        obs_steps=2,
        action_steps=4,
        cameras=2 if images else 0,
        vision={"type": "patch", "patch_size": 4, "dim": 16} if images else None,
        fusion={"dim": 16, "heads": 2, "layers": 2, "ff_dim": 32},
        head={"type": "conv", "down_dims": (16, 32), "time_dim": 8}
        if head == "conv"
        else {
            "type": "transformer",
            "dim": 16,
            "heads": 2,
            "layers": 1,
            "ff_dim": 32,
            "time_dim": 8,
        },
        **kwargs,
    )


@pytest.mark.parametrize("sizes", [(2,), (4, 4, 3), (1, 3, 2, 5, 7)])
def test_independent_queries(sizes):
    queries = QueryTokens(IndexedConditioning(vocab_sizes=sizes), 8)
    ids = torch.zeros((3, len(sizes)), dtype=torch.int64)
    original = queries(3, ids).detach().clone()
    assert original.shape == (3, len(sizes), 8)
    with torch.no_grad():
        queries.embeddings[0].weight.add_(1)
    changed = queries(3, ids)
    torch.testing.assert_close(changed[:, 0], original[:, 0] + 1)
    torch.testing.assert_close(changed[:, 1:], original[:, 1:])
    assert queries.expand_observations(changed, 2, 3).shape == (18, len(sizes), 8)
    ids[0, -1] = sizes[-1]
    with pytest.raises(ValueError, match="slot"):
        queries(3, ids)
    with pytest.raises(ValueError, match="integer"):
        queries(3, ids.float())
    with pytest.raises(ValueError, match="requires"):
        queries(3, ids[:, :-1])


@pytest.mark.parametrize("head", ["conv", "transformer"])
@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
def test_shared_policy_gradients_and_prediction(head, images, indexed):
    torch.manual_seed(10)
    config = policy_config(
        head,
        images,
        conditioning={"type": "indexed", "vocab_sizes": [4, 4, 3]}
        if indexed
        else {"type": "learned", "num_tokens": 3},
    )
    policy = build_policy(config)
    inputs = PolicyInput(
        state=torch.randn(3, 2, 2),
        images=torch.randn(3, 2, 2, 3, 16, 16) if images else None,
        task_indices=torch.zeros((3, 3), dtype=torch.int64) if indexed else None,
        actions=torch.randn(3, 4, 1),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss = policy.loss(inputs)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoder.queries.parameters()
    )
    if images:
        assert policy.encoder.vision.backbone.weight.grad.abs().sum() > 0
    policy.eval()
    observation = replace(inputs, actions=None)
    prediction = policy.predict(observation, torch.zeros(3, 4, 1))
    assert prediction.shape == (3, 4, 1) and torch.isfinite(prediction).all()
    with pytest.raises(ValueError, match="Training requires"):
        policy.loss(observation)


@pytest.mark.parametrize("algorithm", ["flow_matching", "improved_mean_flow"])
def test_transformer_excludes_state_from_cross_attention(algorithm):
    policy = build_policy(policy_config(images=True, algorithm={"type": algorithm}))
    captured = {}

    def capture_inputs(_module, _args, kwargs):
        captured.update(kwargs)

    handle = policy.field.network.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    context = torch.randn(3, policy.encoder.token_count, policy.config.fusion.dim)
    options = {"time_delta": torch.rand(3)} if algorithm == "improved_mean_flow" else {}
    policy.field(torch.randn(3, 4, 1), torch.rand(3), context, **options)
    handle.remove()

    torch.testing.assert_close(captured["robot_state"], context[:, : policy.config.obs_steps])
    torch.testing.assert_close(captured["context"], context[:, policy.config.obs_steps :])


def test_transformer_can_condition_on_current_state_only():
    config = policy_config(images=True)
    config = PolicyConfig.model_validate(
        {
            **config.model_dump(),
            "head": {**config.head.model_dump(), "state_conditioning": "current"},
        }
    )
    policy = build_policy(config)
    captured = []
    handle = policy.field.network.state_proj.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    context = torch.randn(3, policy.encoder.token_count, policy.config.fusion.dim)
    policy.field(torch.randn(3, 4, 1), torch.rand(3), context)
    handle.remove()
    torch.testing.assert_close(captured[0], context[:, policy.config.obs_steps - 1])


@pytest.mark.parametrize(
    ("fusion", "mask"),
    [
        ("cross", "none"),
        ("cross", "query"),
        ("self", "none"),
        ("self", "query"),
        ("self", "query_strict"),
    ],
)
def test_attention_encoding_parity(fusion, mask):
    config = policy_config(images=True)
    config = PolicyConfig.model_validate(
        {
            **config.model_dump(),
            "fusion": {**config.fusion.model_dump(), "type": fusion, "mask": mask},
        }
    )
    policy = build_policy(config).eval()
    inputs = PolicyInput(state=torch.randn(2, 2, 2), images=torch.randn(2, 2, 2, 3, 16, 16))
    encoded, _ = policy.encode(inputs)
    with_maps, maps = policy.encode(inputs, return_attention=True)
    torch.testing.assert_close(encoded, with_maps, atol=1e-5, rtol=1e-5)
    assert maps["query_attn"].shape == (2, 2, 2, 1, 16)
    assert torch.isfinite(maps["query_attn"]).all()
    if mask == "query_strict":
        blocked = policy.encoder.fusion.get_mask(torch.device("cpu"), num_queries=3, num_tokens=4)
        assert blocked[3:, :3].all() and not blocked[:3, :3].diagonal().any()


def test_learned_temporal_encoding_marks_observation_order():
    config = policy_config(images=True)
    config = PolicyConfig.model_validate(
        {
            **config.model_dump(),
            "fusion": {**config.fusion.model_dump(), "temporal_encoding": "learned"},
        }
    )
    policy = build_policy(config)
    embedding = policy.encoder.temporal_embedding
    assert embedding is not None and embedding.shape == (1, 2, 1, 16)
    inputs = PolicyInput(state=torch.zeros(1, 2, 2), images=torch.zeros(1, 2, 2, 3, 16, 16))
    context, _ = policy.encode(inputs)
    unmarked_state = context[0, :2] - embedding[0, :, 0]
    torch.testing.assert_close(unmarked_state[0], unmarked_state[1])
    assert not torch.equal(embedding[:, 0], embedding[:, 1])


def test_entity_selection():
    policy = build_policy(policy_config(entity_dim=5, selected_entities=2))
    entities = torch.randn(3, 2, 7, 5, requires_grad=True)
    inputs = PolicyInput(
        state=torch.randn(3, 2, 2),
        entities=entities,
        entity_indices=torch.tensor([[4, 0], [4, 0], [4, 0]]),
        actions=torch.randn(3, 4, 1),
    )
    context, _ = policy.encode(inputs)
    context.sum().backward()
    assert entities.grad[:, :, 0].abs().sum() > 0 and entities.grad[:, :, 4].abs().sum() > 0
    assert entities.grad[:, :, 1:4].abs().sum() == 0
    with pytest.raises(ValueError, match="range"):
        policy.encode(replace(inputs, entity_indices=torch.full((3, 2), 7)))
