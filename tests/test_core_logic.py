import torch

from loss_functions import LossFactory
from model_components import ModelComponents


def test_supervised_contrastive_loss_averages_only_valid_anchors():
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1])
    criterion = LossFactory.SupervisedContrastiveLoss(temperature=1.0)

    loss = criterion(features, labels)

    normalized = torch.nn.functional.normalize(features, dim=1)
    logits = normalized @ normalized.T
    logits.fill_diagonal_(float("-inf"))
    log_probs = torch.log_softmax(logits, dim=1)
    expected = -(log_probs[0, 1] + log_probs[1, 0]) / 2

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert features.grad is not None


def test_supervised_contrastive_loss_without_positive_pairs_is_connected_zero():
    features = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    criterion = LossFactory.SupervisedContrastiveLoss()

    loss = criterion(features, labels)

    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


def test_attention_pooling_matches_additive_attention_and_masks_padding():
    pooler = ModelComponents.AttentionPooling(hidden_dim=2)
    with torch.no_grad():
        pooler.projection.weight.copy_(torch.eye(2))
        pooler.projection.bias.zero_()
        pooler.scorer.weight.copy_(torch.tensor([[1.0, 0.0]]))

    frames = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [100.0, 100.0]]])
    padding_mask = torch.tensor([[False, False, True]])

    pooled = pooler(frames, padding_mask)

    scores = torch.tanh(frames[0, :2])[:, 0]
    weights = torch.softmax(scores, dim=0)
    expected = (weights.unsqueeze(1) * frames[0, :2]).sum(dim=0, keepdim=True)
    torch.testing.assert_close(pooled, expected)


def test_bilstm_valid_output_is_invariant_to_right_padding():
    torch.manual_seed(7)
    fusion = ModelComponents.GatedFeatureFusion(
        feature_dims={"a": 4, "b": 4},
        num_groups=2,
        dropout_rate=0.0,
        temperature=0.1,
    ).eval()

    valid_a = torch.randn(1, 3, 4)
    valid_b = torch.randn(1, 3, 4)
    single_mask = torch.zeros(1, 3, dtype=torch.bool)

    padded_a = torch.cat([valid_a, torch.zeros(1, 2, 4)], dim=1)
    padded_b = torch.cat([valid_b, torch.zeros(1, 2, 4)], dim=1)
    padded_mask = torch.tensor([[False, False, False, True, True]])

    with torch.no_grad():
        single_output, _, _ = fusion(
            {"a": valid_a, "b": valid_b},
            single_mask,
        )
        padded_output, gate_weights, _ = fusion(
            {"a": padded_a, "b": padded_b},
            padded_mask,
        )

    torch.testing.assert_close(single_output, padded_output[:, :3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(padded_output[:, 3:], torch.zeros_like(padded_output[:, 3:]))
    assert gate_weights["a"].shape == (1, 5, 2)
    assert gate_weights["b"].shape == (1, 5, 2)

    source_sum = gate_weights["a"] + gate_weights["b"]
    torch.testing.assert_close(source_sum, torch.ones_like(source_sum))
