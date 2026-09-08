"""Smoke tests for the four encoders and, above all, for the gradient path.

The single most fragile thing in this reproduction is Appendix B: the computation
graph must be cut at the attention module so the gradient does not flow through it.
Get that wrong and every correlation comes out too high, in exactly the direction
that would make the paper look wrong. So it gets a test that fails loudly.
"""
import numpy as np
import pytest
import torch

from src.importance import gradient_importance, leave_one_out_importance
from src.model import CONTEXTUALISATION_ORDER, AttentionClassifier, Batch

VOCAB, CLASSES, B, T = 50, 2, 4, 7


def make_batch(seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, VOCAB, (B, T), generator=g)
    mask = torch.ones(B, T)
    mask[0, 5:] = 0.0          # one short instance, to exercise the padding paths
    mask[1, 6:] = 0.0
    tokens = (tokens * mask.long())
    labels = torch.randint(0, CLASSES, (B,), generator=g)
    return Batch(tokens=tokens, mask=mask, labels=labels)


@pytest.mark.parametrize("encoder", CONTEXTUALISATION_ORDER)
def test_every_encoder_produces_a_valid_attention_distribution(encoder):
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, encoder)
    batch = make_batch()
    alpha = model.attention_weights(batch.tokens, batch.mask)

    assert alpha.shape == (B, T)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-5)
    # No attention mass may land on padding, or the correlation is computed against
    # positions that carry no token.
    padded = alpha[batch.mask.numpy() == 0]
    np.testing.assert_allclose(padded, 0.0, atol=1e-6)


@pytest.mark.parametrize("encoder", CONTEXTUALISATION_ORDER)
def test_gradient_importance_is_finite_and_masked(encoder):
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, encoder)
    batch = make_batch()
    g = gradient_importance(model, batch, class_index=1)

    assert g.shape == (B, T)
    assert np.isfinite(g).all()
    assert (g >= 0).all()                                   # it is an absolute value
    assert (g[batch.mask.numpy() == 0] == 0).all()          # padding contributes none
    assert g.sum() > 0                                      # and it is not all zeros


def test_detaching_attention_actually_changes_the_gradient():
    """Appendix B, made falsifiable.

    If cutting the graph at the attention module made no difference, the flag would
    be decorative and the reproduction would silently be measuring something else.
    """
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, "bilstm")
    batch = make_batch()

    def grad(detach: bool) -> np.ndarray:
        emb = model.embed(batch.tokens)
        emb.retain_grad()
        out, _ = model.forward_from_embeddings(emb, batch.mask, detach_attention=detach)
        model.zero_grad(set_to_none=True)
        out[:, 1].sum().backward()
        return (emb.grad * emb).sum(-1).abs().detach().numpy()

    cut = grad(True)
    through = grad(False)
    assert not np.allclose(cut, through), (
        "detach_attention=True produced the same gradient as False: the graph is "
        "not actually being cut at the attention module"
    )


def test_leave_one_out_is_zero_beyond_the_real_length():
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, "average")
    batch = make_batch()
    loo = leave_one_out_importance(model, batch)

    assert loo.shape == (B, T)
    assert np.isfinite(loo).all()
    assert (loo >= 0).all()
    assert (loo[0, 5:] == 0).all()      # instance 0 has length 5


def test_substituting_attention_leaves_the_encoder_untouched():
    """Algorithms 2 and 3 both note that h is not recomputed.

    Uniform attention over a masked instance must give the same answer whether the
    weights are handed in or computed, as long as h is the same. This pins the
    contract that decode_from_attention does not re-run anything but the decoder.
    """
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, "bilstm")
    batch = make_batch()

    alpha = torch.as_tensor(model.attention_weights(batch.tokens, batch.mask))
    substituted = model.decode_from_attention(batch.tokens, batch.mask, alpha)
    direct = model.predict_proba(batch.tokens, batch.mask)
    np.testing.assert_allclose(substituted, direct, atol=1e-5)
