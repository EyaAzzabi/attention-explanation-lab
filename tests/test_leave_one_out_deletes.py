"""Leave-one-out must delete the token, not merely stop attending to it.

This test exists because the wrong version shipped and produced a number that looked
like a result. Masking position t leaves the token in the sequence: a BiLSTM still
runs over it and it still shapes the hidden states of its neighbours. What that
measures is "what if attention stopped looking here", which is close to monotone in
alpha_t by construction, and it gave tau_loo around 0.83 for every contextualising
encoder against the 0.06 to 0.20 the paper reports for the BiLSTM on 20 Newsgroups.

The two operations coincide exactly on the average encoder, where a position
contributes nothing but its own weighted embedding. They must differ on any encoder
that mixes positions. Both halves are asserted, because a fix that made them differ
everywhere would be just as wrong.
"""
import numpy as np
import torch

from src.importance import leave_one_out_importance
from src.model import AttentionClassifier, Batch


def batch4() -> Batch:
    tokens = torch.tensor([[3, 4, 5, 6],
                           [7, 8, 9, 0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0, 0.0]])
    return Batch(tokens=tokens, mask=mask, labels=torch.tensor([0, 1]))


@torch.no_grad()
def masking_version(model, batch) -> np.ndarray:
    """The implementation that was wrong, kept here as the thing to differ from."""
    from src.importance import tvd

    base = model.predict_proba(batch.tokens, batch.mask)
    lengths = batch.mask.sum(dim=1).long()
    b, t = batch.tokens.shape
    out = np.zeros((b, t))
    for pos in range(t):
        active = (lengths > pos) & (lengths > 1)
        if not bool(active.any()):
            continue
        keep = batch.mask.clone()
        keep[active, pos] = 0.0
        probs = model.predict_proba(batch.tokens, keep)
        for i in range(b):
            if bool(active[i]):
                out[i, pos] = tvd(probs[i], base[i])
    return out


def test_deletion_differs_from_masking_on_a_contextualising_encoder():
    """On a BiLSTM the token still shapes its neighbours unless it is removed."""
    torch.manual_seed(0)
    model = AttentionClassifier(20, 2, "bilstm")
    batch = batch4()
    assert not np.allclose(
        leave_one_out_importance(model, batch), masking_version(model, batch), atol=1e-6
    ), "deletion and masking gave identical results: the token is not being removed"


def test_deletion_matches_masking_on_the_average_encoder():
    """The sanity half of the pair.

    With no contextualisation, position t contributes only its own embedding, so
    dropping it from the weighted sum and deleting it from the input are the same
    operation. If these ever diverge, the deletion code is corrupting the sequence.
    """
    torch.manual_seed(0)
    model = AttentionClassifier(20, 2, "average")
    batch = batch4()
    np.testing.assert_allclose(
        leave_one_out_importance(model, batch), masking_version(model, batch), atol=1e-5
    )


def test_deletion_shifts_the_tail_and_shortens_the_mask():
    """Directly check the surgery, independently of any model."""
    b = batch4()
    t = b.tokens.shape[1]
    pos = 1
    lengths = b.mask.sum(dim=1).long()
    active = (lengths > pos) & (lengths > 1)

    tokens, mask = b.tokens.clone(), b.mask.clone()
    tokens[active, pos:t - 1] = b.tokens[active, pos + 1:]
    mask[active, pos:t - 1] = b.mask[active, pos + 1:]
    tokens[active, t - 1] = 0
    mask[active, t - 1] = 0.0

    assert tokens[0].tolist() == [3, 5, 6, 0]          # the 4 at position 1 is gone
    assert mask[0].tolist() == [1.0, 1.0, 1.0, 0.0]    # and the length dropped by one
    assert int(mask.sum(dim=1)[1]) == int(lengths[1]) - 1


def test_padding_positions_are_never_measured():
    torch.manual_seed(0)
    model = AttentionClassifier(20, 2, "bilstm")
    loo = leave_one_out_importance(model, batch4())
    assert loo[1, 3] == 0.0                            # instance 1 has length 3
