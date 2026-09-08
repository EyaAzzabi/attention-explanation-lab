"""Leave-one-out on documents that cannot survive having a token removed.

This is the second bug the attention tripwire caught, and it is a different one from
the empty documents. Training ran fine for six epochs; the failure only appeared when
the measurement started. Removing the single token of a one-token document leaves an
all-zero mask, which is the state that produces NaN attention.

A one-token document has no defined leave-one-out importance and no defined Kendall
tau, so the correct behaviour is to leave it out of the measurement, not to crash and
not to silently record a zero as though it had been measured.
"""
import numpy as np
import torch

from src.importance import kendall_tau_per_instance, leave_one_out_importance
from src.model import AttentionClassifier, Batch

VOCAB, CLASSES = 20, 2


def batch_with_a_single_token_document() -> Batch:
    tokens = torch.tensor([[3, 4, 5, 6],
                           [7, 0, 0, 0],      # one token only
                           [8, 9, 0, 0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0]])
    return Batch(tokens=tokens, mask=mask, labels=torch.tensor([0, 1, 0]))


def test_leave_one_out_survives_a_single_token_document():
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, "bilstm")
    loo = leave_one_out_importance(model, batch_with_a_single_token_document())

    assert loo.shape == (3, 4)
    assert np.isfinite(loo).all()


def test_the_single_token_document_is_left_unmeasured_rather_than_scored_zero():
    """Zero would be a measurement. NaN downstream is the honest outcome."""
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, "bilstm")
    b = batch_with_a_single_token_document()
    loo = leave_one_out_importance(model, b)

    assert (loo[1] == 0).all()                       # nothing was measured for it
    taus, _ = kendall_tau_per_instance(loo, loo, b.mask.numpy())
    assert np.isnan(taus[1])                         # and it enters no average


def test_other_documents_in_the_batch_are_still_measured():
    """The guard must skip one row, not disable the whole batch."""
    torch.manual_seed(0)
    model = AttentionClassifier(VOCAB, CLASSES, "bilstm")
    b = batch_with_a_single_token_document()
    loo = leave_one_out_importance(model, b)

    assert loo[0, :4].sum() > 0
    assert loo[2, :2].sum() > 0
    assert (loo[2, 2:] == 0).all()                   # padding stays untouched


def test_removing_a_token_never_empties_a_mask():
    """The property the tripwire enforces, checked directly.

    If any eligible row could be reduced to zero unmasked positions, the attention
    module would raise and the run would die in the middle of a measurement, which
    is what happened on 20 Newsgroups.
    """
    b = batch_with_a_single_token_document()
    lengths = b.mask.sum(dim=1)
    for pos in range(b.mask.shape[1]):
        active = (lengths > pos) & (lengths > 1)
        keep = b.mask.clone()
        keep[active, pos] = 0.0
        assert bool((keep.sum(dim=1) > 0).all()), f"position {pos} emptied a row"
