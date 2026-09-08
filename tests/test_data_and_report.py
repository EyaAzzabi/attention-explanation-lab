"""Tests for preprocessing and for the honesty guards in the report.

The empty-document tests exist because of a real failure, not a hypothetical one.
20 Newsgroups with headers, footers and quotes removed contains 37 empty training
documents. Each one produces an all-zero mask, the attention softmax runs over a row
of -inf, and the result is NaN, which propagates into the gradient and takes the whole
run with it. The symptom was a loss of `nan`, accuracy at chance, and every Kendall
tau coming back `nan`. Nothing crashed.
"""
import numpy as np
import pytest
import torch

from src.data import NUM, Corpus, Vocabulary, drop_empty, tokenize
from src.model import AttentionClassifier
from src.report import aggregate, fmt, overlaps


# --------------------------------------------------------------------------- #
# Tokenisation, per Appendix A
# --------------------------------------------------------------------------- #
def test_tokens_containing_digits_become_qqq():
    assert tokenize("I waited 15 minutes") == ["i", "waited", NUM, "minutes"]


def test_digit_rule_applies_to_mixed_tokens_not_only_bare_numbers():
    """`covid19` contains a digit and maps to qqq, exactly like a bare number.

    spaCy splits the trailing punctuation off `2019!`, so the expected output keeps
    the `!` as its own token. That is the tokeniser's behaviour, not the digit rule's.
    """
    assert tokenize("covid19 in 2019!") == [NUM, "in", NUM, "!"]


def test_tokenisation_is_lowercased():
    assert tokenize("The Movie") == ["the", "movie"]


def test_out_of_vocabulary_maps_to_unk_index_one():
    vocab = Vocabulary.build([["a", "b"]])
    ids, mask = vocab.encode(["a", "zzz"], max_len=3)
    assert ids[1] == 1                        # <unk>
    assert ids[2] == 0 and mask[2] == 0.0     # <pad>


# --------------------------------------------------------------------------- #
# The empty-document bug
# --------------------------------------------------------------------------- #
def _corpus_with_empties() -> Corpus:
    return Corpus(
        name="toy",
        train_texts=["good movie", "", "   ", "bad movie"],
        train_labels=np.array([1, 0, 1, 0]),
        test_texts=["fine", ""],
        test_labels=np.array([1, 0]),
        n_classes=2,
    )


def test_drop_empty_removes_documents_that_tokenise_to_nothing():
    corpus = drop_empty(_corpus_with_empties())
    assert corpus.train_texts == ["good movie", "bad movie"]
    assert corpus.test_texts == ["fine"]


def test_drop_empty_keeps_labels_aligned_with_texts():
    """Dropping by index must not shift the labels, or the task silently changes."""
    corpus = drop_empty(_corpus_with_empties())
    assert list(corpus.train_labels) == [1, 0]      # the labels of the two kept texts
    assert len(corpus.train_labels) == len(corpus.train_texts)


def test_drop_empty_records_what_it_removed():
    """A deviation from the corpus as distributed has to be visible in the output."""
    corpus = drop_empty(_corpus_with_empties())
    assert "dropped 2 empty train and 1 empty test documents" in corpus.notes
    assert corpus.notes in str(corpus.describe())


def test_drop_empty_says_nothing_when_there_was_nothing_to_drop():
    corpus = Corpus("toy", ["a b"], np.array([1]), ["c"], np.array([0]), 2)
    assert drop_empty(corpus).notes == ""


def test_a_fully_masked_row_raises_instead_of_returning_nan():
    """The tripwire. A silent NaN here cost a full training run once already."""
    model = AttentionClassifier(10, 2, "bilstm")
    tokens = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(2, 4)
    mask[1] = 0.0                                  # instance 1 is entirely padding
    with pytest.raises(ValueError, match="no unmasked positions"):
        model(tokens, mask)


# --------------------------------------------------------------------------- #
# Report honesty guards
# --------------------------------------------------------------------------- #
def _run(mean, std=0.0, seed=0, acc=0.8, sig=0.5):
    return {
        "dataset": "toy", "encoder": "bilstm", "seed": seed,
        "tau_gradient": {"mean": mean, "std": std, "sig_frac": sig, "n": 10},
        "performance": {"accuracy": acc, "f1": acc, "n": 10},
    }


def test_aggregate_reports_spread_across_seeds_not_across_instances():
    """The two spreads are different quantities and conflating them would mislead.

    Here every seed has a large within-instance std but the seed means are identical,
    so the across-seed std must be zero.
    """
    runs = [_run(0.5, std=0.3, seed=s) for s in range(3)]
    a = aggregate(runs)
    assert a["mean"] == pytest.approx(0.5)
    assert a["std"] == pytest.approx(0.0)
    assert a["within_instance_std"] == pytest.approx(0.3)


def test_a_single_seed_has_no_spread_and_is_labelled_as_such():
    a = aggregate([_run(0.42)])
    assert a["n_seeds"] == 1
    assert np.isnan(a["std"])
    assert "1 seed" in fmt(a["mean"], a["std"])


def test_a_single_seed_can_never_establish_an_ordering():
    one = aggregate([_run(0.10)])
    many = aggregate([_run(0.90, seed=s) for s in range(3)])
    assert overlaps(one, many), "a lone run must not be allowed to rank against others"


def test_clearly_separated_configurations_are_ranked():
    low = aggregate([_run(0.10, seed=s) for s in range(3)])
    high = aggregate([_run(0.90, seed=s) for s in range(3)])
    assert not overlaps(low, high)


def test_configurations_within_one_standard_deviation_are_not_ranked():
    a = aggregate([_run(m, seed=i) for i, m in enumerate([0.40, 0.50, 0.60])])
    b = aggregate([_run(m, seed=i) for i, m in enumerate([0.45, 0.55, 0.65])])
    assert overlaps(a, b)
