"""Feature importance and distribution-distance measures from Jain & Wallace (2019).

Everything the paper's two experiments rest on lives here, and nothing in this module
touches a model definition, so it can be unit-tested against values computed by hand.

Definitions taken from the paper:

  TVD(y1, y2) = 1/2 * sum_i |y1_i - y2_i|                            (Section 4)
  JSD(a1, a2) = 1/2 KL(a1 || m) + 1/2 KL(a2 || m),  m = (a1+a2)/2    (Section 4)
  tau_g       = Kendall-tau(alpha, g),   g_t = |sum_w 1[x_tw=1] dy/dx_tw|
  tau_loo     = Kendall-tau(alpha, dy),  dy_t = TVD(y(x_-t), y(x))

Two details that are easy to get wrong and that the paper is explicit about:

  * Appendix B: the computation graph is cut at the attention module, so the gradient
    does not flow through attention and contribute to the importance score. Attention
    is treated as a separate input. Getting this wrong inflates the correlation, which
    is precisely the number under test.
  * Section 4.1, Algorithm 1: the gradient is taken with respect to the one-hot input,
    not the embedding. Because x_e = x E, that reduces to the dot product of the
    embedding gradient with the embedding itself, which is what `gradient_importance`
    computes.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Distances between distributions
# --------------------------------------------------------------------------- #
def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two output distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    return float(0.5 * np.abs(p - q).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / (q[mask] + EPS))).sum())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats, between two attention distributions.

    The paper notes JSD between two categorical distributions is bounded above by
    0.69, which is ln(2). That bound is what makes the adversarial histograms in
    Figure 4 readable, so this returns nats rather than bits deliberately.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


JSD_UPPER_BOUND = float(np.log(2))  # 0.693..., the paper's 0.69


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def gradient_importance(
    model,
    batch,
    class_index: int | None = None,
) -> np.ndarray:
    """g_t for every token, with the graph cut at the attention module.

    `model.forward` must accept `detach_attention=True`, which stops the gradient
    flowing back through the softmax over attention scores. See Appendix B of the
    paper for why this is the right choice rather than a convenience.

    Returns an array of shape (batch, seq_len). Padding positions are zero.
    """
    model.eval()
    embeddings = model.embed(batch.tokens)          # (B, T, d), a leaf we can hook
    embeddings.retain_grad()

    output, _ = model.forward_from_embeddings(
        embeddings, batch.mask, detach_attention=True
    )
    target = output if class_index is None else output[:, class_index]
    model.zero_grad(set_to_none=True)
    target.sum().backward()

    if embeddings.grad is None:
        raise RuntimeError(
            "no gradient reached the embeddings: check that forward_from_embeddings "
            "consumes the tensor it was given rather than re-embedding the tokens"
        )

    # g_t = |<dy/demb_t, emb_t>|, which is the one-hot gradient of Algorithm 1.
    g = (embeddings.grad * embeddings).sum(dim=-1).abs()
    g = g * batch.mask
    return g.detach().cpu().numpy()


@torch.no_grad()
def leave_one_out_importance(model, batch) -> np.ndarray:
    """dy_t = TVD(y(x_-t), y(x)) for every token position.

    One forward pass per token, so this is the expensive measure. The paper applies
    it at the input layer, dropping the token rather than zeroing a hidden dimension.
    """
    model.eval()
    base = model.predict_proba(batch.tokens, batch.mask)          # (B, C)
    lengths = batch.mask.sum(dim=1).long()
    b, t = batch.tokens.shape
    out = np.zeros((b, t), dtype=float)

    for pos in range(t):
        keep = batch.mask.clone()
        keep[:, pos] = 0.0
        # Instances shorter than pos have nothing to remove; leave them at zero.
        active = (lengths > pos).cpu().numpy()
        if not active.any():
            break
        probs = model.predict_proba(batch.tokens, keep)
        for i in range(b):
            if active[i]:
                out[i, pos] = tvd(probs[i], base[i])
    return out


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def kendall_tau_per_instance(
    attention: np.ndarray,
    importance: np.ndarray,
    mask: np.ndarray,
    min_length: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Kendall's tau per instance, plus the p-value, over unpadded positions only.

    Returns (taus, pvalues), both of shape (batch,), with NaN for instances too
    short to correlate. Reporting the p-values is what lets the results table carry
    the paper's "Sig. Frac." column, the fraction of instances where the correlation
    is statistically significant. Without it a mean tau of 0.34 says much less: the
    paper's own point is that the correlation tends to exist, just weakly.
    """
    b = attention.shape[0]
    taus = np.full(b, np.nan)
    pvals = np.full(b, np.nan)
    for i in range(b):
        keep = mask[i] > 0
        if keep.sum() < min_length:
            continue
        a, g = attention[i][keep], importance[i][keep]
        if np.allclose(a, a[0]) or np.allclose(g, g[0]):
            continue  # tau is undefined when one side is constant
        tau, p = kendalltau(a, g)
        taus[i], pvals[i] = tau, p
    return taus, pvals


def summarise(taus: np.ndarray, pvals: np.ndarray, alpha: float = 0.05) -> dict:
    """Mean, std and significant fraction, in the shape of the paper's Table 2."""
    valid = ~np.isnan(taus)
    if not valid.any():
        return {"mean": float("nan"), "std": float("nan"), "sig_frac": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(taus[valid])),
        "std": float(np.std(taus[valid])),
        "sig_frac": float(np.mean(pvals[valid] < alpha)),
        "n": int(valid.sum()),
    }
