"""Experiment 2 of Jain & Wallace (2019): counterfactual attention distributions.

Two procedures, both of which keep the encoder output h fixed and change only the
weights over it. The paper flags this explicitly in Algorithms 2 and 3 ("h is not
changed"), and it is the whole point: the question is not what a different model would
have predicted, it is whether *this* model would have predicted the same thing while
pointing somewhere else.

  4.2.1  Permutation. Shuffle the attention weights 100 times, record the median
         total variation distance in the output. The paper reports a median output
         difference of 0.006 for the example in Figure 1.

  4.2.2  Adversarial attention. Find k distributions as far as possible from the
         observed one, and from each other, while keeping the output within epsilon:

             maximise   sum_i JSD[a_i, a_hat] + 1/(k(k-1)) sum_{i<j} JSD[a_i, a_j]
             s.t.       TVD[y(x, a_i), y(x, a_hat)] <= epsilon

         Optimised as the relaxed objective of Equation 2 with Adam and lambda = 500,
         epsilon = 0.01 for classification and 0.05 for QA.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .importance import JSD_UPPER_BOUND, jsd, tvd

EPSILON_CLASSIFICATION = 0.01
EPSILON_QA = 0.05
ADVERSARIAL_LAMBDA = 500.0


@torch.no_grad()
def permutation_test(model, batch, n_permutations: int = 100, rng=None) -> np.ndarray:
    """Median TVD in output over n random permutations of the attention weights.

    Returns one value per instance. A small number here means the model reaches the
    same answer while attending almost anywhere, which is the paper's first piece of
    evidence against reading a heatmap as an explanation.
    """
    rng = rng or np.random.default_rng(0)
    alpha = torch.as_tensor(model.attention_weights(batch.tokens, batch.mask))
    base = model.predict_proba(batch.tokens, batch.mask)
    lengths = batch.mask.sum(dim=1).long().cpu().numpy()

    b = alpha.shape[0]
    medians = np.zeros(b)
    for i in range(b):
        n = int(lengths[i])
        if n < 2:
            continue
        diffs = []
        row = alpha[i, :n].cpu().numpy()
        for _ in range(n_permutations):
            permuted = alpha[i].clone()
            permuted[:n] = torch.as_tensor(rng.permutation(row))
            probs = model.decode_from_attention(
                batch.tokens[i: i + 1], batch.mask[i: i + 1], permuted.unsqueeze(0)
            )
            diffs.append(tvd(probs[0], base[i]))
        medians[i] = float(np.median(diffs))
    return medians


def adversarial_attention(
    model,
    batch,
    k: int = 4,
    epsilon: float = EPSILON_CLASSIFICATION,
    steps: int = 500,
    lr: float = 0.01,
    lam: float = ADVERSARIAL_LAMBDA,
) -> np.ndarray:
    """epsilon-max JSD per instance: how far can attention move for free?

    Returns one value per instance, the largest JSD from the observed distribution
    among the k adversarial candidates that stayed within epsilon of the original
    output. Zero means no candidate qualified. Values near the JSD upper bound of
    0.69 mean the attention distribution can be replaced wholesale without the model
    noticing, which is the paper's strongest claim.
    """
    model.eval()
    with torch.no_grad():
        h = model.encoder(model.embedding(batch.tokens), batch.mask)
        alpha_hat = model.attention(h, batch.mask)
        y_hat = F.softmax(model.decoder(torch.bmm(alpha_hat.unsqueeze(1), h).squeeze(1)), -1)

    b, t = alpha_hat.shape
    # k free parameter sets per instance, softmaxed into distributions.
    logits = torch.zeros(k, b, t, device=alpha_hat.device).normal_(0, 0.1)
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    neg_inf_mask = (batch.mask == 0).unsqueeze(0).expand(k, b, t)

    for _ in range(steps):
        opt.zero_grad()
        alphas = F.softmax(logits.masked_fill(neg_inf_mask, float("-inf")), dim=-1)

        # Outputs under each candidate, h held fixed.
        ctx = torch.einsum("kbt,bth->kbh", alphas, h)
        y = F.softmax(model.decoder(ctx), dim=-1)
        constraint = 0.5 * (y - y_hat.unsqueeze(0)).abs().sum(-1)      # TVD, (k, B)

        # Distance from the observed distribution, and from each other.
        far_from_original = _jsd_torch(alphas, alpha_hat.unsqueeze(0)).sum(0)
        pairwise = 0.0
        if k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    pairwise = pairwise + _jsd_torch(alphas[i], alphas[j])
            pairwise = pairwise / (k * (k - 1))

        objective = far_from_original + pairwise
        penalty = lam * torch.clamp(constraint - epsilon, min=0).sum(0)
        (-(objective - penalty)).sum().backward()
        opt.step()

    with torch.no_grad():
        alphas = F.softmax(logits.masked_fill(neg_inf_mask, float("-inf")), dim=-1)
        ctx = torch.einsum("kbt,bth->kbh", alphas, h)
        y = F.softmax(model.decoder(ctx), dim=-1)
        within = (0.5 * (y - y_hat.unsqueeze(0)).abs().sum(-1) <= epsilon)   # (k, B)
        divergence = _jsd_torch(alphas, alpha_hat.unsqueeze(0))              # (k, B)
        divergence = divergence.masked_fill(~within, 0.0)
        best = divergence.max(dim=0).values

    result = best.cpu().numpy()
    if (result > JSD_UPPER_BOUND + 1e-6).any():
        raise AssertionError("JSD exceeded ln(2): the distributions are not normalised")
    return result


def _jsd_torch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable JSD along the last dimension, broadcasting over the rest."""
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(-1)
    return 0.5 * kl_pm + 0.5 * kl_qm
