"""Encoders and the attention head, following Appendix A of Jain & Wallace (2019).

The whole design constraint of this file: **only the encoder changes.** The embedding
size, the attention mechanism, the decoder and the training loop are identical across
Average, CNN, BiLSTM and Transformer. If the Kendall tau moves, the encoder is the only
thing that could have moved it. That is what makes the extension an experiment rather
than four unrelated runs.

Hyperparameters are the paper's, not invented:

  * embedding size 300 for every encoder                        (A.1, A.2, A.3)
  * BiLSTM hidden size 128                                      (A.1)
  * CNN kernels [1, 3, 5, 7], 64 filters each, hidden 256       (A.2)
  * Average: projection 256 with ReLU                           (A.3)
  * additive (Bahdanau) attention: v^T tanh(W1 h + W2 Q)        (Section 2)
  * L2 regularisation lambda = 1e-5 on all parameters           (A.1)
  * Adam with PyTorch defaults, maximum likelihood loss         (A.1)

The Transformer encoder is the addition. It is given a hidden size of 256 so it sits in
the same range as the CNN and the BiLSTM (2 x 128), because a width difference would be
an obvious confound in a comparison whose entire subject is the encoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 300
L2_LAMBDA = 1e-5


@dataclass
class Batch:
    tokens: torch.Tensor   # (B, T) long
    mask: torch.Tensor     # (B, T) float, 1.0 for real tokens
    labels: torch.Tensor   # (B,) long


# --------------------------------------------------------------------------- #
# Encoders. Each maps (B, T, EMBED_DIM) -> (B, T, hidden)
# --------------------------------------------------------------------------- #
class AverageEncoder(nn.Module):
    """No contextualisation at all: a per-token projection.

    This is the paper's 'average' variant and it is the anchor of the extension.
    It is where attention and gradient importance agree most closely, because
    position t carries information about token t and nothing else.
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.proj = nn.Linear(EMBED_DIM, hidden)
        self.hidden = hidden

    def forward(self, x, mask):
        return F.relu(self.proj(x))


class CNNEncoder(nn.Module):
    """Local contextualisation, bounded by the widest kernel."""

    def __init__(self, kernels=(1, 3, 5, 7), filters: int = 64):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(EMBED_DIM, filters, k, padding=k // 2) for k in kernels
        )
        self.hidden = filters * len(kernels)

    def forward(self, x, mask):
        z = x.transpose(1, 2)                       # (B, d, T)
        outs = []
        for conv in self.convs:
            h = F.relu(conv(z))
            outs.append(h[..., : x.size(1)])        # even kernels pad one extra
        return torch.cat(outs, dim=1).transpose(1, 2)


class BiLSTMEncoder(nn.Module):
    """Full sequence contextualisation. The paper's headline encoder."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.rnn = nn.LSTM(EMBED_DIM, hidden, batch_first=True, bidirectional=True)
        self.hidden = hidden * 2

    def forward(self, x, mask):
        out, _ = self.rnn(x)
        return out


class _TransformerLayer(nn.Module):
    """One pre-norm block, with the residual connections switchable.

    `nn.TransformerEncoderLayer` bakes the residual connections in and offers no way
    to remove them, which is why this is written out by hand. Turning them off is the
    experiment: the hypothesis under test is that a Transformer's residual stream
    keeps a direct additive path from the embedding at position t to the hidden state
    at position t, and that the gradient travels that path while token deletion does
    not. If that is what separates the two importance measures, removing the
    connections should bring them back together.
    """

    def __init__(self, hidden: int, heads: int, residual: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.residual = residual

    def forward(self, x, key_padding_mask):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x),
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a if self.residual else a
        f = self.ff(self.n2(x))
        return x + f if self.residual else f


class TransformerEncoder(nn.Module):
    """The encoder the paper never tested, and the reason this repository exists.

    Every position mixes with every other position at every layer. If the paper's
    explanation for the weak correlation is right, that attention over a BiLSTM state
    is not attention over a token because the state already holds the sentence, then
    this is the same problem made worse.

    `residual` and `layers` exist so the residual-stream hypothesis can be tested
    rather than asserted. Without the residual connections a deep stack trains
    poorly, so any run with `residual=False` has to be read next to its accuracy: a
    correlation measured on a model that did not learn says nothing.
    """

    def __init__(self, hidden: int = 256, layers: int = 2, heads: int = 4,
                 residual: bool = True):
        super().__init__()
        self.inp = nn.Linear(EMBED_DIM, hidden)
        self.layers = nn.ModuleList(
            _TransformerLayer(hidden, heads, residual) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.hidden = hidden
        self._pos = None

    def _positional(self, t: int, d: int, device) -> torch.Tensor:
        if self._pos is not None and self._pos.size(0) >= t:
            return self._pos[:t].to(device)
        pos = torch.arange(t, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(t, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self._pos = pe
        return pe.to(device)

    def forward(self, x, mask):
        h = self.inp(x)
        h = h + self._positional(h.size(1), h.size(2), h.device)
        pad = (mask == 0)
        for layer in self.layers:
            h = layer(h, pad)
        return self.norm(h)


#: The four encoders of the main comparison, plus the ablations that test the
#: residual-stream hypothesis rather than leaving it as a story. `transformer_nores`
#: removes the direct additive path the hypothesis rests on; `transformer_l4`
#: doubles the depth, which should widen the gap if the path is what causes it.
ENCODERS = {
    "average": AverageEncoder,
    "cnn": CNNEncoder,
    "bilstm": BiLSTMEncoder,
    "transformer": TransformerEncoder,
    "transformer_nores": lambda: TransformerEncoder(residual=False),
    "transformer_l1": lambda: TransformerEncoder(layers=1),
    "transformer_l4": lambda: TransformerEncoder(layers=4),
}

#: Ordered by how much the encoder mixes information across positions. This is the
#: x-axis of the extension's headline figure, and the order is an argument, not a
#: convenience: the paper's own numbers say tau falls as you move right.
CONTEXTUALISATION_ORDER = ["average", "cnn", "bilstm", "transformer"]

#: The residual-stream ablations. Kept out of CONTEXTUALISATION_ORDER because they
#: are not points on the contextualisation axis; they are controls on one encoder.
ABLATIONS = ["transformer_l1", "transformer_l4", "transformer_nores"]


# --------------------------------------------------------------------------- #
# Attention head and full model
# --------------------------------------------------------------------------- #
class AdditiveAttention(nn.Module):
    """phi(h, Q) = v^T tanh(W1 h + W2 Q), with Q absent for plain classification."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def scores(self, h, mask):
        # A row with no unmasked position would softmax over a vector of -inf and
        # produce NaN, which then propagates through the gradient and quietly ruins
        # the run. data.drop_empty removes such instances upstream; this is the
        # tripwire that fires if one ever gets through, because a loud failure is
        # worth far more here than a plausible-looking number.
        if not bool((mask.sum(dim=1) > 0).all()):
            raise ValueError(
                "an instance has no unmasked positions: attention would be NaN. "
                "Empty documents must be removed before batching (see data.drop_empty)"
            )
        s = self.v(torch.tanh(self.w1(h))).squeeze(-1)     # (B, T)
        return s.masked_fill(mask == 0, float("-inf"))

    def forward(self, h, mask):
        return F.softmax(self.scores(h, mask), dim=-1)


class AttentionClassifier(nn.Module):
    def __init__(self, vocab: int, n_classes: int, encoder: str,
                 embeddings: np.ndarray | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab, EMBED_DIM, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embeddings))
        self.encoder = ENCODERS[encoder]()
        self.attention = AdditiveAttention(self.encoder.hidden)
        self.decoder = nn.Linear(self.encoder.hidden, n_classes)
        self.encoder_name = encoder

    # -- pieces, so importance.py can hold on to the embedding tensor ---------- #
    def embed(self, tokens):
        e = self.embedding(tokens)
        e.requires_grad_(True)
        return e

    def forward_from_embeddings(self, embeddings, mask, detach_attention: bool = False):
        h = self.encoder(embeddings, mask)
        alpha = self.attention(h, mask)
        if detach_attention:
            # Appendix B: cut the graph at the attention module so the gradient does
            # not flow through it. Without this line the measured correlation is not
            # the paper's measured correlation.
            alpha = alpha.detach()
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return self.decoder(context), alpha

    def forward(self, tokens, mask):
        return self.forward_from_embeddings(self.embedding(tokens), mask)

    # -- convenience ---------------------------------------------------------- #
    @torch.no_grad()
    def predict_proba(self, tokens, mask) -> np.ndarray:
        logits, _ = self.forward(tokens, mask)
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def attention_weights(self, tokens, mask) -> np.ndarray:
        _, alpha = self.forward(tokens, mask)
        return alpha.cpu().numpy()

    @torch.no_grad()
    def decode_from_attention(self, tokens, mask, alpha) -> np.ndarray:
        """Re-run the decoder with a substituted attention distribution.

        Both counterfactual experiments need this: h is *not* recomputed, only the
        weights over it change. Algorithm 2 in the paper carries the same note.
        """
        h = self.encoder(self.embedding(tokens), mask)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return F.softmax(self.decoder(context), dim=-1).cpu().numpy()


def make_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """Adam with PyTorch defaults and the paper's L2 term (A.1)."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_LAMBDA)
