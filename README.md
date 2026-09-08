# attention-explanation-lab

<!-- Badges go live after the first push and first CI run. Same set as
     rag-eval-lab and daily-climate-pipeline. -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![no API key](https://img.shields.io/badge/API%20key-not%20required-success.svg)](#running-it)

**Jain and Wallace showed attention weights correlate weakly with feature importance
under a BiLSTM encoder. Their own table shows the correlation nearly doubles when the
encoder does not contextualise at all. So where does a Transformer land?**

This repository reproduces *Attention is not Explanation* (Jain & Wallace, NAACL 2019)
and then answers a question the paper left open: it tested BiLSTM, CNN and average
embedding encoders, and explicitly excluded self-attention architectures. Seven years
later, the encoder everyone actually uses is the one that was never measured.

---

## The observation the extension is built on

Table 2 of the paper reports Kendall's tau between attention weights and gradient-based
feature importance. Reading it by encoder rather than by dataset:

| Dataset | BiLSTM tau_g | Average tau_g | Gap |
|---|---|---|---|
| SST | 0.34 / 0.36 | 0.61 / 0.60 | +0.26 |
| IMDB | 0.44 / 0.43 | 0.67 / 0.68 | +0.24 |
| 20 News | 0.07 / 0.21 | 0.79 / 0.75 | +0.63 |
| AG News | 0.36 / 0.42 | 0.78 / 0.76 | +0.40 |

The authors state the effect in one sentence: gradients in average embedding models
show "very high degree of correspondence" with attention, roughly **0.375 higher on
average** than for the BiLSTM. They attribute it to contextualisation. A BiLSTM hidden
state at position *t* already contains the whole sentence, so attending to position *t*
is not the same as using token *t*.

**That is a claim with a direction.** If contextualisation is what breaks the
correspondence, then a Transformer encoder, where every position mixes with every other
at every layer, should break it further. If instead the correlation recovers, the
explanation is wrong and something else is going on.

Nobody has to guess. It is measurable, on the paper's own datasets, with the paper's own
metric.

---

## What is reproduced

Both experiments from the original paper, on a subset of its datasets (SST, IMDB,
AG News, 20 Newsgroups):

**Experiment 1, correlation.** Kendall's tau between attention weights and (a)
gradient-based importance, (b) leave-one-out importance. The gradient is computed with
the computation graph cut at the attention module, as the paper specifies in Appendix B,
so attention is treated as a separate input and the gradient does not flow through it.

**Experiment 2, counterfactual attention.** Attention permutation (100 permutations per
instance, median total variation distance in output) and adversarial attention, which
maximises Jensen-Shannon divergence from the observed distribution subject to the output
moving by less than epsilon. Epsilon is 0.01 for classification, as in the paper.

Model configuration follows Appendix A exactly: embedding size 300, BiLSTM hidden size
128, additive (Bahdanau) attention, L2 regularisation at 1e-5, Adam with PyTorch
defaults, spaCy tokenisation, numeric tokens mapped to `qqq`.

## What is added

| Addition | Why |
|---|---|
| **Transformer encoder** | The gap in the original. Same attention head, same metrics, only the encoder changes. |
| **Five seeds per configuration** | The original reports single runs. The rebuttal (Wiegreffe & Pinter, EMNLP 2019) makes seed variance one of its four objections, so it is built in here rather than added later. |
| **Correlation as a function of contextualisation** | Average, CNN, BiLSTM, Transformer on one axis. This is the actual result of the extension. |

---

## Results

<!-- FILL IN from results/. Do not hand-type: src/report.py regenerates this table.
     Keep the paper column visible so the reproduction gap is legible.
     Report mean +/- std over 5 seeds, and say when two rows overlap. -->

### Reproduction: Kendall tau between attention and gradient importance

| Dataset | Encoder | tau_g (this repo) | tau_g (paper) | Delta |
|---|---|---|---|---|
| SST | BiLSTM | | 0.34 / 0.36 | |
| SST | Average | | 0.61 / 0.60 | |
| IMDB | BiLSTM | | 0.44 / 0.43 | |
| IMDB | Average | | 0.67 / 0.68 | |

### Extension: where the Transformer falls

| Dataset | Average | CNN | BiLSTM | **Transformer** |
|---|---|---|---|---|
| SST | | | | |
| IMDB | | | | |
| AG News | | | | |

---

## What this does not show

- Kendall's tau against gradients is not a measure of truth. The original paper says so
  itself: it does not claim gradient importance is ground truth, only that attention
  correlates poorly with several independent measures of it.
- Results are on classification with unstructured output spaces. Sequence-to-sequence
  tasks are outside this, as they were outside the original.
- A low correlation for the Transformer would not prove its attention is meaningless.
  It would show that the same argument the paper makes about BiLSTMs applies at least as
  strongly to the architecture that replaced them.

## Running it

```bash
pip install -r requirements.txt
python -m src.experiment --dataset sst --encoder bilstm --seeds 5
python -m src.experiment --dataset sst --encoder transformer --seeds 5
python -m src.report                 # regenerates every table above from results/
```

Seeds are fixed, versions are pinned, and no number in this README is typed by hand.

## The papers

- Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation.* NAACL-HLT.
  [aclanthology.org/N19-1357](https://aclanthology.org/N19-1357/) ·
  [code](https://github.com/successar/AttentionExplanation)
- Wiegreffe, S., & Pinter, Y. (2019). *Attention is not not Explanation.* EMNLP.
  [aclanthology.org/D19-1002](https://aclanthology.org/D19-1002/) ·
  [code](https://github.com/sarahwie/attention)
