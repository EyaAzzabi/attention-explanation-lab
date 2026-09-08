"""Train one configuration over several seeds and record the paper's measurements.

One run is one (dataset, encoder, seed). Everything it produces goes into a single
JSON file under `results/`, and `src/report.py` builds every table in the README from
those files. No number in this repository is transcribed by hand.

Why seeds are a first-class argument rather than an afterthought: the original paper
reports single runs, and the first of the four objections in Wiegreffe & Pinter (EMNLP
2019) is that attention weights vary enough between random initialisations that a
single run cannot establish the size of an effect. On SST, where the training set is
about 3,000 sentences, that objection has teeth. So the unit of reporting here is
mean and standard deviation over seeds, and `report.py` refuses to rank two
configurations whose intervals overlap.

Usage:

    python -m src.experiment --dataset sst --encoder bilstm --seeds 5
    python -m src.experiment --dataset sst --encoder transformer --seeds 5
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Vocabulary, load, tokenize
from .importance import (
    gradient_importance,
    kendall_tau_per_instance,
    leave_one_out_importance,
    summarise,
)
from .counterfactual import EPSILON_CLASSIFICATION, adversarial_attention, permutation_test
from .model import (ABLATIONS, CONTEXTUALISATION_ORDER, ENCODERS, AttentionClassifier,
                    Batch, make_optimizer)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batches(ids, masks, labels, batch_size, device, shuffle=False, rng=None):
    order = np.arange(len(ids))
    if shuffle:
        (rng or np.random.default_rng(0)).shuffle(order)
    for start in range(0, len(order), batch_size):
        sel = order[start: start + batch_size]
        yield Batch(
            tokens=torch.as_tensor(ids[sel], dtype=torch.long, device=device),
            mask=torch.as_tensor(masks[sel], dtype=torch.float, device=device),
            labels=torch.as_tensor(labels[sel], dtype=torch.long, device=device),
        )


def encode_corpus(texts, vocab, max_len):
    ids, masks = [], []
    for text in texts:
        i, m = vocab.encode(tokenize(text), max_len)
        ids.append(i)
        masks.append(m)
    return np.asarray(ids), np.asarray(masks, dtype=np.float32)


def train(model, batches_fn, epochs, device, lr=1e-3):
    opt = make_optimizer(model, lr=lr)
    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for batch in batches_fn(shuffle=True):
            logits, _ = model(batch.tokens, batch.mask)
            loss = F.cross_entropy(logits, batch.labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(batch.labels)
            n += len(batch.labels)
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / max(n, 1):.4f}")
    return model


@torch.no_grad()
def evaluate(model, batches_fn) -> dict:
    model.eval()
    correct = total = 0
    tp = fp = fn = 0
    for batch in batches_fn():
        logits, _ = model(batch.tokens, batch.mask)
        pred = logits.argmax(-1)
        correct += int((pred == batch.labels).sum())
        total += len(batch.labels)
        tp += int(((pred == 1) & (batch.labels == 1)).sum())
        fp += int(((pred == 1) & (batch.labels == 0)).sum())
        fn += int(((pred == 0) & (batch.labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": correct / max(total, 1), "f1": f1, "n": total}


def measure_correlations(model, batches, max_instances: int) -> dict:
    """tau_g and tau_loo over up to max_instances test examples.

    Leave-one-out costs one forward pass per token, so it is the reason this is
    capped. The cap is recorded in the result file rather than left implicit.
    """
    taus_g, pvals_g, taus_loo, pvals_loo = [], [], [], []
    seen = 0
    for batch in batches:
        if seen >= max_instances:
            break
        alpha = model.attention_weights(batch.tokens, batch.mask)
        mask = batch.mask.cpu().numpy()

        g = gradient_importance(model, batch, class_index=1)
        t, p = kendall_tau_per_instance(alpha, g, mask)
        taus_g.append(t)
        pvals_g.append(p)

        loo = leave_one_out_importance(model, batch)
        t, p = kendall_tau_per_instance(alpha, loo, mask)
        taus_loo.append(t)
        pvals_loo.append(p)

        seen += len(batch.labels)

    return {
        "tau_gradient": summarise(np.concatenate(taus_g), np.concatenate(pvals_g)),
        "tau_loo": summarise(np.concatenate(taus_loo), np.concatenate(pvals_loo)),
        "instances_measured": seen,
    }


def measure_counterfactual(model, batches, max_instances: int, args) -> dict:
    """Experiment 2 of the paper: does the prediction survive a different attention?

    Two summaries per configuration. The permutation median is the paper's first
    piece of evidence: if shuffling the weights barely moves the output, the heatmap
    was not carrying the prediction. The epsilon-max JSD is the second and stronger
    one: how far attention can be pushed, in Jensen-Shannon divergence, while the
    output stays within epsilon. Its ceiling is ln(2) = 0.693, and mass near that
    ceiling means the distribution can be replaced wholesale unnoticed.
    """
    perms, advs, seen = [], [], 0
    for batch in batches:
        if seen >= max_instances:
            break
        perms.append(permutation_test(model, batch, n_permutations=args.permutations))
        advs.append(adversarial_attention(
            model, batch, k=args.adversarial_k, epsilon=EPSILON_CLASSIFICATION,
            steps=args.adversarial_steps))
        seen += len(batch.labels)

    perm = np.concatenate(perms) if perms else np.array([np.nan])
    adv = np.concatenate(advs) if advs else np.array([np.nan])
    return {
        "permutation_tvd": {
            "median": float(np.nanmedian(perm)), "mean": float(np.nanmean(perm)),
            "p90": float(np.nanpercentile(perm, 90)), "n": int(len(perm)),
        },
        "adversarial_jsd": {
            "mean": float(np.nanmean(adv)), "median": float(np.nanmedian(adv)),
            "frac_above_half_bound": float(np.nanmean(adv > 0.5 * float(np.log(2)))),
            "found_frac": float(np.nanmean(adv > 0)), "n": int(len(adv)),
        },
    }


def run_one(dataset: str, encoder: str, seed: int, args) -> dict:
    device = torch.device(args.device)
    set_seed(seed)

    corpus = load(dataset)
    train_tok = [tokenize(t) for t in corpus.train_texts]
    vocab = Vocabulary.build(train_tok, min_count=args.min_count)

    tr_ids, tr_mask = encode_corpus(corpus.train_texts, vocab, args.max_len)
    te_ids, te_mask = encode_corpus(corpus.test_texts, vocab, args.max_len)

    rng = np.random.default_rng(seed)
    train_batches = lambda shuffle=False: make_batches(
        tr_ids, tr_mask, corpus.train_labels, args.batch_size, device, shuffle, rng)
    test_batches = lambda shuffle=False: make_batches(
        te_ids, te_mask, corpus.test_labels, args.batch_size, device, shuffle, rng)

    model = AttentionClassifier(len(vocab), corpus.n_classes, encoder).to(device)
    started = time.time()
    train(model, train_batches, args.epochs, device, lr=args.lr)

    result = {
        "dataset": dataset,
        "encoder": encoder,
        "seed": seed,
        "vocab_size": len(vocab),
        "corpus": corpus.describe(),
        "performance": evaluate(model, test_batches),
        "seconds": round(time.time() - started, 1),
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "max_len": args.max_len, "min_count": args.min_count,
            "correlation_sample": args.correlation_sample,
        },
    }
    result.update(measure_correlations(model, test_batches(), args.correlation_sample))
    if args.counterfactual:
        result.update(measure_counterfactual(
            model, test_batches(), args.counterfactual_sample, args))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sst")
    ap.add_argument("--encoder", default="bilstm",
                    choices=sorted(ENCODERS) + ["all", "ablations", "everything"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--correlation-sample", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--counterfactual", action="store_true",
                    help="also run experiment 2 (permutation and adversarial attention)")
    ap.add_argument("--counterfactual-sample", type=int, default=96)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--adversarial-k", type=int, default=4)
    ap.add_argument("--adversarial-steps", type=int, default=500)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    encoders = {
        "all": CONTEXTUALISATION_ORDER,
        "ablations": ABLATIONS,
        "everything": CONTEXTUALISATION_ORDER + ABLATIONS,
    }.get(args.encoder, [args.encoder])

    for encoder in encoders:
        for seed in range(args.seeds):
            print(f"\n== {args.dataset} / {encoder} / seed {seed}")
            result = run_one(args.dataset, encoder, seed, args)
            out = RESULTS / f"{args.dataset}__{encoder}__seed{seed}.json"
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            perf = result["performance"]
            tau = result["tau_gradient"]
            print(f"  acc {perf['accuracy']:.3f}  f1 {perf['f1']:.3f}  "
                  f"tau_g {tau['mean']:.3f} +/- {tau['std']:.3f}  -> {out.name}")


if __name__ == "__main__":
    main()
