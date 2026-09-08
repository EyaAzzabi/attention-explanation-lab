"""Corpus download, checksum manifest, and the paper's preprocessing.

Option B: nothing is fetched silently at run time. A corpus is downloaded once into
`data/raw/`, its SHA-256 is recorded in `data/MANIFEST.json`, and every later run
verifies the bytes against that manifest before touching them.

The manifest is committed. That is the point: it lets a stranger who clones this
repository find out whether the file they downloaded is the file the results were
computed on. On a project whose entire subject is whether a published result
reproduces, depending on a silent download would be a contradiction.

Checksums are **recorded on first download, not hard-coded ahead of time**. Hard-coding
a hash nobody has verified is worse than no hash at all: it looks like a guarantee and
is a guess. Once a corpus has been fetched, the manifest entry becomes the pin.

Preprocessing follows Appendix A of the paper:
  * spaCy for tokenisation. The blank English pipeline is used, which is the tokeniser
    alone and needs no model download; the paper uses spaCy only to tokenise.
  * every token containing a digit is mapped to `qqq`
  * out-of-vocabulary tokens map to `<unk>`
  * SST: neutral sentences dropped, (4,5) positive and (1,2) negative
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.json"

PAD, UNK, NUM = "<pad>", "<unk>", "qqq"


@dataclass
class Corpus:
    name: str
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    n_classes: int
    source: str = ""
    notes: str = ""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "train": len(self.train_texts),
            "test": len(self.test_texts),
            "classes": self.n_classes,
            "source": self.source,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Download with a recorded checksum
# --------------------------------------------------------------------------- #
SOURCES: dict[str, str] = {
    "sst": "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip",
    "imdb": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")


def fetch(name: str, force: bool = False) -> Path:
    """Download a corpus once, then verify it against the manifest on every later run.

    Raises if the bytes on disk do not match a checksum already in the manifest. That
    is the whole reason the manifest exists, so it is an error and not a warning.
    """
    if name not in SOURCES:
        raise KeyError(f"no download URL registered for {name!r}")

    RAW.mkdir(parents=True, exist_ok=True)
    url = SOURCES[name]
    target = RAW / url.rsplit("/", 1)[-1]
    manifest = _load_manifest()

    if force and target.exists():
        target.unlink()

    if not target.exists():
        print(f"downloading {name} from {url}")
        urllib.request.urlretrieve(url, target)

    digest = _sha256(target)
    recorded = manifest.get(name, {}).get("sha256")

    if recorded is None:
        manifest[name] = {
            "url": url,
            "file": target.name,
            "sha256": digest,
            "bytes": target.stat().st_size,
        }
        _save_manifest(manifest)
        print(f"recorded checksum for {name}: {digest[:16]}...")
    elif recorded != digest:
        raise RuntimeError(
            f"{target.name} does not match the manifest.\n"
            f"  expected {recorded}\n  found    {digest}\n"
            "The results in this repository were computed on the expected bytes. "
            "Delete the file and re-fetch, or update the manifest deliberately."
        )
    return target


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_sst() -> Corpus:
    """Stanford Sentiment Treebank, binarised as the paper describes.

    10,662 sentences scored 1 to 5. Neutral (3) is dropped, (4,5) becomes positive
    and (1,2) negative, leaving roughly the 3034/3321 train and 863/862 test split
    reported in Table 1.
    """
    path = fetch("sst")
    with zipfile.ZipFile(path) as z:
        def read(member: str) -> list[str]:
            name = next(n for n in z.namelist() if n.endswith(member))
            return z.read(name).decode("utf-8", "replace").splitlines()

        sentences = {}
        for line in read("datasetSentences.txt")[1:]:
            idx, text = line.split("\t", 1)
            sentences[int(idx)] = text.strip()

        splits = {}
        for line in read("datasetSplit.txt")[1:]:
            idx, split = line.split(",")
            splits[int(idx)] = int(split)          # 1 train, 2 test, 3 dev

        phrase_id = {}
        for line in read("dictionary.txt"):
            if "|" in line:
                phrase, pid = line.rsplit("|", 1)
                phrase_id[phrase.strip()] = int(pid)

        sentiment = {}
        for line in read("sentiment_labels.txt")[1:]:
            pid, value = line.split("|")
            sentiment[int(pid)] = float(value)

    buckets: dict[int, tuple[list[str], list[int]]] = {1: ([], []), 2: ([], [])}
    for idx, text in sentences.items():
        split = splits.get(idx)
        if split not in buckets:
            continue
        pid = phrase_id.get(text.strip())
        if pid is None:
            continue
        score = sentiment[pid]
        if 0.4 < score <= 0.6:                     # neutral, dropped
            continue
        label = 1 if score > 0.6 else 0
        buckets[split][0].append(text)
        buckets[split][1].append(label)

    return Corpus(
        name="sst",
        train_texts=buckets[1][0], train_labels=np.array(buckets[1][1]),
        test_texts=buckets[2][0], test_labels=np.array(buckets[2][1]),
        n_classes=2,
        source=SOURCES["sst"],
        notes=(
            "neutral scores in (0.4, 0.6] dropped; positive if score > 0.6. "
            "KNOWN GAP: this yields 3122/3446 train and 873/876 test, against the "
            "3034/3321 and 863/862 of the paper's Table 1, about 3 per cent more "
            "sentences in every cell. The likely cause is sentence-to-phrase "
            "matching in dictionary.txt, where a few sentences fail to match on "
            "escaping and are dropped; the paper does not say how many it lost. "
            "The gap is recorded rather than closed, because tuning the "
            "preprocessing until the counts match would be fitting the pipeline to "
            "the target and would make the reproduction meaningless."
        ),
    )


def load_20news() -> Corpus:
    """20 Newsgroups, hockey against baseball, as in the paper.

    scikit-learn caches this under ~/scikit_learn_data on first call, so it is
    downloaded once like everything else here. It has no manifest entry because
    scikit-learn owns the file layout; the subset and the split are pinned instead.
    """
    from sklearn.datasets import fetch_20newsgroups

    cats = ["rec.sport.hockey", "rec.sport.baseball"]
    kw = dict(categories=cats, remove=("headers", "footers", "quotes"))
    tr = fetch_20newsgroups(subset="train", **kw)
    te = fetch_20newsgroups(subset="test", **kw)
    return Corpus(
        name="20news",
        train_texts=list(tr.data), train_labels=np.asarray(tr.target),
        test_texts=list(te.data), test_labels=np.asarray(te.target),
        n_classes=2,
        source="sklearn.datasets.fetch_20newsgroups",
        notes="baseball=0, hockey=1; headers, footers and quotes removed",
    )


def load_imdb(max_per_class: int | None = None) -> Corpus:
    """IMDB Large Movie Review Corpus, 25k train and 25k test, binary."""
    path = fetch("imdb")
    data = {"train": ([], []), "test": ([], [])}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if len(parts) < 4 or not member.isfile() or not member.name.endswith(".txt"):
                continue
            _, split, polarity, _ = parts[0], parts[1], parts[2], parts[3]
            if split not in data or polarity not in ("pos", "neg"):
                continue
            label = 1 if polarity == "pos" else 0
            if max_per_class is not None:
                if sum(1 for l in data[split][1] if l == label) >= max_per_class:
                    continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            data[split][0].append(fh.read().decode("utf-8", "replace"))
            data[split][1].append(label)

    return Corpus(
        name="imdb",
        train_texts=data["train"][0], train_labels=np.array(data["train"][1]),
        test_texts=data["test"][0], test_labels=np.array(data["test"][1]),
        n_classes=2,
        source=SOURCES["imdb"],
    )


LOADERS = {"sst": load_sst, "20news": load_20news, "imdb": load_imdb}


def drop_empty(corpus: Corpus) -> Corpus:
    """Remove documents that tokenise to nothing, and record how many.

    This is not housekeeping. A document with no tokens produces an all-zero mask,
    the attention softmax then runs over a row of -inf, and the result is NaN, which
    propagates through the gradient and silently destroys the whole run: the loss
    prints as nan, accuracy sits at chance, and every Kendall tau comes back nan.

    20 Newsgroups with headers, footers and quotes removed contains 37 such documents
    in train and 23 in test. Dropping them is a deliberate deviation from the corpus
    as distributed, so it is counted and reported rather than done quietly.
    """
    def keep(texts, labels):
        idx = [i for i, text in enumerate(texts) if tokenize(text)]
        return [texts[i] for i in idx], np.asarray(labels)[idx], len(texts) - len(idx)

    tr_texts, tr_labels, n_tr = keep(corpus.train_texts, corpus.train_labels)
    te_texts, te_labels, n_te = keep(corpus.test_texts, corpus.test_labels)

    if n_tr or n_te:
        note = f"dropped {n_tr} empty train and {n_te} empty test documents"
        corpus.notes = f"{corpus.notes}; {note}" if corpus.notes else note
    corpus.train_texts, corpus.train_labels = tr_texts, tr_labels
    corpus.test_texts, corpus.test_labels = te_texts, te_labels
    return corpus


def load(name: str, **kwargs) -> Corpus:
    if name not in LOADERS:
        raise KeyError(f"unknown corpus {name!r}; have {sorted(LOADERS)}")
    return drop_empty(LOADERS[name](**kwargs))


# --------------------------------------------------------------------------- #
# Tokenisation and vocabulary, per Appendix A
# --------------------------------------------------------------------------- #
_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import spacy
        # Blank pipeline: the tokeniser only, no model download needed. The paper
        # uses spaCy for tokenisation and nothing else.
        _TOKENIZER = spacy.blank("en").tokenizer
    return _TOKENIZER


def tokenize(text: str) -> list[str]:
    """Lowercased spaCy tokens, with any token containing a digit mapped to `qqq`."""
    out = []
    for tok in _tokenizer()(text):
        s = tok.text.strip().lower()
        if not s:
            continue
        out.append(NUM if any(c.isdigit() for c in s) else s)
    return out


@dataclass
class Vocabulary:
    itos: list[str] = field(default_factory=lambda: [PAD, UNK])
    stoi: dict[str, int] = field(default_factory=lambda: {PAD: 0, UNK: 1})

    @classmethod
    def build(cls, tokenized: list[list[str]], min_count: int = 1) -> "Vocabulary":
        counts: dict[str, int] = {}
        for doc in tokenized:
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
        vocab = cls()
        for tok, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if n >= min_count:
                vocab.stoi[tok] = len(vocab.itos)
                vocab.itos.append(tok)
        return vocab

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], list[float]]:
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        mask = [1.0] * len(ids)
        pad = max_len - len(ids)
        return ids + [0] * pad, mask + [0.0] * pad

    def __len__(self) -> int:
        return len(self.itos)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="fetch a corpus and record its checksum")
    ap.add_argument("name", choices=sorted(set(LOADERS) | set(SOURCES)))
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.name in SOURCES:
        fetch(args.name, force=args.force)
    corpus = load(args.name)
    print(json.dumps(corpus.describe(), indent=2))
    print("manifest:", MANIFEST)
