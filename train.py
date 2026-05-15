"""
Health Fake News Detection — RoBERTa + Evidence Ranking
========================================================
Solves RoBERTa's 512-token limit by replacing blind truncation of main_text
with top-k sentence selection ranked by semantic similarity to the claim.

Pipeline:
  claim + explanation + top_k_evidence  →  RoBERTa-base  →  4-class label
                                                          (true/false/mixture/unproven)
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def sent_tokenize(text: str) -> list[str]:
    """
    Lightweight regex sentence splitter — no NLTK download required.
    Splits on '.', '!', '?' followed by whitespace + capital letter.
    Falls back gracefully on edge cases.
    """
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"])', text.strip())
    return [s.strip() for s in parts if s.strip()]

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME   = "roberta-base"
MAX_TOKENS   = 512
TOP_K        = 5        # top-k sentences selected from main_text
BATCH_SIZE   = 16
GRAD_ACCUM   = 2        # effective batch = 32
EPOCHS       = 5
LR           = 2e-5
WARMUP_FRAC  = 0.10
SEED         = 42

LABEL_MAP    = {0: "true", 1: "false", 2: "unproven", 3: "mixture"}
NUM_LABELS   = 4

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ─── EVIDENCE RANKER ─────────────────────────────────────────────────────────

class EvidenceRanker:
    """
    Ranks sentences in main_text by TF-IDF cosine similarity to the claim.
    Returns the top-k most relevant sentences joined as a string.

    Why TF-IDF and not a neural embedder?
    - Zero extra GPU memory at training time
    - Fast enough for 11 k samples in seconds
    - Kotonya & Toni (2020) show it's competitive with neural ranking
      for this specific task (arXiv:2010.09926)
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def rank(self, claim: str, main_text: str) -> str:
        if not main_text or not main_text.strip():
            return ""

        sentences = sent_tokenize(main_text)
        if len(sentences) == 0:
            return ""
        if len(sentences) <= self.top_k:
            return " ".join(sentences)        # nothing to rank

        # Fit TF-IDF on [claim] + all sentences so vocab is shared
        corpus = [claim] + sentences
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=10_000)
            vecs  = tfidf.fit_transform(corpus)
        except ValueError:
            # corpus too sparse (e.g., all stop-words) → fallback
            return " ".join(sentences[: self.top_k])

        claim_vec = vecs[0]          # shape (1, vocab)
        sent_vecs = vecs[1:]         # shape (n_sents, vocab)

        scores = cosine_similarity(claim_vec, sent_vecs).flatten()  # (n_sents,)
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        top_idx_sorted = sorted(top_idx)                             # preserve order

        return " ".join(sentences[i] for i in top_idx_sorted)


ranker = EvidenceRanker(top_k=TOP_K)

# ─── TOKENIZER ───────────────────────────────────────────────────────────────

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)


def build_input(claim: str, explanation: str, main_text: str) -> str:
    """
    Build the text fed to RoBERTa.

    Token budget (512 total):
      [CLS]  claim  [SEP][SEP]  explanation  [SEP][SEP]  top_k_evidence  [SEP]
      ≈  2   ~60       2         ~100            2           ~346            1  = 512

    claim and explanation are preserved in full (they are short).
    main_text is replaced by the top-k ranked sentences.
    """
    evidence = ranker.rank(claim, main_text)

    # RoBERTa uses </s> as separator; tokenizer handles special tokens
    # We concatenate with [SEP] markers so the model sees structure.
    # RobertaTokenizerFast with text_pair encodes as:
    #   <s> A </s></s> B </s>
    # For 3 segments we concatenate B = explanation + </s></s> + evidence
    # so the two separator boundaries are preserved in attention.
    part_a = claim.strip()
    part_b = f"{explanation.strip()} {tokenizer.sep_token}{tokenizer.sep_token} {evidence.strip()}"
    return part_a, part_b


def tokenize_sample(sample):
    claim       = sample.get("claim",       "") or ""
    explanation = sample.get("explanation", "") or ""
    main_text   = sample.get("main_text",   "") or ""

    part_a, part_b = build_input(claim, explanation, main_text)

    enc = tokenizer(
        part_a,
        part_b,
        max_length=MAX_TOKENS,
        truncation="only_second",   # never truncate the claim
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "label":          torch.tensor(sample["label"], dtype=torch.long),
    }

# ─── DATASET ─────────────────────────────────────────────────────────────────

class PubHealthDataset(Dataset):
    def __init__(self, hf_split):
        self.samples = [tokenize_sample(s) for s in hf_split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


DATA_DIR = "/export/hhome/nlp203/NLP/data/pubhealth"

def load_pubhealth():
    """Load the three TSV splits directly from the local VM filesystem.
    No HuggingFace Hub connection required.
    """
    import pandas as pd
    from datasets import DatasetDict, Dataset as HFDataset

    label_map_str = {"true": 0, "false": 1, "unproven": 2, "mixture": 3}

    def read_tsv(path):
        print(f"  Reading {path}")
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
        df = df.dropna(subset=["claim", "label"])
        df["label"] = df["label"].str.strip().str.lower().map(label_map_str)
        df = df.dropna(subset=["label"])   # drop rows with unrecognised label strings
        df["label"] = df["label"].astype(int)
        for col in ["explanation", "main_text"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("")
        return HFDataset.from_pandas(
            df[["claim", "explanation", "main_text", "label"]].reset_index(drop=True)
        )

    ds = DatasetDict({
        "train":      read_tsv(f"{DATA_DIR}/train.tsv"),
        "validation": read_tsv(f"{DATA_DIR}/dev.tsv"),
        "test":       read_tsv(f"{DATA_DIR}/test.tsv"),
    })

    print(f"  Loaded — train={len(ds['train'])}  val={len(ds['validation'])}  test={len(ds['test'])}")
    return ds["train"], ds["validation"], ds["test"]

# ─── CLASS WEIGHTS ───────────────────────────────────────────────────────────

def compute_class_weights(hf_split) -> torch.Tensor:
    """Inverse-frequency weights so rare classes (unproven) get higher penalty."""
    counts = np.zeros(NUM_LABELS, dtype=np.float32)
    for s in hf_split:
        counts[s["label"]] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_LABELS   # normalise
    print(f"Class weights: { {LABEL_MAP[i]: f'{weights[i]:.3f}' for i in range(NUM_LABELS)} }")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ─── TRAINING ────────────────────────────────────────────────────────────────

def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", zero_division=0)


def train(train_split, val_split):
    print("\nBuilding datasets (evidence ranking happens here)…")
    train_ds = PubHealthDataset(train_split)
    val_ds   = PubHealthDataset(val_split)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_weights = compute_class_weights(train_split)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1  = 0.0
    best_ckpt    = "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss   = criterion(logits, labels) / GRAD_ACCUM
            loss.backward()
            train_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["label"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_f1 = macro_f1(all_preds, all_labels)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={avg_train_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best saved  (val_f1={best_val_f1:.4f})")

    print(f"\nBest validation Macro-F1: {best_val_f1:.4f}")
    return model, best_ckpt

# ─── EVALUATION ──────────────────────────────────────────────────────────────

def evaluate(model_or_ckpt, test_split):
    if isinstance(model_or_ckpt, str):
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_or_ckpt, map_location=DEVICE))

    test_ds     = PubHealthDataset(test_split)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    print("\n" + "="*55)
    print("TEST RESULTS")
    print("="*55)
    print(classification_report(
        all_labels, all_preds,
        target_names=[LABEL_MAP[i] for i in range(NUM_LABELS)],
        zero_division=0,
    ))
    test_f1 = macro_f1(all_preds, all_labels)
    print(f"Test Macro-F1: {test_f1:.4f}")
    return test_f1

# ─── ABLATION ────────────────────────────────────────────────────────────────

def run_ablation(train_split, val_split, test_split):
    """
    Three input variants (mirrors the ablation study in the slides):
      1. Claim only
      2. Claim + explanation
      3. Claim + explanation + top-k evidence  ← proposed solution
    """
    results = {}

    configs = [
        ("claim_only",          lambda s: (s.get("claim",""), "")),
        ("claim+explanation",   lambda s: (s.get("claim",""), s.get("explanation",""))),
        ("full_topk_evidence",  lambda s: build_input(
            s.get("claim",""), s.get("explanation",""), s.get("main_text","")
        )),
    ]

    for name, input_fn in configs:
        print(f"\n{'─'*50}\nABLATION: {name}\n{'─'*50}")

        class AblationDataset(Dataset):
            def __init__(self, split):
                self.data = []
                for s in split:
                    parts = input_fn(s)
                    enc = tokenizer(
                        parts[0], parts[1] if parts[1] else None,
                        max_length=MAX_TOKENS, truncation=True,
                        padding="max_length", return_tensors="pt",
                    )
                    self.data.append({
                        "input_ids":      enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "label":          torch.tensor(s["label"], dtype=torch.long),
                    })
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return self.data[i]

        tr_ds = AblationDataset(train_split)
        va_ds = AblationDataset(val_split)
        te_ds = AblationDataset(test_split)

        tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_ldr = DataLoader(va_ds, batch_size=BATCH_SIZE)
        te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE)

        cw    = compute_class_weights(train_split)
        crit  = nn.CrossEntropyLoss(weight=cw)
        mdl   = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        opt   = torch.optim.AdamW(mdl.parameters(), lr=LR)
        total = (len(tr_ldr) // GRAD_ACCUM) * 3      # 3 epochs for ablation
        sch   = get_linear_schedule_with_warmup(opt, int(total*0.1), total)

        best_f1 = 0.0
        for ep in range(3):
            mdl.train(); opt.zero_grad()
            for step, b in enumerate(tr_ldr):
                ids  = b["input_ids"].to(DEVICE)
                mask = b["attention_mask"].to(DEVICE)
                lbl  = b["label"].to(DEVICE)
                loss = crit(mdl(input_ids=ids, attention_mask=mask).logits, lbl) / GRAD_ACCUM
                loss.backward()
                if (step+1) % GRAD_ACCUM == 0:
                    opt.step(); sch.step(); opt.zero_grad()

            mdl.eval(); p_all, l_all = [], []
            with torch.no_grad():
                for b in va_ldr:
                    ids  = b["input_ids"].to(DEVICE)
                    mask = b["attention_mask"].to(DEVICE)
                    p_all.extend(mdl(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu().numpy())
                    l_all.extend(b["label"].numpy())
            f1 = macro_f1(p_all, l_all)
            print(f"  ep {ep+1}  val_f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in mdl.state_dict().items()}

        # test
        mdl.load_state_dict(best_state)
        mdl.eval(); p_all, l_all = [], []
        with torch.no_grad():
            for b in te_ldr:
                ids  = b["input_ids"].to(DEVICE)
                mask = b["attention_mask"].to(DEVICE)
                p_all.extend(mdl(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu().numpy())
                l_all.extend(b["label"].numpy())
        test_f1 = macro_f1(p_all, l_all)
        results[name] = {"val_best": best_f1, "test_macro_f1": test_f1}
        print(f"  → test Macro-F1: {test_f1:.4f}")

    print("\n" + "="*55)
    print("ABLATION SUMMARY")
    print("="*55)
    for cfg, r in results.items():
        print(f"  {cfg:<30}  test_f1={r['test_macro_f1']:.4f}")
    return results


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_split, val_split, test_split = load_pubhealth()

    print(f"\nDataset sizes: train={len(train_split)}  val={len(val_split)}  test={len(test_split)}")

    # Full training run
    model, ckpt = train(train_split, val_split)
    evaluate(ckpt, test_split)

    # Ablation study
    run_ablation(train_split, val_split, test_split)