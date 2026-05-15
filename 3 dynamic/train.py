# -*- coding: utf-8 -*-
"""
Health Fake News Detection - RoBERTa + Evidence Ranking
========================================================
Solves RoBERTa's 512-token limit by replacing blind truncation of main_text
with DYNAMIC top-k sentence selection ranked by semantic similarity to the claim.

Instead of a fixed TOP_K, the ranker greedily fills the remaining token budget:
it adds sentences (highest-similarity first) until the next sentence would push
the total past MAX_TOKENS.  This maximises evidence coverage without ever
overflowing the context window.

Pipeline:
  claim + explanation + dynamic_evidence  ->  RoBERTa-base  ->  4-class label
                                                            (true/false/mixture/unproven)
"""

import os, random, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # suppress tokenizer fork warning
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # no unauthenticated HF Hub noise

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
# NOTE: load_dataset removed — data is loaded from local TSVs, no Hub calls needed
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- REPRODUCIBILITY ---------------------------------------------------------

SEED = 42

def set_seed(seed: int = SEED):
    """Seed Python, NumPy, PyTorch (CPU + all GPUs) for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU
    # Makes cuDNN deterministic at a small speed cost; remove if speed is critical
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)

set_seed(SEED)

def sent_tokenize(text: str) -> list[str]:
    """
    Lightweight regex sentence splitter -- no NLTK download required.
    Splits on '.', '!', '?' followed by whitespace + capital letter.
    Falls back gracefully on edge cases.
    """
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"])', text.strip())
    return [s.strip() for s in parts if s.strip()]

# --- CONFIG ------------------------------------------------------------------

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME   = "roberta-base"
MAX_TOKENS   = 512
BATCH_SIZE   = 16
GRAD_ACCUM   = 2        # effective batch = 32
EPOCHS       = 5
LR           = 2e-5
WARMUP_FRAC  = 0.10

LABEL_MAP    = {0: "true", 1: "false", 2: "unproven", 3: "mixture"}
NUM_LABELS   = 4

print(f"Device: {DEVICE}  |  Seed: {SEED}")

# --- TOKENIZER (loaded early so EvidenceRanker can use it) -------------------

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

# --- EVIDENCE RANKER ---------------------------------------------------------

class EvidenceRanker:
    """
    Ranks sentences in main_text by TF-IDF cosine similarity to the claim,
    then greedily fills the remaining token budget instead of using a fixed top-k.

    How the budget is computed
    --------------------------
    The full input is:  <s> claim </s></s> explanation </s></s> evidence </s>
    RoBERTa special tokens cost 4 tokens (1 + 2 + 1).
    We tokenize (claim + explanation) once, subtract from MAX_TOKENS, and use
    whatever is left for evidence sentences - adding them highest-score-first
    until the next sentence would overflow.

    Why TF-IDF and not a neural embedder?
    - Zero extra GPU memory at training time
    - Fast enough for 11 k samples in seconds
    - Kotonya & Toni (2020) show it is competitive with neural ranking
      for this specific task (arXiv:2010.09926)
    """

    # Special-token overhead: <s> ... </s></s> ... </s></s> ... </s>
    SPECIAL_TOKEN_OVERHEAD = 4   # <s>, </s>, </s><s> x2 separators ? net 4 extra

    def rank(self, claim: str, explanation: str, main_text: str) -> str:
        """
        Return as many ranked sentences from main_text as fit inside the
        remaining token budget after encoding claim + explanation.
        """
        if not main_text or not main_text.strip():
            return ""

        sentences = sent_tokenize(main_text)
        if not sentences:
            return ""

        # ---- compute how many tokens are already spent -------------------
        # Encode claim + explanation, clamped to MAX_TOKENS so we never
        # produce a sequence the model can't handle (avoids the >512 warning).
        base_enc = tokenizer(
            claim.strip(),
            explanation.strip() if explanation else None,
            add_special_tokens=True,
            truncation=True,          # clamp to MAX_TOKENS — no >512 warnings
            max_length=MAX_TOKENS,
        )
        base_token_len = len(base_enc["input_ids"])

        # Budget left for evidence tokens (including its trailing </s>)
        budget = MAX_TOKENS - base_token_len
        if budget <= 2:               # claim+explanation already fills window
            return ""

        # ---- rank sentences by relevance to claim -------------------------
        if len(sentences) == 1:
            # Nothing to rank; just check it fits
            tok_len = len(tokenizer(sentences[0], add_special_tokens=False)["input_ids"])
            return sentences[0] if tok_len <= budget else ""

        corpus = [claim] + sentences
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=10_000)
            vecs  = tfidf.fit_transform(corpus)
        except ValueError:
            # Corpus too sparse (all stop-words, etc.) ? take first sentences
            vecs = None

        if vecs is not None:
            claim_vec = vecs[0]
            sent_vecs = vecs[1:]
            scores    = cosine_similarity(claim_vec, sent_vecs).flatten()
            ranked_idx = list(np.argsort(scores)[::-1])   # best first
        else:
            ranked_idx = list(range(len(sentences)))       # fallback: original order

        # ---- greedily fill budget -----------------------------------------
        selected_idx = []
        tokens_used  = 0
        for idx in ranked_idx:
            sent_tokens = len(
                tokenizer(sentences[idx], add_special_tokens=False)["input_ids"]
            )
            # +1 for the space separator between sentences
            if tokens_used + sent_tokens + 1 > budget:
                continue          # skip this sentence, try smaller ones
            selected_idx.append(idx)
            tokens_used += sent_tokens + 1

        if not selected_idx:
            return ""

        # Restore original document order so text reads naturally
        selected_idx.sort()
        return " ".join(sentences[i] for i in selected_idx)


ranker = EvidenceRanker()

# --- INPUT BUILDER -----------------------------------------------------------

def build_input(claim: str, explanation: str, main_text: str):
    """
    Build the (part_a, part_b) pair fed to RoBERTa.

    RobertaTokenizerFast with text_pair encodes as:
      <s> part_a </s></s> part_b </s>

    part_b = explanation + </s></s> + dynamic_evidence
    so the model sees two internal separation points.
    """
    evidence = ranker.rank(claim, explanation, main_text)

    part_a = claim.strip()
    sep    = f"{tokenizer.sep_token}{tokenizer.sep_token}"
    part_b = f"{explanation.strip()} {sep} {evidence.strip()}" if evidence else explanation.strip()
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

# --- DATASET -----------------------------------------------------------------

class PubHealthDataset(Dataset):
    def __init__(self, hf_split):
        self.samples = [tokenize_sample(s) for s in hf_split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


DATA_DIR = "/export/hhome/nlp203/NLP/data/pubhealth"

def load_pubhealth():
    """Load the three TSV splits directly from the local VM filesystem."""
    import pandas as pd
    from datasets import DatasetDict, Dataset as HFDataset

    label_map_str = {"true": 0, "false": 1, "unproven": 2, "mixture": 3}

    def read_tsv(path):
        print(f"  Reading {path}")
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip", encoding="utf-8")
        df = df.dropna(subset=["claim", "label"])
        df["label"] = df["label"].str.strip().str.lower().map(label_map_str)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        for col in ["explanation", "main_text"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("")
        return HFDataset.from_pandas(
            df[["claim", "explanation", "main_text", "label"]].reset_index(drop=True)
        )

    ds_dict = {
        "train":      read_tsv(f"{DATA_DIR}/train.tsv"),
        "validation": read_tsv(f"{DATA_DIR}/dev.tsv"),
        "test":       read_tsv(f"{DATA_DIR}/test.tsv"),
    }
    print(f"  Loaded -- train={len(ds_dict['train'])}  "
          f"val={len(ds_dict['validation'])}  test={len(ds_dict['test'])}")
    return ds_dict["train"], ds_dict["validation"], ds_dict["test"]

# --- CLASS WEIGHTS -----------------------------------------------------------

def compute_class_weights(hf_split) -> torch.Tensor:
    """Inverse-frequency weights so rare classes (unproven) get higher penalty."""
    counts = np.zeros(NUM_LABELS, dtype=np.float32)
    for s in hf_split:
        counts[s["label"]] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_LABELS
    print(f"Class weights: { {LABEL_MAP[i]: f'{weights[i]:.3f}' for i in range(NUM_LABELS)} }")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# --- TRAINING ----------------------------------------------------------------

def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", zero_division=0)


def train(train_split, val_split):
    set_seed(SEED)   # re-seed before model init so weights are deterministic

    print("\nBuilding datasets (dynamic evidence ranking happens here)...")
    train_ds = PubHealthDataset(train_split)
    val_ds   = PubHealthDataset(val_split)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, worker_init_fn=lambda wid: set_seed(SEED + wid)
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
    history      = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        # -- train ----------------------------------------------------------
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

        # -- validate -------------------------------------------------------
        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["label"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                val_loss_total += criterion(logits, labels).item()
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader)
        val_f1 = macro_f1(all_preds, all_labels)
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={avg_train_loss:.4f}"
              f"  val_loss={avg_val_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ? New best saved  (val_f1={best_val_f1:.4f})")

    print(f"\nBest validation Macro-F1: {best_val_f1:.4f}")
    _save_training_curves(history, PLOT_DIR)
    return model, best_ckpt

# --- EVALUATION --------------------------------------------------------------

def evaluate(model_or_ckpt, test_split):
    if isinstance(model_or_ckpt, str):
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_or_ckpt, map_location=DEVICE))
    else:
        model = model_or_ckpt

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

# --- PLOTS -------------------------------------------------------------------

def _save_training_curves(history: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], marker="o", color="#4C72B0",
             linewidth=2, markersize=5, label="Train loss")
    ax1.plot(epochs, history["val_loss"], marker="s", color="#DD8452",
             linewidth=2, markersize=5, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Train vs Validation Loss", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, history["val_f1"], marker="o", color="#55A868", linewidth=2, markersize=5)
    best_ep = epochs[history["val_f1"].index(max(history["val_f1"]))]
    best_f1 = max(history["val_f1"])
    ax2.axvline(best_ep, linestyle="--", color="#C44E52", alpha=0.7,
                label=f"Best epoch {best_ep}  (F1={best_f1:.3f})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation Macro-F1", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Full Model - Training Curves (claim + explanation + dynamic evidence)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def save_plots(results: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    os.makedirs(out_dir, exist_ok=True)
    CLASS_NAMES = [LABEL_MAP[i] for i in range(NUM_LABELS)]
    COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    EXP_COLORS  = ["#6baed6", "#fd8d3c", "#74c476"]

    for name, r in results.items():
        y_true = np.array(r["y_true"])
        y_pred = np.array(r["y_pred"])

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix\n{name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"confusion_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {path}")

        # Per-class F1 bar chart
        from sklearn.metrics import f1_score as sk_f1
        per_class_f1 = sk_f1(y_true, y_pred, average=None,
                              labels=list(range(NUM_LABELS)), zero_division=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(CLASS_NAMES, per_class_f1, color=COLORS,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"Per-Class F1  |  Macro-F1 = {r['test_macro_f1']:.3f}\n{name}",
                     fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
        for bar, val in zip(bars, per_class_f1):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(out_dir, f"f1_perclass_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {path}")

    # Cross-experiment Macro-F1 comparison
    names        = list(results.keys())
    labels       = ["Claim\nOnly", "Claim +\nExplanation", "Claim + Explanation\n+ Dynamic Evidence"]
    macro_scores = [results[n]["test_macro_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, macro_scores, color=EXP_COLORS,
                  edgecolor="white", linewidth=0.8, zorder=3, width=0.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Macro-F1 (Test)", fontsize=12)
    ax.set_title("Ablation Study - Macro-F1 by Input Configuration",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    for bar, val in zip(bars, macro_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "macro_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


PLOT_DIR = "/export/hhome/nlp203/NLP/detection"

# --- ABLATION ----------------------------------------------------------------

def run_ablation(train_split, val_split, test_split):
    """
    Three input variants:
      1. Claim only
      2. Claim + explanation
      3. Claim + explanation + dynamic evidence (budget-filling ranker)
    """
    results = {}

    configs = [
        ("claim_only",         lambda s: (s.get("claim",""), "")),
        ("claim+explanation",  lambda s: (s.get("claim",""), s.get("explanation",""))),
        ("full_dynamic_evidence", lambda s: build_input(
            s.get("claim",""), s.get("explanation",""), s.get("main_text","")
        )),
    ]

    for name, input_fn in configs:
        set_seed(SEED)   # re-seed for each ablation run
        print(f"\n{'-'*50}\nABLATION: {name}\n{'-'*50}")

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

        tr_ldr = DataLoader(
            tr_ds, batch_size=BATCH_SIZE, shuffle=True,
            worker_init_fn=lambda wid: set_seed(SEED + wid)
        )
        va_ldr = DataLoader(va_ds, batch_size=BATCH_SIZE)
        te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE)

        cw   = compute_class_weights(train_split)
        crit = nn.CrossEntropyLoss(weight=cw)
        mdl  = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        opt   = torch.optim.AdamW(mdl.parameters(), lr=LR)
        total = (len(tr_ldr) // GRAD_ACCUM) * 3
        sch   = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

        best_f1    = 0.0
        best_state = None
        for ep in range(3):
            mdl.train(); opt.zero_grad()
            for step, b in enumerate(tr_ldr):
                ids  = b["input_ids"].to(DEVICE)
                mask = b["attention_mask"].to(DEVICE)
                lbl  = b["label"].to(DEVICE)
                loss = crit(mdl(input_ids=ids, attention_mask=mask).logits, lbl) / GRAD_ACCUM
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
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
                best_f1    = f1
                best_state = {k: v.clone() for k, v in mdl.state_dict().items()}

        mdl.load_state_dict(best_state)
        mdl.eval(); p_all, l_all = [], []
        with torch.no_grad():
            for b in te_ldr:
                ids  = b["input_ids"].to(DEVICE)
                mask = b["attention_mask"].to(DEVICE)
                p_all.extend(mdl(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu().numpy())
                l_all.extend(b["label"].numpy())

        test_f1 = macro_f1(p_all, l_all)
        results[name] = {
            "val_best":       best_f1,
            "test_macro_f1":  test_f1,
            "y_true":         l_all,
            "y_pred":         p_all,
        }
        print(f"  -> test Macro-F1: {test_f1:.4f}")

    print("\n" + "=" * 55)
    print("ABLATION SUMMARY")
    print("=" * 55)
    for cfg, r in results.items():
        print(f"  {cfg:<35}  test_f1={r['test_macro_f1']:.4f}")

    print(f"\nSaving plots to {PLOT_DIR} ...")
    save_plots(results, PLOT_DIR)
    return results


# --- ENTRY POINT -------------------------------------------------------------

if __name__ == "__main__":
    train_split, val_split, test_split = load_pubhealth()
    print(f"\nDataset sizes: train={len(train_split)}  val={len(val_split)}  test={len(test_split)}")

    # Full training run
    model, ckpt = train(train_split, val_split)
    evaluate(ckpt, test_split)

    # Ablation study
    run_ablation(train_split, val_split, test_split)