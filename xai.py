"""
Explainability (XAI) — Attention-based token heatmaps
======================================================
Reproduces the "Clinical Heatmap Output" from the project slides.

Usage:
    from xai import explain_prediction
    explain_prediction(model, tokenizer, claim, explanation, main_text)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

import re

try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("Captum not installed — falling back to raw attention weights.")

LABEL_MAP = {0: "true", 1: "false", 2: "unproven", 3: "mixture"}
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── HELPER: raw last-layer attention ────────────────────────────────────────

def _attention_attribution(model, inputs):
    """
    Average attention weights from the last transformer layer, head-averaged.
    Returns a (seq_len,) numpy array normalised to [0, 1].
    """
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )
    # output.attentions: tuple of (batch, heads, seq, seq) per layer
    last_layer_attn = output.attentions[-1]          # (1, heads, seq, seq)
    # CLS token attends to all others: take row 0 of the attention matrix
    cls_attn = last_layer_attn[0, :, 0, :]            # (heads, seq)
    cls_attn = cls_attn.mean(dim=0).cpu().numpy()     # (seq,)  head-averaged
    # Normalise to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-9)
    return cls_attn, output.logits.argmax(-1).item()


# ─── HELPER: Captum integrated gradients ─────────────────────────────────────

def _captum_attribution(model, inputs, predicted_class):
    """
    Layer Integrated Gradients on the embedding layer.
    Returns a (seq_len,) numpy array of L2-normed attribution scores.
    """
    def forward_fn(input_ids):
        return model(input_ids=input_ids,
                     attention_mask=inputs["attention_mask"]).logits

    lig = LayerIntegratedGradients(forward_fn, model.roberta.embeddings.word_embeddings)
    baseline = torch.zeros_like(inputs["input_ids"])

    attrs, _ = lig.attribute(
        inputs=inputs["input_ids"],
        baselines=baseline,
        target=predicted_class,
        n_steps=50,
        return_convergence_delta=True,
    )
    # attrs: (1, seq, hidden) → L2 norm across hidden dim → (seq,)
    scores = attrs[0].norm(dim=-1).cpu().detach().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def explain_prediction(
    model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizerFast,
    claim: str,
    explanation: str = "",
    evidence: str = "",
    save_path: str = "heatmap.png",
    use_captum: bool = True,
) -> dict:
    """
    Run the model on one sample and produce a token-level attribution heatmap.

    Returns
    -------
    dict with keys: predicted_label, confidence, token_scores, figure_path
    """
    model.eval()
    model.to(DEVICE)

    # Tokenise exactly as in training
    part_b = f"{explanation.strip()} {tokenizer.sep_token}{tokenizer.sep_token} {evidence.strip()}"
    enc = tokenizer(
        claim.strip(),
        part_b if part_b.strip() else None,
        max_length=512,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    if use_captum and HAS_CAPTUM:
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_class = logits.argmax(-1).item()
        scores = _captum_attribution(model, inputs, pred_class)
        method = "Integrated Gradients (Captum)"
    else:
        scores, pred_class = _attention_attribution(model, inputs)
        method = "Attention weights (last layer)"

    # Decode tokens (skip padding)
    input_ids = enc["input_ids"][0].tolist()
    attn_mask = enc["attention_mask"][0].tolist()
    tokens    = tokenizer.convert_ids_to_tokens(input_ids)
    active    = [(t, s) for t, s, m in zip(tokens, scores, attn_mask) if m == 1]
    # Drop special tokens for display
    active    = [(t, s) for t, s in active if t not in ("<s>", "</s>", "<pad>")]

    # Confidence
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

    result = {
        "predicted_label": LABEL_MAP[pred_class],
        "confidence":      float(probs[pred_class]),
        "class_probs":     {LABEL_MAP[i]: float(probs[i]) for i in range(4)},
        "token_scores":    active,
        "method":          method,
    }

    # ── Plot ───────────────────────────────────────────────────────────────
    _plot_heatmap(active, result, claim, save_path)
    result["figure_path"] = save_path
    return result


def _plot_heatmap(active_tokens, result, claim, save_path):
    tokens = [t for t, _ in active_tokens]
    scores = np.array([s for _, s in active_tokens])

    # Wrap to ≤20 tokens per row
    ROW = 20
    rows  = [tokens[i:i+ROW] for i in range(0, len(tokens), ROW)]
    s_rows = [scores[i:i+ROW] for i in range(0, len(scores), ROW)]

    cmap   = mcolors.LinearSegmentedColormap.from_list(
        "clinical", ["#f5f5f5", "#d4a5a5", "#7b2d2d"]
    )

    fig_h = max(3, len(rows) * 1.2 + 3)
    fig, axes = plt.subplots(len(rows), 1, figsize=(min(20, ROW * 0.9), fig_h))
    if len(rows) == 1:
        axes = [axes]

    for ax, row_toks, row_scores in zip(axes, rows, s_rows):
        for j, (tok, sc) in enumerate(zip(row_toks, row_scores)):
            colour = cmap(sc)
            ax.add_patch(plt.Rectangle((j, 0), 0.92, 0.85, color=colour, zorder=1))
            ax.text(j + 0.46, 0.425, tok.replace("Ġ", ""),
                    ha="center", va="center", fontsize=9,
                    color="white" if sc > 0.6 else "#333333", zorder=2)
        ax.set_xlim(0, ROW)
        ax.set_ylim(0, 1)
        ax.axis("off")

    label   = result["predicted_label"].upper()
    conf    = result["confidence"]
    method  = result["method"]
    plt.suptitle(
        f'Prediction: {label}  ({conf:.1%} confidence)\n'
        f'Claim: "{claim[:90]}{"…" if len(claim)>90 else ""}"\n'
        f'Attribution: {method}',
        fontsize=10, y=1.01, ha="left", x=0.02,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.01, pad=0.01,
                 label="Attribution score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {save_path}")
