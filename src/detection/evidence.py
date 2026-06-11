# -*- coding: utf-8 -*-
"""
Evidence sentence selection from the article body

The EvidenceRanker scores every sentence in main_text by TF-IDF cosine similarity
to the claim and then selects sentences in one of two modes: a fixed top-k count,
or a dynamic mode that greedily fills the remaining RoBERTa token budget after the
claim and explanation are accounted for, maximising evidence coverage without
overflowing the 512-token context window

TF-IDF ranking is used instead of a neural re-ranker because it needs no extra GPU
memory and runs in seconds over the whole corpus, keeping evidence selection cheap
and deterministic
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from detection.config import MAX_TOKENS

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def sent_tokenize(text: str) -> list:
    """
    Split text into sentences with a lightweight regex, no NLTK download required
    """
    if not text:
        return []
    parts = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


class EvidenceRanker:
    """
    Rank and select evidence sentences relative to a claim

    A single ranker supports both selection strategies used across the experiments
    so the seven original scripts collapse to one configurable component:
      - mode "topk": return the top_k highest-similarity sentences in original order
      - mode "dynamic": greedily add highest-similarity sentences until the next one
        would exceed the token budget left after the claim and explanation

    IMPORTANT: dynamic mode needs the RoBERTa tokenizer to measure the budget; the
    tokenizer is resolved lazily, or an explicit one can be injected for testing
    """

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from detection.tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    def rank(self, claim: str, explanation: str, main_text: str,
             mode: str = "dynamic", top_k: int = None, max_tokens: int = MAX_TOKENS) -> str:
        """
        Select evidence sentences from main_text for a single sample

        Args:
            claim (str): The claim text used as the ranking query
            explanation (str): The explanation paragraph, only consumed by the budget
                calculation in dynamic mode
            main_text (str): The article body to extract sentences from
            mode (str): Either "topk" or "dynamic"
            top_k (int): Number of sentences for top-k mode, ignored in dynamic mode
            max_tokens (int): Total context window used by the dynamic budget
        Returns:
            str: The selected sentences joined in their original order, or an empty
                string when no evidence is available or fits
        """
        if not main_text or not main_text.strip():
            return ""

        sentences = sent_tokenize(main_text)
        if not sentences:
            return ""

        ranked_idx = self._rank_indices(claim, sentences)

        if mode == "topk":
            return self._select_topk(sentences, ranked_idx, top_k)
        if mode == "dynamic":
            return self._select_dynamic(claim, explanation, sentences, ranked_idx, max_tokens)
        raise ValueError(f"Unknown evidence mode {mode}")

    def _rank_indices(self, claim: str, sentences: list) -> list:
        """
        Return sentence indices ordered by descending TF-IDF cosine to the claim
        """
        if len(sentences) == 1:
            return [0]

        corpus = [claim] + sentences           # Shared vocabulary across claim and sentences
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=10_000)
            vecs = tfidf.fit_transform(corpus)
        except ValueError:
            return list(range(len(sentences)))  # Degenerate corpus falls back to document order

        scores = cosine_similarity(vecs[0], vecs[1:]).flatten()
        return list(np.argsort(scores)[::-1])

    def _select_topk(self, sentences: list, ranked_idx: list, top_k: int) -> str:
        """
        Keep the top_k highest-scoring sentences, restored to their original order
        """
        if top_k is None:
            raise ValueError("top_k must be provided for topk mode")
        if len(sentences) <= top_k:
            return " ".join(sentences)         # Nothing to rank, keep everything
        chosen = sorted(ranked_idx[:top_k])
        return " ".join(sentences[i] for i in chosen)

    def _select_dynamic(self, claim: str, explanation: str, sentences: list,
                        ranked_idx: list, max_tokens: int) -> str:
        """
        Greedily add sentences highest-score-first until the token budget is full

        The input is encoded as <s> claim </s></s> explanation </s></s> evidence </s>,
        so the claim and explanation are tokenised once to find how many tokens remain
        for evidence; each candidate costs its own length plus one separating token
        """
        tok = self.tokenizer
        base = tok(
            claim.strip(),
            explanation.strip() if explanation else None,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens,
        )
        budget = max_tokens - len(base["input_ids"])
        if budget <= 2:
            return ""

        selected, tokens_used = [], 0
        for idx in ranked_idx:
            sent_tokens = len(tok(sentences[idx], add_special_tokens=False)["input_ids"])
            if tokens_used + sent_tokens + 1 > budget:
                continue                       # Skip rather than break so shorter later sentences can still fit
            selected.append(idx)
            tokens_used += sent_tokens + 1

        if not selected:
            return ""
        selected.sort()
        return " ".join(sentences[i] for i in selected)
