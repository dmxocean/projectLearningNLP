# -*- coding: utf-8 -*-
"""
Lazy RoBERTa tokenizer accessor

The tokenizer is loaded on first use rather than at import time so that modules
depending on it can be imported in offline contexts (for example unit tests that
exercise pure logic) without triggering a model download
"""

from detection.config import MODEL_NAME

_TOKENIZER = None


def get_tokenizer():
    """
    Return the shared RobertaTokenizerFast instance, loading it once on demand
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import RobertaTokenizerFast
        _TOKENIZER = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    return _TOKENIZER
