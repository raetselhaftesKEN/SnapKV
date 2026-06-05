"""
Utilities for using slow LLaMA/Mistral tokenizers in long-running SnapKV/KIVI evaluation.

This file does not change model behavior. It only works around occasional corruption of
PreTrainedTokenizer.tokens_trie in slow-tokenizer mode, which can raise errors like:
    TypeError: '>=' not supported between instances of 'Trie' and 'int'
inside transformers/tokenization_utils.py::Trie.split.
"""
from typing import Any, Dict


def rebuild_slow_tokenizer_trie(tokenizer) -> None:
    """Rebuild tokenizer.tokens_trie for slow HuggingFace tokenizers.

    Slow tokenizers use tokens_trie to split no-split/special tokens before calling the
    model-specific tokenizer. In some old transformers environments this trie can become
    corrupted during long loops. Rebuilding it is cheap and safe.
    """
    # Newer slow tokenizers usually expose this helper.
    if hasattr(tokenizer, "_update_trie"):
        tokenizer._update_trie()
        return

    # Older PreTrainedTokenizer exposes _create_trie(unique_no_split_tokens).
    if hasattr(tokenizer, "_create_trie"):
        unique_no_split_tokens = getattr(tokenizer, "unique_no_split_tokens", None)
        if unique_no_split_tokens is None:
            unique_no_split_tokens = getattr(tokenizer, "all_special_tokens", [])
        tokenizer.tokens_trie = tokenizer._create_trie(unique_no_split_tokens)
        return

    # Very old fallback.
    from transformers.tokenization_utils import Trie

    trie = Trie()
    candidates = []
    for attr in ("unique_no_split_tokens", "all_special_tokens", "additional_special_tokens"):
        vals = getattr(tokenizer, attr, None)
        if vals:
            candidates.extend(list(vals))
    for tok in sorted(set(candidates), key=len, reverse=True):
        if isinstance(tok, str) and tok:
            trie.add(tok)
    tokenizer.tokens_trie = trie


def safe_tokenize(tokenizer, text: str, device=None, retry: int = 2, **kwargs) -> Dict[str, Any]:
    """Call tokenizer(text, **kwargs), rebuilding slow-tokenizer trie on Trie-related failure.

    Usage:
        inputs = safe_tokenize(tokenizer, prompt, device=device, truncation=False, return_tensors="pt")
    """
    if not isinstance(text, str):
        text = str(text)

    last_err = None
    for attempt in range(retry + 1):
        try:
            out = tokenizer(text, **kwargs)
            return out.to(device) if device is not None and hasattr(out, "to") else out
        except TypeError as e:
            msg = str(e)
            last_err = e
            if "Trie" not in msg and "tokens_trie" not in msg and ">=" not in msg:
                raise
            rebuild_slow_tokenizer_trie(tokenizer)
        except AttributeError as e:
            msg = str(e)
            last_err = e
            if "Trie" not in msg and "tokens_trie" not in msg:
                raise
            rebuild_slow_tokenizer_trie(tokenizer)
    raise last_err
