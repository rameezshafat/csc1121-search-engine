from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

# Unicode letters/digits, excluding underscore.
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
}


@dataclass(frozen=True)
class PreprocessConfig:
    min_token_len: int = 2
    use_stopwords: bool = False
    use_stemming: bool = False


def normalize_text(text: str) -> str:
    return text.lower()


def _load_stopwords() -> Set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return FALLBACK_STOPWORDS


def _build_stemmer():
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer()
    except Exception:
        return None


class TextAnalyzer:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.stopwords: Set[str] = _load_stopwords() if cfg.use_stopwords else set()
        self.stemmer = _build_stemmer() if cfg.use_stemming else None

    def tokenize(self, text: str, min_len: int | None = None) -> List[str]:
        min_length = self.cfg.min_token_len if min_len is None else min_len
        tokens = TOKEN_RE.findall(normalize_text(text))
        tokens = [tok for tok in tokens if len(tok) >= min_length]

        if self.stopwords:
            tokens = [tok for tok in tokens if tok not in self.stopwords]

        if self.stemmer is not None:
            tokens = [self.stemmer.stem(tok) for tok in tokens]

        return tokens


def tokenize(text: str, min_len: int = 2) -> List[str]:
    # Backward-compatible default tokenizer (no stopword/stemming)
    tokens = TOKEN_RE.findall(normalize_text(text))
    return [tok for tok in tokens if len(tok) >= min_len]
