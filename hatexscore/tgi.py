from __future__ import annotations

from typing import List
import re

import jieba
from konlpy.tag import Okt


def lemmatize(language: str, text: str, nlp=None, okt: Okt | None = None) -> List[str]:
    if language == "zh":
        text = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", text.lower())
        tokens = list(jieba.cut(text))
        tokens = [tok for tok in tokens if tok.strip()]
        return tokens
    elif language == "kr":
        if okt is not None:
            text = re.sub(r"[^\w\s\u3131-\u318E\uAC00-\uD7A3]+", " ", text.lower())
            tokens = okt.morphs(text, stem=True)
            return [tok for tok in tokens if tok.strip()]
        text = re.sub(r"[^\w\s\uac00-\ud7a3]+", " ", text.lower())
        return [tok for tok in text.split() if tok.strip()]
    else:
        doc = nlp(text.lower())
        return [token.lemma_ for token in doc if token.lemma_.strip()]


def tgi(reasoning: str, quotes: List[str], target_group_list: List[str], language: str, nlp=None, okt: Okt | None = None) -> int:
    tokens = lemmatize(language, reasoning, nlp=nlp, okt=okt)
    ngram_set = set()
    max_n = 3
    for n in range(1, max_n + 1):
        ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_set.update(ngrams)

    target_set = set()
    for tg in target_group_list:
        tg_tokens = lemmatize(language, tg, nlp=nlp, okt=okt)
        for n in range(1, len(tg_tokens) + 1):
            tg_ngram = " ".join(tg_tokens[:n])
            target_set.add(tg_ngram)

    if ngram_set & target_set:
        return 1
    return 0
