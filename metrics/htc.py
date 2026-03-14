from __future__ import annotations

import re


def extract_answer_yes(sentence: str) -> str:
    if sentence is not None:
        sentence_ = sentence.strip().lower()
    else:
        sentence_ = ""

    pred_answers = re.findall(
        r"\bnon-hateful\b|\bhateful\b|\byes\b|\bno\b|\bhate speech\b|\b是，\b|\b不是，\b|\b비증오적인\b|\b증오적인\b|\b아니요\b|\b예\b",
        sentence_,
    )
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[-1]
    else:
        return ""


def htc(reasoning: str, pred_label: str) -> float:
    reasoning_lower = reasoning.lower()
    pred = extract_answer_yes(reasoning_lower)
    if not pred:
        return float(0)
    else:
        return float(1)
