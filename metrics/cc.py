from __future__ import annotations


def consistency_check(pred_label: str, qf: float, tgi: int) -> int:
    if pred_label == "hateful":
        if float(qf) >= 0.45 and tgi == 1:
            return 1
        else:
            return 0
    elif pred_label == "non-hateful":
        if float(qf) < 0.45 or tgi == 1:
            return 0
        else:
            return 1
    else:
        return 0
