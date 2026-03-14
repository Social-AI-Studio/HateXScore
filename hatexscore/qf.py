from __future__ import annotations

from typing import List
import json
import re

import jieba
import numpy as np
from fuzzysearch import find_near_matches
from konlpy.tag import Okt
from openai import OpenAI


def mask_rationales(text: str, quotes: List[str]) -> str:
    masked = text
    for q in sorted(quotes, key=len, reverse=True):
        masked, n = re.subn(re.escape(q), "[MASK]", masked, flags=re.I)
        if n > 0:
            continue
        if find_near_matches is None:
            continue
        max_dist = min(5, int(len(q) * 0.15) or 1)
        matches = find_near_matches(q, masked, max_l_dist=max_dist)
        for m in reversed(matches):
            masked = masked[: m.start] + "[MASK]" + masked[m.end :]
    masked = re.sub(r"\s{2,}", " ", masked).strip()
    return masked


def norm(language: str, t: str, okt: Okt | None = None) -> List[str]:
    t = t.lower()
    if language == "zh":
        t = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", t)
        tokens = list(jieba.cut(t))
        tokens = [tok for tok in tokens if tok.strip()]
        return tokens
    elif language == "kr":
        t = re.sub(r"[^\w\s\uac00-\ud7a3]+", " ", t)
        if okt is not None:
            tokens = okt.morphs(t)
        else:
            tokens = t.split()
        return [tok for tok in tokens if tok.strip()]
    else:
        return re.sub(r"[^\w\s]+", " ", t).split()


def quoted_phrases(language: str, text: str, reasoning: str, okt: Okt | None = None) -> List[str]:
    text_tokens = norm(language, text, okt)
    reasoning_tokens = norm(language, reasoning, okt)
    overlap = [tok for tok in text_tokens if tok in reasoning_tokens]
    return list(set(overlap))


def overlap_quoted_phrases(language: str, text: str, analysis: str, okt: Okt | None = None) -> List[str]:
    text_tokens = " ".join(norm(language, text, okt))
    hits = []
    for p in quoted_phrases(language, text, analysis, okt):
        if " ".join(norm(language, p, okt)) in text_tokens:
            hits.append(p)
    return hits


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[0])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")
    return pred_answer


def predict_proba(text: str, args) -> float:
    print(text)
    system_prompt = """
            Please estimate how likely the following text is hateful. Output a single number between 0 and 1 (where 1 means highly likely to be hateful and 0 means not likely at all). Only provide the number. Do not include any explanation or additional text.
            """

    user_prompt = f"Text: {text}\n\n"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="...",
    )

    if args.model == "gpt":
        model_name = "openai/gpt-5.4"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    elif args.model == "mistral7b":
        model_name = "mistralai/mistral-7b-instruct"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    elif args.model == "gemma2b":
        model_name = "google/gemma-3-4b-it"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    elif args.model == "gemma9b":
        model_name = "google/gemma-3-12b-it"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    elif args.model == "gemma27b":
        model_name = "google/gemma-2-27b-it"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        print("checking::::::!!!!!!", response.choices[0])
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    elif args.model == "qwen7b":
        model_name = "qwen/qwen-2.5-7b-instruct"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    elif args.model == "llama8b":
        model_name = "meta-llama/llama-3.1-8b-instruct"
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        if response.choices[0]:
            output = response.choices[0].message.content
            try:
                output = extract_answer_number(output)
                return float(output)
            except ValueError:
                print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
                return 0.0
        else:
            print(f"Invalid output for float conversion: '{response.choices[0].message.content}'")
            return 0.0

    return 0.0


def quotation_faithfulness(
    text: str,
    reasoning: str,
    prediction: str,
    language: str,
    args,
    okt: Okt | None = None,
) -> float:
    quotes = quoted_phrases(language, text, reasoning, okt)
    print("quotations: ", quotes)
    if quotes == [text] or len(quotes) == 0:
        return 0
    if text in quotes and len(quotes) > 1:
        quotes = [q for q in quotes if q != text]

    masked_text = mask_rationales(text, quotes)
    p_orig = predict_proba(text, args)
    p_mask = predict_proba(masked_text, args)
    print(text, masked_text)
    print(p_orig, p_mask)
    if len(quotes) > 0 and prediction == "hateful":
        score = abs(p_orig - p_mask)
    elif len(quotes) > 0 and prediction == "non-hateful":
        score = 1 - abs(p_orig - p_mask)
    else:
        score = 0
    if np.isinf(score):
        score = 0
    if score > 1:
        score = 1

    return score
