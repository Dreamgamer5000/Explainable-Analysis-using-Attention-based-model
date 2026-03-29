import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import pipeline

try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

try:
    import shap
except Exception:
    shap = None

analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None,
)

_ALLOWED_EXPLAIN_METHODS = {"lime", "shap"}

_DOMAIN_STOPWORDS = {"film", "movie", "watch"}
_STOPWORDS = set(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS

_ASPECT_KEYWORDS = {
    'Acting': [
        'acting', 'actor', 'actress', 'performance', 'performances',
        'cast', 'casting', 'portrayal', 'role', 'plays', 'played',
    ],
    'Directing': [
        'director', 'direction', 'directed', 'directing',
        'filmmaker', 'vision', 'pacing', 'pace', 'helm', 'helmed',
        'directly', 'directorial',
    ],
    'Plot': [
        'plot', 'story', 'script', 'writing', 'narrative', 'storyline',
        'ending', 'twist', 'screenplay', 'dialogue', 'scene', 'scenes',
        'premise',
    ],
    'Cinematography': [
        'cinematography', 'visual', 'visuals', 'photography',
        'lighting', 'shot', 'shots', 'camera', 'color', 'colour',
        'beautiful', 'stunning',
    ],
    'Soundtrack': [
        'soundtrack', 'music', 'score', 'song', 'songs', 'audio',
        'sound', 'musical', 'composer', 'track', 'tracks',
    ],
}


def _contains_keyword(text, keywords):
    lower = text.lower()
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            return True
    return False


def _split_clauses(sentence):
    return [
        part.strip()
        for part in re.split(r"\b(?:but|however|although|though|yet)\b|[,;:]", sentence, flags=re.IGNORECASE)
        if part.strip()
    ]


def _extract_aspect_snippets(sentence, keywords):
    snippets = []
    clauses = _split_clauses(sentence)

    # Prefer clause-level snippets so mixed statements are attributed correctly per aspect.
    for clause in clauses:
        if _contains_keyword(clause, keywords):
            snippets.append(clause)

    if not snippets and _contains_keyword(sentence, keywords):
        snippets.append(sentence)

    return snippets


def format_scores(scores):
    pos_score = next(item["score"] for item in scores if item["label"] == "POSITIVE")
    neg_score = next(item["score"] for item in scores if item["label"] == "NEGATIVE")
    return pos_score, neg_score


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _normalize_signed(values):
    if not values:
        return []
    max_abs = max(abs(v) for v in values)
    if max_abs == 0:
        return [0.0 for _ in values]
    return [v / max_abs for v in values]


def _prediction_dict(text):
    output = analyzer(text, truncation=True, max_length=512)[0]
    pos, neg = format_scores(output)
    if pos >= neg:
        return {
            "class_label": "POSITIVE",
            "class_probability": pos,
            "positive_probability": pos,
            "negative_probability": neg,
        }
    return {
        "class_label": "NEGATIVE",
        "class_probability": neg,
        "positive_probability": pos,
        "negative_probability": neg,
    }


def _predict_proba(texts):
    probabilities = []
    for text in texts:
        output = analyzer(text, truncation=True, max_length=512)[0]
        pos, neg = format_scores(output)
        probabilities.append([neg, pos])
    return probabilities


def _build_summary(tokens, top_k=5):
    positives = [t for t in tokens if t["attribution"] > 0]
    negatives = [t for t in tokens if t["attribution"] < 0]
    positives.sort(key=lambda item: item["attribution"], reverse=True)
    negatives.sort(key=lambda item: item["attribution"])

    return {
        "top_positive": [
            {"token": t["token"], "score": round(t["attribution"], 4)}
            for t in positives[:top_k]
        ],
        "top_negative": [
            {"token": t["token"], "score": round(t["attribution"], 4)}
            for t in negatives[:top_k]
        ],
    }


def _build_alignment_from_review(text):
    tokens = []
    for match in re.finditer(r"\S+", text):
        raw = match.group(0)
        cleaned = raw.strip(".,!?\"'()[]{}")
        if not cleaned:
            continue
        tokens.append(
            {
                "token": cleaned,
                "raw": raw,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _explain_with_lime(text):
    if LimeTextExplainer is None:
        raise RuntimeError("LIME is not installed")

    prediction = _prediction_dict(text)
    class_label = prediction["class_label"]
    class_index = 1
    attribution_target = "POSITIVE"

    explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])
    explanation = explainer.explain_instance(
        text,
        _predict_proba,
        labels=[class_index],
        num_features=20,
        num_samples=500,
    )

    token_weights = dict(explanation.as_list(label=class_index))
    aligned_tokens = _build_alignment_from_review(text)

    attributions = []
    raw_values = []
    for token in aligned_tokens:
        weight = float(token_weights.get(token["token"], 0.0))
        weight = _clamp(weight, -1.0, 1.0)
        raw_values.append(weight)
        attributions.append(
            {
                "token": token["token"],
                "start": token["start"],
                "end": token["end"],
                "attribution": weight,
            }
        )

    normalized = _normalize_signed(raw_values)
    for idx, value in enumerate(normalized):
        attributions[idx]["normalized_attribution"] = round(value, 4)
        attributions[idx]["attribution"] = round(attributions[idx]["attribution"], 4)
        attributions[idx]["sign"] = (
            "positive" if attributions[idx]["attribution"] > 0 else "negative" if attributions[idx]["attribution"] < 0 else "neutral"
        )

    return {
        "method": "lime",
        "class_label": class_label,
        "class_probability": round(prediction["class_probability"], 4),
        "attribution_target": attribution_target,
        "tokens": attributions,
        "summary": _build_summary(attributions),
    }


def _explain_with_shap(text):
    if shap is None:
        raise RuntimeError("SHAP is not installed")

    prediction = _prediction_dict(text)
    class_label = prediction["class_label"]
    class_index = 1
    attribution_target = "POSITIVE"

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(_predict_proba, masker)
    shap_values = explainer([text], max_evals=500)

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    attributions = []
    raw_values = []
    cursor = 0

    for i, token in enumerate(tokens):
        token_text = str(token)
        if not token_text.strip():
            continue

        idx = text.find(token_text, cursor)
        if idx == -1:
            idx = text.find(token_text)
            if idx == -1:
                continue

        cursor = idx + len(token_text)
        value = float(values[i][class_index])
        value = _clamp(value, -1.0, 1.0)
        raw_values.append(value)
        attributions.append(
            {
                "token": token_text,
                "start": idx,
                "end": idx + len(token_text),
                "attribution": value,
            }
        )

    normalized = _normalize_signed(raw_values)
    for idx, value in enumerate(normalized):
        attributions[idx]["normalized_attribution"] = round(value, 4)
        attributions[idx]["attribution"] = round(attributions[idx]["attribution"], 4)
        attributions[idx]["sign"] = (
            "positive" if attributions[idx]["attribution"] > 0 else "negative" if attributions[idx]["attribution"] < 0 else "neutral"
        )

    return {
        "method": "shap",
        "class_label": class_label,
        "class_probability": round(prediction["class_probability"], 4),
        "attribution_target": attribution_target,
        "tokens": attributions,
        "summary": _build_summary(attributions),
    }


def _explain_with_leave_one_out(text, requested_method, errors=None):
    """Deterministic fallback explanation using token removal impact on positive probability."""
    prediction = _prediction_dict(text)
    aligned_tokens = _build_alignment_from_review(text)

    baseline_pos = float(prediction["positive_probability"])
    attributions = []
    raw_values = []

    for idx, token in enumerate(aligned_tokens):
        perturbed_tokens = [t["raw"] for i, t in enumerate(aligned_tokens) if i != idx]
        perturbed_text = " ".join(perturbed_tokens).strip()
        if not perturbed_text:
            perturbed_pos = baseline_pos
        else:
            perturbed_prediction = _prediction_dict(perturbed_text)
            perturbed_pos = float(perturbed_prediction["positive_probability"])

        impact = _clamp(baseline_pos - perturbed_pos, -1.0, 1.0)
        raw_values.append(impact)
        attributions.append(
            {
                "token": token["token"],
                "start": token["start"],
                "end": token["end"],
                "attribution": impact,
            }
        )

    normalized = _normalize_signed(raw_values)
    for idx, value in enumerate(normalized):
        attributions[idx]["normalized_attribution"] = round(value, 4)
        attributions[idx]["attribution"] = round(attributions[idx]["attribution"], 4)
        attributions[idx]["sign"] = (
            "positive" if attributions[idx]["attribution"] > 0 else "negative" if attributions[idx]["attribution"] < 0 else "neutral"
        )

    error_text = "; ".join(errors or [])
    return {
        "method": f"{requested_method}_fallback",
        "class_label": prediction["class_label"],
        "class_probability": round(prediction["class_probability"], 4),
        "attribution_target": "POSITIVE",
        "tokens": attributions,
        "summary": _build_summary(attributions),
        "note": "Fallback explanation was used because the requested explainer failed.",
        "fallback_reason": error_text,
    }


def explain_review(text, method="lime"):
    selected = (method or "lime").lower()
    if selected not in _ALLOWED_EXPLAIN_METHODS:
        raise ValueError("Unsupported explain method")

    if selected == "lime":
        try:
            return _explain_with_lime(text)
        except Exception as lime_exc:
            try:
                return _explain_with_shap(text)
            except Exception as shap_exc:
                return _explain_with_leave_one_out(
                    text,
                    requested_method="lime",
                    errors=[f"lime error: {lime_exc}", f"shap error: {shap_exc}"],
                )

    try:
        return _explain_with_shap(text)
    except Exception as shap_exc:
        try:
            return _explain_with_lime(text)
        except Exception as lime_exc:
            return _explain_with_leave_one_out(
                text,
                requested_method="shap",
                errors=[f"shap error: {shap_exc}", f"lime error: {lime_exc}"],
            )


def _split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def analyze_trajectory(text):
    """Analyze sentiment per sentence, returning a narrative arc."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    results = []
    for i, sentence in enumerate(sentences):
        output = analyzer(sentence, truncation=True, max_length=512)[0]
        pos, neg = format_scores(output)
        results.append({
            "index": i,
            "sentence": sentence,
            "score": round(pos - neg, 4),
            "positive": round(pos, 4),
            "negative": round(neg, 4),
        })
    return results


def analyze_aspects(text):
    """Score sentiment for each cinematic aspect via keyword matching."""
    sentences = _split_sentences(text)
    aspect_scores = {aspect: [] for aspect in _ASPECT_KEYWORDS}

    for sentence in sentences:
        for aspect, keywords in _ASPECT_KEYWORDS.items():
            snippets = _extract_aspect_snippets(sentence, keywords)
            for snippet in snippets:
                output = analyzer(snippet, truncation=True, max_length=512)[0]
                pos, neg = format_scores(output)
                aspect_scores[aspect].append(pos - neg)

    return {
        aspect: round(sum(scores) / len(scores), 4) if scores else None
        for aspect, scores in aspect_scores.items()
    }


def analyze_scatter(reviews):
    """Classify a batch of reviews and return per-word positive/negative frequency data."""
    positive_counts = Counter()
    negative_counts = Counter()

    for text in reviews:
        text = text.strip()
        if not text:
            continue

        output = analyzer(text, truncation=True, max_length=512)[0]
        pos, neg = format_scores(output)

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in _STOPWORDS]

        if pos >= neg:
            positive_counts.update(words)
        else:
            negative_counts.update(words)

    all_words = set(positive_counts) | set(negative_counts)
    data = []
    for word in all_words:
        pos_freq = positive_counts.get(word, 0)
        neg_freq = negative_counts.get(word, 0)
        if pos_freq + neg_freq >= 2:
            data.append({
                "word": word,
                "positive_freq": pos_freq,
                "negative_freq": neg_freq,
            })

    data.sort(key=lambda x: x["positive_freq"] + x["negative_freq"], reverse=True)
    return data[:200]


def analyze_review(text, explain_method="lime"):
    overall_result = analyzer(text, truncation=True, max_length=512)[0]
    pos_prob, neg_prob = format_scores(overall_result)

    token_scores = []
    for token in text.split():
        clean_token = token.strip(".,!?\"'()[]{}")
        if not clean_token:
            continue

        token_result = analyzer(clean_token)[0]
        token_pos_prob, token_neg_prob = format_scores(token_result)
        token_scores.append({
            "token": clean_token,
            "positive_probability": token_pos_prob,
            "negative_probability": token_neg_prob,
        })

    result = {
        "review": text,
        "overall": {
            "positive_probability": pos_prob,
            "negative_probability": neg_prob,
        },
        "tokens": token_scores,
        "trajectory": analyze_trajectory(text),
        "aspects": analyze_aspects(text),
    }

    result["explanation"] = explain_review(text, explain_method)

    return result


def analyze_full_and_word_wise(text):
    result = analyze_review(text)

    print(f"\n{'=' * 50}")
    print(f"REVIEW: '{text}'")
    print(f"{'=' * 50}")

    print("--- OVERALL PROBABILITY ---")
    print(f"Positive: {result['overall']['positive_probability'] * 100:.1f}%")
    print(f"Negative: {result['overall']['negative_probability'] * 100:.1f}%\n")

    print("--- TOKEN-BY-TOKEN PROBABILITY ---")
    print(f"{'Token':<15} | {'Positive %':<12} | {'Negative %':<12}")
    print("-" * 45)

    for token_result in result["tokens"]:
        token = token_result["token"]
        token_pos = token_result["positive_probability"]
        token_neg = token_result["negative_probability"]
        print(f"{token:<15} | {token_pos * 100:>9.1f}% | {token_neg * 100:>9.1f}%")
