import re
from collections import Counter

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

_ALLOWED_EXPLAIN_METHODS = {"auto", "lime", "shap"}

_STOPWORDS = {
    'the', 'and', 'for', 'was', 'this', 'that', 'with', 'are', 'its', 'but',
    'not', 'you', 'all', 'can', 'had', 'her', 'she', 'him', 'his', 'how',
    'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did',
    'get', 'has', 'let', 'put', 'say', 'too', 'use', 'from', 'they', 'have',
    'been', 'more', 'will', 'also', 'what', 'when', 'than', 'then', 'them',
    'some', 'into', 'just', 'like', 'time', 'very', 'even', 'most', 'over',
    'such', 'only', 'come', 'could', 'would', 'should', 'there', 'their',
    'which', 'about', 'after', 'before', 'other', 'these', 'those', 'being',
    'while', 'where', 'every', 'still', 'might', 'think', 'good', 'well',
    'much', 'make', 'know', 'take', 'back', 'give', 'same', 'here', 'does',
    'each', 'both', 'many', 'film', 'movie', 'watch',
}

_ASPECT_KEYWORDS = {
    'Acting': [
        'acting', 'actor', 'actress', 'performance', 'performances',
        'cast', 'casting', 'portrayal', 'role', 'plays', 'played',
        'character', 'characters',
    ],
    'Directing': [
        'director', 'direction', 'directed', 'directing',
        'filmmaker', 'vision', 'pacing', 'pace', 'helm', 'helmed',
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
    class_index = 1 if class_label == "POSITIVE" else 0

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
        "tokens": attributions,
        "summary": _build_summary(attributions),
    }


def _explain_with_shap(text):
    if shap is None:
        raise RuntimeError("SHAP is not installed")

    prediction = _prediction_dict(text)
    class_label = prediction["class_label"]
    class_index = 1 if class_label == "POSITIVE" else 0

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
        "tokens": attributions,
        "summary": _build_summary(attributions),
    }


def explain_review(text, method="auto"):
    selected = (method or "auto").lower()
    if selected not in _ALLOWED_EXPLAIN_METHODS:
        raise ValueError("Unsupported explain method")

    if selected == "lime":
        return _explain_with_lime(text)
    if selected == "shap":
        return _explain_with_shap(text)

    try:
        return _explain_with_lime(text)
    except Exception:
        return _explain_with_shap(text)


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
        lower = sentence.lower()
        for aspect, keywords in _ASPECT_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                output = analyzer(sentence, truncation=True, max_length=512)[0]
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


def analyze_review(text, explain_method="auto"):
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

    try:
        result["explanation"] = explain_review(text, explain_method)
    except Exception as exc:
        result["explanation"] = {
            "method": "unavailable",
            "error": str(exc),
            "tokens": [],
            "summary": {"top_positive": [], "top_negative": []},
        }

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


if __name__ == "__main__":
    test_reviews = [
        "An absolute triumph of cinema. The acting was brilliant, and the visuals were stunning.",
        "A complete waste of time. The plot was poorly written, and the dialogue felt incredibly wooden.",
        "The special effects were amazing, but the storyline was terribly boring and dragged on forever.",
        "Absolutely loved it! Highly recommended.",
        "Oh sure, because watching paint dry is exactly what I wanted to pay twenty dollars for. Absolute garbage.",
        "The movie was okay, but the villain was completely terrifying and awesome.",
    ]

    for review in test_reviews:
        analyze_full_and_word_wise(review)
