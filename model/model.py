import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import pipeline

analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None,
)

_DOMAIN_STOPWORDS = {"film", "movie", "watch"}
_STOPWORDS = set(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS

_ASPECT_KEYWORDS = {
    "Acting": [
        "acting", "actor", "actress", "performance", "performances",
        "cast", "casting", "portrayal", "role", "plays", "played",
    ],
    "Directing": [
        "director", "direction", "directed", "directing",
        "filmmaker", "vision", "pacing", "pace", "helm", "helmed",
        "directly", "directorial",
    ],
    "Plot": [
        "plot", "story", "script", "writing", "narrative", "storyline",
        "ending", "twist", "screenplay", "dialogue", "scene", "scenes",
        "premise",
    ],
    "Cinematography": [
        "cinematography", "visual", "visuals", "photography",
        "lighting", "shot", "shots", "camera", "color", "colour",
        "beautiful", "stunning",
    ],
    "Soundtrack": [
        "soundtrack", "music", "score", "song", "songs", "audio",
        "sound", "musical", "composer", "track", "tracks",
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


def _split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def analyze_trajectory(text):
    sentences = _split_sentences(text)
    if not sentences:
        return []

    results = []
    for i, sentence in enumerate(sentences):
        output = analyzer(sentence, truncation=True, max_length=512)[0]
        pos, neg = format_scores(output)
        results.append(
            {
                "index": i,
                "sentence": sentence,
                "score": round(pos - neg, 4),
                "positive": round(pos, 4),
                "negative": round(neg, 4),
            }
        )
    return results


def analyze_aspects(text):
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
    positive_counts = Counter()
    negative_counts = Counter()

    for text in reviews:
        text = text.strip()
        if not text:
            continue

        output = analyzer(text, truncation=True, max_length=512)[0]
        pos, neg = format_scores(output)

        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
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
            data.append(
                {
                    "word": word,
                    "positive_freq": pos_freq,
                    "negative_freq": neg_freq,
                }
            )

    data.sort(key=lambda x: x["positive_freq"] + x["negative_freq"], reverse=True)
    return data[:200]


def analyze_review(text):
    overall_result = analyzer(text, truncation=True, max_length=512)[0]
    pos_prob, neg_prob = format_scores(overall_result)

    token_scores = []
    for token in text.split():
        clean_token = token.strip(".,!?\"'()[]{}")
        if not clean_token:
            continue

        token_result = analyzer(clean_token)[0]
        token_pos_prob, token_neg_prob = format_scores(token_result)
        token_scores.append(
            {
                "token": clean_token,
                "positive_probability": token_pos_prob,
                "negative_probability": token_neg_prob,
            }
        )

    return {
        "review": text,
        "overall": {
            "positive_probability": pos_prob,
            "negative_probability": neg_prob,
        },
        "tokens": token_scores,
        "trajectory": analyze_trajectory(text),
        "aspects": analyze_aspects(text),
    }


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
