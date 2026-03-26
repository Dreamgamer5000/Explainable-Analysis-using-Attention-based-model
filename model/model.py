from transformers import pipeline

analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None,
)


def format_scores(scores):
    pos_score = next(item["score"] for item in scores if item["label"] == "POSITIVE")
    neg_score = next(item["score"] for item in scores if item["label"] == "NEGATIVE")
    return pos_score, neg_score


def analyze_review(text):
    overall_result = analyzer(text)[0]
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
