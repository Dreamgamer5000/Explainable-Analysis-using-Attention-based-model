import argparse
from time import perf_counter

from datasets import load_dataset

from model.model import analyzer, format_scores


def predict_labels(texts, batch_size):
    outputs = analyzer(
        texts,
        truncation=True,
        max_length=512,
        batch_size=batch_size,
    )

    predictions = []
    for output in outputs:
        pos, neg = format_scores(output)
        predictions.append(1 if pos >= neg else 0)
    return predictions


def evaluate_imdb(split, batch_size, limit=None):
    dataset = load_dataset("imdb", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    total = len(dataset)
    correct = 0

    start = perf_counter()
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = dataset[start_idx:end_idx]

        texts = batch["text"]
        labels = batch["label"]
        preds = predict_labels(texts, batch_size=batch_size)

        correct += sum(int(p == y) for p, y in zip(preds, labels))

        if (start_idx // batch_size) % 20 == 0:
            done = end_idx
            pct = (done / total) * 100
            print(f"Progress: {done}/{total} ({pct:.1f}%)")

    elapsed = perf_counter() - start
    accuracy = correct / total if total else 0.0

    print("\nEvaluation complete")
    print(f"Split: {split}")
    print(f"Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Elapsed seconds: {elapsed:.2f}")

    return accuracy, total, elapsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate current sentiment model on IMDB")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick runs")
    args = parser.parse_args()

    evaluate_imdb(split=args.split, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()