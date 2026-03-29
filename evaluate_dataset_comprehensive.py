import argparse
from time import perf_counter

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model.model import analyzer, format_scores
from model.dataset_visuals import create_dataset_visuals, plot_confusion_matrix


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


def evaluate_and_visualize(split, batch_size, limit=None):
    print(f"\nLoading {split} dataset...")
    dataset = load_dataset("imdb", split=split)
    
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Limited to {limit} samples")

    texts = dataset["text"]
    labels = dataset["label"]
    total = len(dataset)

    print(f"Total samples: {total}")
    print("Running inference...")
    
    start = perf_counter()
    predictions = []
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = texts[start_idx:end_idx]
        batch_preds = predict_labels(batch_texts, batch_size=batch_size)
        predictions.extend(batch_preds)

        if (start_idx // batch_size) % 20 == 0:
            done = end_idx
            pct = (done / total) * 100
            print(f"  Progress: {done}/{total} ({pct:.1f}%)")

    elapsed = perf_counter() - start

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {split.upper()}")
    print("="*60)
    print(f"Accuracy:  {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1-Score:  {f1:.6f}")
    print(f"Runtime:   {elapsed:.2f} seconds")
    print("="*60)

    # Generate visualizations
    print(f"\nGenerating visualizations for {split} split...")
    visuals = create_dataset_visuals(texts, labels, split)
    confusion = plot_confusion_matrix(labels, predictions, split)
    visuals["confusion_matrix"] = confusion

    print(f"Visualizations saved to visuls/")
    for name, path in visuals.items():
        print(f"  - {path}")

    return {
        "split": split,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": total,
        "elapsed": elapsed,
        "visuals": visuals,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive IMDB dataset evaluation and visualization")
    parser.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per split for quick runs")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    results = {}

    for split in splits:
        result = evaluate_and_visualize(split, args.batch_size, args.limit)
        results[split] = result

    # Summary
    print("\n" + "="*60)
    print("SUMMARY ACROSS SPLITS")
    print("="*60)
    for split, result in results.items():
        print(f"{split.upper():<10} | Acc: {result['accuracy']:.4f} | F1: {result['f1']:.4f} | Samples: {result['total_samples']}")
    print("="*60)


if __name__ == "__main__":
    main()
