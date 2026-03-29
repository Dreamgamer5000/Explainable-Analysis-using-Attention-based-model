from pathlib import Path
import re
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from model.model import analyzer, format_scores

matplotlib.use("Agg")


VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuls"


def _ensure_visuals_dir(output_dir=None):
    target_dir = Path(output_dir) if output_dir else VISUALS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_word_sentiment_bubbles(positive_words, negative_words, output_dir=None, max_words=100):
    """
    Create a bubble plot where:
    - X-axis: sentiment position (left=-1 for negative, right=+1 for positive)
    - Y-axis: random jitter for visual separation
    - Bubble size: word frequency
    - Color: sentiment (green for positive, red for negative)
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "word_sentiment_bubbles.png"

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare positive words
    top_pos = sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:max_words // 2]
    pos_words = [w for w, _ in top_pos]
    pos_freqs = [f for _, f in top_pos]

    # Prepare negative words
    top_neg = sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:max_words // 2]
    neg_words = [w for w, _ in top_neg]
    neg_freqs = [f for _, f in top_neg]

    if pos_words or neg_words:
        max_freq = max(max(pos_freqs) if pos_freqs else 1, max(neg_freqs) if neg_freqs else 1)

        # Plot positive words (right side, green)
        if pos_words:
            pos_x = np.random.uniform(0.3, 1.0, len(pos_words))
            pos_y = np.random.uniform(-1, 1, len(pos_words))
            pos_sizes = [100 + (f / max_freq) * 500 for f in pos_freqs]
            ax.scatter(pos_x, pos_y, s=pos_sizes, c="#1f9d55", alpha=0.6, edgecolors="#0d5f2f", linewidth=1)
            for i, word in enumerate(pos_words):
                ax.text(pos_x[i], pos_y[i], word, fontsize=8, ha="center", va="center", weight="bold")

        # Plot negative words (left side, red)
        if neg_words:
            neg_x = np.random.uniform(-1.0, -0.3, len(neg_words))
            neg_y = np.random.uniform(-1, 1, len(neg_words))
            neg_sizes = [100 + (f / max_freq) * 500 for f in neg_freqs]
            ax.scatter(neg_x, neg_y, s=neg_sizes, c="#c53030", alpha=0.6, edgecolors="#7f1d1d", linewidth=1)
            for i, word in enumerate(neg_words):
                ax.text(neg_x[i], neg_y[i], word, fontsize=8, ha="center", va="center", weight="bold")

    ax.axvline(x=0, color="#4a5568", linestyle="--", linewidth=2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Sentiment Polarity (← Negative | Positive →)", fontsize=12, weight="bold")
    ax.set_ylabel("Distribution Space", fontsize=12, weight="bold")
    ax.set_title("Word-Level Sentiment Bubbles (Size = Frequency)", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.set_yticks([])

    _save_figure(fig, output_path)
    return str(output_path)


def plot_class_distribution(labels, split_name, output_dir=None):
    """Plot positive vs negative distribution."""
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / f"class_distribution_{split_name}.png"

    pos_count = sum(1 for l in labels if l == 1)
    neg_count = sum(1 for l in labels if l == 0)
    total = len(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Positive", "Negative"], [pos_count, neg_count], color=["#1f9d55", "#c53030"], alpha=0.75)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 100, f"{int(height)}\n({height/total*100:.1f}%)", 
                ha="center", va="bottom", fontsize=10, weight="bold")

    ax.set_ylabel("Sample Count", fontsize=11, weight="bold")
    ax.set_title(f"Class Distribution - {split_name.upper()}", fontsize=13, weight="bold")
    ax.set_ylim(0, max(pos_count, neg_count) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    _save_figure(fig, output_path)
    return str(output_path)


def plot_review_length_distribution(texts, split_name, output_dir=None):
    """Plot distribution of review lengths."""
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / f"review_length_distribution_{split_name}.png"

    lengths = [len(text.split()) for text in texts]
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(lengths, bins=50, color="#2b6cb0", alpha=0.7, edgecolor="black")
    
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    ax.axvline(mean_len, color="#c53030", linestyle="--", linewidth=2, label=f"Mean: {mean_len:.0f}")
    ax.axvline(median_len, color="#1f9d55", linestyle="--", linewidth=2, label=f"Median: {median_len:.0f}")

    ax.set_xlabel("Review Length (words)", fontsize=11, weight="bold")
    ax.set_ylabel("Frequency", fontsize=11, weight="bold")
    ax.set_title(f"Review Length Distribution - {split_name.upper()}", fontsize=13, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    _save_figure(fig, output_path)
    return str(output_path)


def plot_confusion_matrix(y_true, y_pred, split_name, output_dir=None):
    """Plot confusion matrix."""
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / f"confusion_matrix_{split_name}.png"

    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    
    ax.figure.colorbar(im, ax=ax, label="Count")
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12, weight="bold")
    
    ax.set_ylabel("True Label", fontsize=11, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, weight="bold")
    ax.set_title(f"Confusion Matrix - {split_name.upper()}", fontsize=13, weight="bold")

    _save_figure(fig, output_path)
    return str(output_path)


def analyze_dataset_words(texts, labels, split_name, output_dir=None):
    """Extract and analyze word sentiment from dataset."""
    target_dir = _ensure_visuals_dir(output_dir)
    
    pos_words = Counter()
    neg_words = Counter()
    
    STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "was", "are", "be", "have", "has", "been", "it", "this", "that", "these", "those", "as", "from", "i", "you", "he", "she", "we", "they", "what", "which", "who", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now"}
    
    for text, label in zip(texts, labels):
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOPWORDS]
        
        if label == 1:
            pos_words.update(words)
        else:
            neg_words.update(words)
    
    return pos_words, neg_words


def create_dataset_visuals(texts, labels, split_name, output_dir=None):
    """Generate all dataset visualizations."""
    print(f"Generating visualizations for {split_name} split...")
    
    pos_words, neg_words = analyze_dataset_words(texts, labels, split_name, output_dir)
    
    visuals = {
        f"word_sentiment_bubbles_{split_name}": plot_word_sentiment_bubbles(pos_words, neg_words, output_dir),
        f"class_distribution_{split_name}": plot_class_distribution(labels, split_name, output_dir),
        f"review_length_distribution_{split_name}": plot_review_length_distribution(texts, split_name, output_dir),
    }
    
    return visuals
