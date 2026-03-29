# Dataset Visualizations & Evaluation Report

## Overview
Comprehensive evaluation and visualization pipeline for IMDB dataset with the current sentiment analysis model.

## Generated Visualizations

### 1. **Word-Level Sentiment Bubbles** (`word_sentiment_bubbles.png`)
- **Purpose**: Interactive word cloud where words float in 2D space
- **X-Axis**: Sentiment polarity (← Negative | Positive →)
- **Y-Axis**: Distribution space (random jitter for visual separation)
- **Bubble Size**: Word frequency (larger = more frequent)
- **Colors**: Red for negative words, Green for positive words
- **Use Case**: Understand which words are most strongly associated with each sentiment class

### 2. **Class Distribution** (`class_distribution_train.png`, `class_distribution_test.png`)
- **Purpose**: Visualize balance between positive and negative reviews
- **Content**: Bar chart showing count and percentage of each class
- **Use Case**: Identify class imbalance in dataset

### 3. **Review Length Distribution** (`review_length_distribution_train.png`, `review_length_distribution_test.png`)
- **Purpose**: Analyze text length patterns
- **Content**: Histogram with mean and median indicators
- **X-Axis**: Review length in words (0-1200+)
- **Use Case**: Understand input size variations and data characteristics

### 4. **Confusion Matrix** (`confusion_matrix_train.png`, `confusion_matrix_test.png`)
- **Purpose**: Detailed classification performance breakdown
- **Content**: 2×2 matrix showing True Negatives, True Positives, False Negatives, False Positives
- **Use Case**: Diagnose model bias and detection rates per class

## Evaluation Metrics

The pipeline computes:
- **Accuracy**: Overall correctness percentage
- **Precision**: Positive prediction correctness
- **Recall**: Positive detection rate
- **F1-Score**: Harmonic mean of precision and recall

## Files Generated

- [evaluate_dataset_comprehensive.py](evaluate_dataset_comprehensive.py) - Main evaluation script
- [model/dataset_visuals.py](model/dataset_visuals.py) - Visualization module
- All generated charts saved in [visuls/](visuls/)

## Usage

### Quick evaluation (sample):
```bash
python3 evaluate_dataset_comprehensive.py --splits test --batch-size 32 --limit 2000
```

### Full evaluation (all data):
```bash
python3 evaluate_dataset_comprehensive.py --splits train,test --batch-size 32
```

### Options:
- `--splits`: Comma-separated dataset splits (train, test)
- `--batch-size`: Inference batch size (default: 32)
- `--limit`: Sample limit per split for quick runs (optional)

## Key Insights

1. **Word Sentiment Bubbles**: Shows which words drive positive vs negative predictions
2. **Class Distribution**: Reveals if dataset is balanced
3. **Review Lengths**: Shows typical review sizes the model processes
4. **Confusion Matrix**: Identifies systematic errors and biases
