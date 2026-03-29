from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.model import analyzer


VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuls"


def _ensure_visuals_dir(output_dir=None):
    target_dir = Path(output_dir) if output_dir else VISUALS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _clean_token_label(token):
    if token.startswith("##"):
        return token[2:]
    return token


def _attention_rollout(text, max_length=128, discard_ratio=0.6):
    tokenizer = analyzer.tokenizer
    model = analyzer.model
    device = next(model.parameters()).device

    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    attentions = outputs.attentions
    if not attentions:
        return [], []

    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len, device=device, dtype=attentions[0].dtype)
    identity = torch.eye(seq_len, device=device, dtype=attentions[0].dtype)

    for layer_attention in attentions:
        # Fuse heads and discard weak links so rollout emphasizes meaningful paths.
        attn = layer_attention[0].mean(dim=0)

        keep_k = max(1, int((1.0 - discard_ratio) * seq_len))
        top_values, top_indices = torch.topk(attn, k=keep_k, dim=-1)
        sparse_attn = torch.zeros_like(attn)
        sparse_attn.scatter_(dim=-1, index=top_indices, src=top_values)

        # Mix with residual to preserve token identity paths.
        attn = 0.5 * sparse_attn + 0.5 * identity
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        rollout = attn @ rollout

    input_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return rollout.cpu().numpy(), tokens


def plot_attention_rollout_heatmap(text, output_dir=None, max_tokens=32):
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "attention_rollout_heatmap.png"

    fig, ax = plt.subplots(figsize=(11, 9))

    if not text or not text.strip():
        ax.text(0.5, 0.5, "No review text available for attention rollout", ha="center", va="center", fontsize=12)
        ax.axis("off")
        _save_figure(fig, output_path)
        return str(output_path)

    try:
        matrix, tokens = _attention_rollout(text)
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Attention rollout unavailable: {exc}",
            ha="center",
            va="center",
            fontsize=11,
            wrap=True,
        )
        ax.axis("off")
        _save_figure(fig, output_path)
        return str(output_path)

    if len(tokens) == 0:
        ax.text(0.5, 0.5, "Attention rollout returned no tokens", ha="center", va="center", fontsize=12)
        ax.axis("off")
        _save_figure(fig, output_path)
        return str(output_path)

    visible_indices = [
        idx for idx, token in enumerate(tokens)
        if token not in {"[CLS]", "[SEP]", "[PAD]"}
    ]
    if not visible_indices:
        visible_indices = list(range(len(tokens)))

    visible_indices = visible_indices[:max_tokens]
    plot_tokens = [_clean_token_label(tokens[idx]) for idx in visible_indices]
    plot_matrix = matrix[np.ix_(visible_indices, visible_indices)]

    finite_vals = plot_matrix[np.isfinite(plot_matrix)]
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(finite_vals, 5))
        vmax = float(np.percentile(finite_vals, 99))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    image = ax.imshow(plot_matrix, cmap="magma", vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Attention Rollout Weight")

    ax.set_title("Attention Rollout Heatmap (Token-to-Token)", fontsize=14, weight="bold")
    ax.set_xlabel("Attended Token")
    ax.set_ylabel("Source Token")
    ax.set_xticks(range(len(plot_tokens)))
    ax.set_yticks(range(len(plot_tokens)))
    ax.set_xticklabels(plot_tokens, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(plot_tokens, fontsize=8)

    _save_figure(fig, output_path)
    return str(output_path)


def plot_sentiment_trajectory(trajectory, output_dir=None):
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "sentiment_trajectory.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    if trajectory:
        x = list(range(1, len(trajectory) + 1))
        y = [item.get("score", 0.0) for item in trajectory]
        colors = ["#1f9d55" if score >= 0 else "#c53030" for score in y]

        ax.plot(x, y, color="#2b6cb0", linewidth=2, marker="o", markersize=6)
        ax.scatter(x, y, c=colors, s=65, zorder=3)
        for i, score in enumerate(y):
            ax.text(x[i], score + (0.05 if score >= 0 else -0.08), f"{score:.2f}", ha="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No sentence-level trajectory data available", ha="center", va="center", fontsize=12)

    ax.axhline(0, color="#4a5568", linestyle="--", linewidth=1)
    ax.set_title("Sentiment Trajectory Across Sentences", fontsize=15, weight="bold")
    ax.set_xlabel("Sentence Number")
    ax.set_ylabel("Sentiment Score (Positive - Negative)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)

    _save_figure(fig, output_path)
    return str(output_path)


def plot_aspect_sentiment_bar(aspects, output_dir=None):
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "aspect_sentiment_bar.png"

    labels = list(aspects.keys()) if aspects else []
    values = [aspects[k] if aspects[k] is not None else 0 for k in labels]
    has_data = [aspects[k] is not None for k in labels]
    colors = ["#1f9d55" if val >= 0 else "#c53030" for val in values]
    colors = [colors[i] if has_data[i] else "#a0aec0" for i in range(len(colors))]

    fig, ax = plt.subplots(figsize=(12, 6))
    if labels:
        bars = ax.bar(labels, values, color=colors)
        for i, bar in enumerate(bars):
            label = "No mentions" if not has_data[i] else f"{values[i]:.2f}"
            y_text = 0.03 if values[i] >= 0 else -0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                values[i] + y_text,
                label,
                ha="center",
                va="center",
                fontsize=9,
            )
        ax.axhline(0, color="#4a5568", linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, "No aspect data available", ha="center", va="center", fontsize=12)

    ax.set_title("Aspect-Based Sentiment Scores", fontsize=15, weight="bold")
    ax.set_xlabel("Movie Aspects")
    ax.set_ylabel("Average Sentiment Score")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _save_figure(fig, output_path)
    return str(output_path)


def plot_explanation_bars(explanation, output_dir=None):
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "lime_shap_explanation.png"

    method = (explanation or {}).get("method", "unavailable")
    tokens = (explanation or {}).get("tokens", [])

    top_positive = sorted([t for t in tokens if t.get("attribution", 0) > 0], key=lambda t: t["attribution"], reverse=True)[:8]
    top_negative = sorted([t for t in tokens if t.get("attribution", 0) < 0], key=lambda t: t["attribution"])[:8]
    selected = top_negative + top_positive

    fig, ax = plt.subplots(figsize=(13, 7))
    if selected:
        labels = [item.get("token", "") for item in selected]
        values = [float(item.get("attribution", 0.0)) for item in selected]
        colors = ["#c53030" if v < 0 else "#1f9d55" for v in values]
        y_positions = list(range(len(labels)))

        bars = ax.barh(y_positions, values, color=colors)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        for i, bar in enumerate(bars):
            x_pos = values[i] + (0.01 if values[i] >= 0 else -0.01)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{values[i]:.3f}", va="center", fontsize=9)
        ax.axvline(0, color="#4a5568", linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, "No LIME/SHAP attribution data available", ha="center", va="center", fontsize=12)

    title_method = method.upper() if method and method != "unavailable" else "LIME/SHAP"
    ax.set_title(f"{title_method} Token Attribution Explanation", fontsize=15, weight="bold")
    ax.set_xlabel("Attribution Score (Positive supports positive sentiment)")
    ax.set_ylabel("Tokens")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    _save_figure(fig, output_path)
    return str(output_path)


def plot_word_frequency_scatter(scatter_data, output_dir=None):
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "word_frequency_scatter.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    if scatter_data:
        x = [d.get("negative_freq", 0) for d in scatter_data]
        y = [d.get("positive_freq", 0) for d in scatter_data]
        words = [d.get("word", "") for d in scatter_data]
        sizes = [40 + 18 * (px + py) for px, py in zip(x, y)]

        ax.scatter(x, y, s=sizes, alpha=0.65, color="#3182ce", edgecolors="#1a365d", linewidth=0.6)

        top_points = sorted(scatter_data, key=lambda d: d.get("positive_freq", 0) + d.get("negative_freq", 0), reverse=True)[:18]
        top_words = {d.get("word", "") for d in top_points}
        for i, word in enumerate(words):
            if word in top_words:
                ax.text(x[i] + 0.04, y[i] + 0.04, word, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No word frequency data available", ha="center", va="center", fontsize=12)

    max_axis = 1
    if scatter_data:
        max_axis = max(
            max([item.get("positive_freq", 0) for item in scatter_data] + [1]),
            max([item.get("negative_freq", 0) for item in scatter_data] + [1]),
        )
    ax.plot([0, max_axis + 1], [0, max_axis + 1], linestyle="--", linewidth=1, color="#4a5568")
    ax.set_xlim(-0.2, max_axis + 1.2)
    ax.set_ylim(-0.2, max_axis + 1.2)

    ax.set_title("Word Frequency Scatter: Positive vs Negative Reviews", fontsize=15, weight="bold")
    ax.set_xlabel("Word Frequency in Negative Reviews")
    ax.set_ylabel("Word Frequency in Positive Reviews")
    ax.grid(True, linestyle="--", alpha=0.35)

    _save_figure(fig, output_path)
    return str(output_path)


def plot_token_probability_table(tokens, output_dir=None, max_rows=30):
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "token_probability_table.png"

    trimmed = tokens[:max_rows] if tokens else []
    rows = []
    for item in trimmed:
        rows.append([
            item.get("token", ""),
            f"{float(item.get('positive_probability', 0.0)):.4f}",
            f"{float(item.get('negative_probability', 0.0)):.4f}",
        ])

    if not rows:
        rows = [["No token data", "-", "-"]]

    fig_height = max(3.5, 1.0 + 0.33 * len(rows))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Token", "Positive Probability", "Negative Probability"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fafc" if row % 2 == 0 else "#edf2f7")

    ax.set_title("Token-Level Sentiment Probability Table", fontsize=14, weight="bold", pad=18)

    _save_figure(fig, output_path)
    return str(output_path)


def create_review_visuals(result, output_dir=None):
    return {
        "sentiment_trajectory": plot_sentiment_trajectory(result.get("trajectory", []), output_dir=output_dir),
        "aspect_sentiment_bar": plot_aspect_sentiment_bar(result.get("aspects", {}), output_dir=output_dir),
        "lime_shap_explanation": plot_explanation_bars(result.get("explanation", {}), output_dir=output_dir),
        "token_probability_table": plot_token_probability_table(result.get("tokens", []), output_dir=output_dir),
        "attention_rollout_heatmap": plot_attention_rollout_heatmap(result.get("review", ""), output_dir=output_dir),
    }


def create_scatter_visual(scatter_data, output_dir=None):
    return {
        "word_frequency_scatter": plot_word_frequency_scatter(scatter_data, output_dir=output_dir),
    }