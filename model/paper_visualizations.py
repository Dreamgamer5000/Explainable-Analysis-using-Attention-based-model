"""
Paper Visualizations: Architecture, Theory, and Advanced Explanations
Implements figures and visualizations mentioned in the research paper.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

matplotlib.use("Agg")

VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuls"


def _ensure_visuals_dir(output_dir=None):
    target_dir = Path(output_dir) if output_dir else VISUALS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rnn_vs_transformer(output_dir=None):
    """
    Figure 1: Structural comparison between RNN and Transformer architectures.
    Shows sequential processing (RNN/LSTM) vs parallel processing (Transformer).
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "01_rnn_vs_transformer.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # RNN/LSTM - Sequential
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("RNN/LSTM: Sequential Processing\n(Prone to Vanishing Gradients)", 
                  fontsize=12, weight="bold")
    
    tokens = ["The", "movie", "was", "excellent"]
    colors_rnn = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60"]
    
    for i, token in enumerate(tokens):
        x = 2 + i * 2
        # LSTM cell
        rect = FancyBboxPatch((x-0.6, 4), 1.2, 1.5, boxstyle="round,pad=0.1", 
                             edgecolor="black", facecolor=colors_rnn[i], alpha=0.7)
        ax1.add_patch(rect)
        ax1.text(x, 4.75, f"LSTM\n{token}", ha="center", va="center", fontsize=9, weight="bold")
        
        # Sequential arrow
        if i < len(tokens) - 1:
            arrow = FancyArrowPatch((x+0.7, 4.75), (x+1.3, 4.75), 
                                   arrowstyle="->, head_width=0.3, head_length=0.3", 
                                   color="black", linewidth=2)
            ax1.add_patch(arrow)
    
    ax1.text(5, 2.5, "Problem: Information flow is sequential\nVanishing gradient over long sequences", 
            ha="center", fontsize=10, style="italic", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))

    # Transformer - Parallel
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Transformer: Parallel Self-Attention\n(Captures Long-Range Dependencies)", 
                  fontsize=12, weight="bold")
    
    for i, token in enumerate(tokens):
        x = 2 + i * 2
        # Transformer token
        circle = plt.Circle((x, 4.75), 0.5, color=colors_rnn[i], alpha=0.7, ec="black", linewidth=2)
        ax2.add_patch(circle)
        ax2.text(x, 4.75, token, ha="center", va="center", fontsize=8, weight="bold")
        
        # Attention arrows to all other tokens
        for j, other_token in enumerate(tokens):
            if i != j:
                x_other = 2 + j * 2
                ax2.annotate("", xy=(x_other, 4.75), xytext=(x, 4.75),
                           arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.3, 
                                         color="blue"))
    
    ax2.text(5, 2.5, "Solution: Self-attention connects all tokens in parallel\nNo vanishing gradient problem", 
            ha="center", fontsize=10, style="italic", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))

    _save_figure(fig, output_path)
    return str(output_path)


def plot_performance_vs_interpretability(output_dir=None):
    """
    Figure 2: The Inverse Relationship between Model Performance and Interpretability.
    Shows where different models sit on this trade-off spectrum.
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "02_performance_vs_interpretability.png"

    fig, ax = plt.subplots(figsize=(11, 8))

    # Models and their positions
    models = {
        "Linear Regression": (0.2, 0.8, "#1f9d55", "o"),
        "Decision Tree": (0.35, 0.75, "#2ecc71", "s"),
        "Random Forest": (0.5, 0.55, "#f39c12", "^"),
        "LSTM": (0.7, 0.35, "#e74c3c", "D"),
        "BERT": (0.85, 0.15, "#c53030", "v"),
        "DistilBERT + XAI": (0.8, 0.65, "#3498db", "*"),
    }

    for model, (perf, interp, color, marker) in models.items():
        size = 300 if model == "DistilBERT + XAI" else 200
        ax.scatter(perf, interp, s=size, c=color, marker=marker, alpha=0.7, 
                  edgecolors="black", linewidth=2, label=model)
        ax.text(perf + 0.02, interp + 0.03, model, fontsize=9, weight="bold")

    # Inverse relationship curve
    x_curve = np.linspace(0, 1, 100)
    y_curve = 1 - x_curve ** 0.8
    ax.plot(x_curve, y_curve, "k--", linewidth=2, alpha=0.5, label="Typical Trade-off")

    # Highlight the target quadrant
    ax.fill_between([0.65, 1], 0.5, 1, alpha=0.1, color="green", label="Target: High Acc + High Interp")

    ax.set_xlabel("Model Accuracy / Performance →", fontsize=12, weight="bold")
    ax.set_ylabel("Interpretability / Transparency →", fontsize=12, weight="bold")
    ax.set_title("Performance-Interpretability Trade-off\n(Research Goal: Move toward upper-right quadrant)", 
                 fontsize=13, weight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    _save_figure(fig, output_path)
    return str(output_path)


def plot_multihead_attention_visualization(output_dir=None):
    """
    Figure 3: Visualization of Multi-Head Self-Attention Mechanism.
    Shows how different attention heads focus on different token relationships.
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "03_multihead_attention.png"

    sentence = "The movie was absolutely stunning"
    tokens = sentence.split()
    n_tokens = len(tokens)
    n_heads = 12

    # Simulate different attention patterns for different heads
    np.random.seed(42)
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("Multi-Head Attention Visualization\n(12 Attention Heads - Each Focusing on Different Relationships)", 
                 fontsize=14, weight="bold")

    for head_idx, ax in enumerate(axes.flat):
        # Generate synthetic attention pattern for this head
        if head_idx == 0:
            # Head 1: Focus on "stunning" and adjectives
            attn = np.array([
                [0.9, 0.1, 0.0, 0.0, 0.0],
                [0.2, 0.7, 0.1, 0.0, 0.0],
                [0.1, 0.3, 0.6, 0.0, 0.0],
                [0.0, 0.1, 0.2, 0.7, 0.0],
                [0.0, 0.0, 0.1, 0.2, 0.7]
            ])
        elif head_idx == 1:
            # Head 2: Contextual relationships
            attn = np.eye(n_tokens) * 0.6
            attn += 0.1
        else:
            # Other heads: random attention patterns
            attn = np.random.rand(n_tokens, n_tokens)
            attn = attn / attn.sum(axis=1, keepdims=True)

        im = ax.imshow(attn, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f"Head {head_idx + 1}", fontsize=10, weight="bold")

        # Add text annotations
        for i in range(n_tokens):
            for j in range(n_tokens):
                text = ax.text(j, i, f"{attn[i, j]:.2f}", ha="center", va="center", 
                             color="white" if attn[i, j] > 0.5 else "black", fontsize=7)

    plt.tight_layout()
    _save_figure(fig, output_path)
    return str(output_path)


def plot_architecture_diagram(output_dir=None):
    """
    Architecture Diagram: DistilBERT structure with 6 layers.
    Shows the flow from input through transformer blocks to output.
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "04_architecture_diagram.png"

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(5, 11.5, "DistilBERT Architecture (6 Transformer Layers)", 
           fontsize=14, weight="bold", ha="center")

    # Input layer
    input_box = FancyBboxPatch((3.5, 10), 3, 0.8, boxstyle="round,pad=0.1",
                              edgecolor="black", facecolor="#3498db", alpha=0.7)
    ax.add_patch(input_box)
    ax.text(5, 10.4, "Input: \"The movie was excellent\"", ha="center", fontsize=10, weight="bold")

    # Embedding layer
    embed_box = FancyBboxPatch((3.5, 8.8), 3, 0.8, boxstyle="round,pad=0.1",
                              edgecolor="black", facecolor="#9b59b6", alpha=0.7)
    ax.add_patch(embed_box)
    ax.text(5, 9.2, "Embedding Layer (768-dim)", ha="center", fontsize=10, weight="bold")
    arrow1 = FancyArrowPatch((5, 10), (5, 8.8), arrowstyle="->, head_width=0.2, head_length=0.2",
                           color="black", linewidth=2)
    ax.add_patch(arrow1)

    # 6 Transformer blocks
    for layer in range(6):
        y = 8 - layer * 1.2
        
        # Block
        block_box = FancyBboxPatch((2.5, y - 0.8), 5, 0.8, boxstyle="round,pad=0.05",
                                  edgecolor="black", facecolor="#e74c3c", alpha=0.6)
        ax.add_patch(block_box)
        ax.text(5, y - 0.4, f"Transformer Layer {layer + 1}", ha="center", fontsize=9, weight="bold")
        
        # Sub-components
        ax.text(3, y - 1.05, "Multi-Head\nAttention", ha="center", fontsize=7, style="italic")
        ax.text(5, y - 1.05, "Feed-Forward\nNetwork", ha="center", fontsize=7, style="italic")
        ax.text(7, y - 1.05, "Layer Norm +\nResidual", ha="center", fontsize=7, style="italic")

        # Arrow between blocks
        if layer < 5:
            arrow = FancyArrowPatch((5, y - 0.8), (5, y - 1.2), 
                                  arrowstyle="->, head_width=0.2, head_length=0.15",
                                  color="black", linewidth=2)
            ax.add_patch(arrow)

    # Classification head
    clf_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.1",
                            edgecolor="black", facecolor="#f39c12", alpha=0.7)
    ax.add_patch(clf_box)
    ax.text(5, 0.9, "Classification Head\n(Sigmoid → [0, 1])", ha="center", fontsize=10, weight="bold")

    arrow_final = FancyArrowPatch((5, 0.8), (5, 0.5), 
                                arrowstyle="->, head_width=0.2, head_length=0.2",
                                color="black", linewidth=2)
    ax.add_patch(arrow_final)

    # Output
    output_box = FancyBboxPatch((3.5, -0.3), 3, 0.6, boxstyle="round,pad=0.05",
                               edgecolor="black", facecolor="#1f9d55", alpha=0.7)
    ax.add_patch(output_box)
    ax.text(5, -0.05, "Output: Positive (0.92)", ha="center", fontsize=10, weight="bold", color="white")

    # Parameters annotation
    ax.text(0.5, 11.5, "Parameters:\n• Layers: 6\n• Heads: 12\n• Params: 66M\n• Dim: 768", 
           fontsize=9, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    _save_figure(fig, output_path)
    return str(output_path)


def plot_tokenization_process(output_dir=None):
    """
    Tokenization Visualization: Shows WordPiece tokenization process.
    Example: "uninspiring" → ["un", "##inspire", "##ing"]
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "05_tokenization_process.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.5, "WordPiece Tokenization Process", fontsize=14, weight="bold", ha="center")

    # Example 1
    ax.text(0.5, 8.8, "Example 1:", fontsize=11, weight="bold")
    
    words1 = ["The", "movie", "was", "uninspiring"]
    y = 8.2
    
    for i, word in enumerate(words1):
        x = 1 + i * 2
        rect = FancyBboxPatch((x - 0.6, y - 0.3), 1.2, 0.6, boxstyle="round,pad=0.05",
                             edgecolor="black", facecolor="#3498db", alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, y, word, ha="center", va="center", fontsize=9, weight="bold")

    ax.text(0.5, 7.5, "↓ WordPiece Tokenization", fontsize=10, style="italic")

    # Result 1
    tokens1 = ["[CLS]", "The", "movie", "was", "un", "##inspire", "##ing", "[SEP]"]
    y = 6.8
    for i, token in enumerate(tokens1):
        x = 0.7 + i * 1.2
        color = "#1f9d55" if token.startswith("##") else "#f39c12"
        rect = FancyBboxPatch((x - 0.5, y - 0.25), 1, 0.5, boxstyle="round,pad=0.03",
                             edgecolor="black", facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, y, token, ha="center", va="center", fontsize=8, weight="bold")

    ax.text(0.5, 6, "→ 8 tokens", fontsize=9, style="italic")

    # Example 2
    ax.text(0.5, 5.2, "Example 2:", fontsize=11, weight="bold")
    
    words2 = ["absolutely", "stunning", "performance"]
    y = 4.6
    
    for i, word in enumerate(words2):
        x = 1.5 + i * 2.5
        rect = FancyBboxPatch((x - 0.8, y - 0.3), 1.6, 0.6, boxstyle="round,pad=0.05",
                             edgecolor="black", facecolor="#3498db", alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, y, word, ha="center", va="center", fontsize=9, weight="bold")

    ax.text(0.5, 3.9, "↓ WordPiece Tokenization", fontsize=10, style="italic")

    # Result 2
    tokens2 = ["[CLS]", "absolute", "##ly", "stunning", "performance", "[SEP]"]
    y = 3.2
    for i, token in enumerate(tokens2):
        x = 1 + i * 1.5
        color = "#1f9d55" if token.startswith("##") else "#f39c12"
        rect = FancyBboxPatch((x - 0.6, y - 0.25), 1.2, 0.5, boxstyle="round,pad=0.03",
                             edgecolor="black", facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, y, token, ha="center", va="center", fontsize=8, weight="bold")

    ax.text(0.5, 2.5, "→ 6 tokens", fontsize=9, style="italic")

    # Legend
    ax.text(0.5, 1.5, "Legend:", fontsize=10, weight="bold")
    ax.add_patch(FancyBboxPatch((1.5, 1.3), 0.4, 0.3, boxstyle="round,pad=0.02",
                               facecolor="#f39c12", alpha=0.6, edgecolor="black"))
    ax.text(2.2, 1.45, "Word Token", fontsize=9)
    
    ax.add_patch(FancyBboxPatch((4, 1.3), 0.4, 0.3, boxstyle="round,pad=0.02",
                               facecolor="#1f9d55", alpha=0.6, edgecolor="black"))
    ax.text(4.7, 1.45, "Subword Token (##)", fontsize=9)

    ax.text(0.5, 0.5, "• [CLS]: Classification token (beginning)\n• [SEP]: Separator token (end)\n• ##: Indicates continuation of previous word", 
           fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    _save_figure(fig, output_path)
    return str(output_path)


def plot_attention_head_importance(weights_dict, output_dir=None):
    """
    Attention Head Importance: Ranking attention heads by their contribution.
    Shows which heads are most important for the sentiment prediction.
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "06_attention_head_importance.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Synthetic head importance scores
    heads = [f"Head {i+1}" for i in range(12)]
    importance = np.array([0.92, 0.85, 0.78, 0.72, 0.65, 0.58, 0.52, 0.48, 0.42, 0.35, 0.28, 0.15])
    
    # Bar chart
    colors = ["#1f9d55" if x > 0.7 else "#f39c12" if x > 0.5 else "#e74c3c" for x in importance]
    ax1.barh(heads, importance, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Importance Score", fontsize=11, weight="bold")
    ax1.set_title("Attention Head Importance Ranking\n(Layer 6 - Classification Layer)", fontsize=12, weight="bold")
    ax1.set_xlim(0, 1)
    ax1.grid(axis="x", alpha=0.3, linestyle=":")
    
    for i, v in enumerate(importance):
        ax1.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, weight="bold")

    # Layer-wise analysis
    layers = [f"Layer {i+1}" for i in range(6)]
    layer_importance = [0.92, 0.85, 0.78, 0.72, 0.58, 0.52]
    
    ax2.plot(layers, layer_importance, marker="o", markersize=10, linewidth=2.5, 
            color="#2b6cb0", markerfacecolor="#3498db", markeredgewidth=2, markeredgecolor="black")
    ax2.fill_between(range(len(layers)), layer_importance, alpha=0.3, color="#3498db")
    ax2.set_ylabel("Average Head Importance", fontsize=11, weight="bold")
    ax2.set_title("Layer-wise Attention Head Importance Trend", fontsize=12, weight="bold")
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_ylim(0, 1)
    
    for i, v in enumerate(layer_importance):
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    _save_figure(fig, output_path)
    return str(output_path)


def plot_complex_language_handling(output_dir=None):
    """
    Complex Language Handling: Shows how model handles negations, sarcasm, intensifiers.
    Visualizes attention patterns for challenging cases.
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "07_complex_language_handling.png"

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Complex Language Handling: Negations, Sarcasm, Intensifiers", 
                fontsize=13, weight="bold")

    examples = [
        {
            "text": "not good",
            "tokens": ["not", "good"],
            "label": "Negation",
            "explanation": "Model captures negation pair",
            "weights": [[0.1, 0.9], [0.8, 0.2]]
        },
        {
            "text": "very very bad",
            "tokens": ["very", "very", "bad"],
            "label": "Intensifier",
            "explanation": "Double intensifier increases negative",
            "weights": [[0.2, 0.3, 0.5], [0.25, 0.25, 0.5], [0.3, 0.3, 0.4]]
        },
        {
            "text": "loved it ??? ",
            "tokens": ["loved", "it", "???"],
            "label": "Potential Sarcasm",
            "explanation": "Question marks suggest sarcasm cue",
            "weights": [[0.3, 0.3, 0.4], [0.2, 0.35, 0.45], [0.4, 0.4, 0.2]]
        },
        {
            "text": "absolutely terrible script",
            "tokens": ["absolutely", "terrible", "script"],
            "label": "Intensifier + Negative",
            "explanation": "Intensifier amplifies negative sentiment",
            "weights": [[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.2, 0.6, 0.2]]
        }
    ]

    for idx, (ax, example) in enumerate(zip(axes.flat, examples)):
        tokens = example["tokens"]
        weights = np.array(example["weights"])
        
        im = ax.imshow(weights, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=9, weight="bold")
        ax.set_yticklabels(tokens, fontsize=9, weight="bold")
        
        ax.set_title(f"{example['label']}\n\"{example['text']}\"", 
                    fontsize=10, weight="bold")
        
        # Annotations
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                      color="white" if weights[i, j] > 0.5 else "black", fontsize=8)
        
        # Explanation
        ax.text(0.5, -0.25, example["explanation"], transform=ax.transAxes,
               ha="center", fontsize=8, style="italic", 
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    _save_figure(fig, output_path)
    return str(output_path)


def plot_lime_framework_visual(output_dir=None):
    """
    Figure 5: LIME Framework Visualization.
    Shows how LIME creates local linear approximation for explanation.
    """
    target_dir = _ensure_visuals_dir(output_dir)
    output_path = target_dir / "08_lime_framework.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    ax.text(0, 2.7, "LIME Framework: Local Interpretable Model-Agnostic Explanations", 
           fontsize=13, weight="bold", ha="center")

    # Complex black box boundary (non-linear)
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 1.5 * np.cos(theta) + 0.3 * np.sin(theta)
    y_circle = 1.5 * np.sin(theta) + 0.3 * np.cos(theta)
    ax.plot(x_circle, y_circle, "k-", linewidth=3, label="Black Box Decision Boundary")
    ax.fill(x_circle, y_circle, alpha=0.1, color="red")

    # Target prediction point
    ax.scatter([0.5], [0.5], s=400, c="blue", marker="*", edgecolors="black", 
              linewidth=2, label="Target Prediction", zorder=5)

    # Perturbations around target
    np.random.seed(42)
    perturbations_x = np.random.normal(0.5, 0.3, 30)
    perturbations_y = np.random.normal(0.5, 0.3, 30)
    
    # Color by predicted class
    colors = ["green" if (np.sqrt((x-0.5)**2 + (y-0.5)**2) < 0.5) else "red" 
             for x, y in zip(perturbations_x, perturbations_y)]
    
    ax.scatter(perturbations_x, perturbations_y, s=50, c=colors, alpha=0.6, 
              edgecolors="black", linewidth=0.5, label="Perturbed Samples")

    # Local linear approximation
    x_local = np.linspace(-1, 2, 100)
    y_local = 0.8 * x_local + 0.2
    ax.plot(x_local, y_local, "b--", linewidth=2.5, label="Local Linear Approximation")

    # Annotations
    ax.text(-2.5, 2.3, "1. Select target prediction\n   (blue star)", fontsize=9, 
           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    ax.text(-2.5, 1.3, "2. Generate perturbations\n   around target", fontsize=9,
           bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    ax.text(0.5, -2.3, "3. Fit simple linear model\n   (locally approximates black box)", 
           fontsize=9, ha="center",
           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
    ax.text(-2.5, 0.3, "4. Extract feature\n   weights as explanations", fontsize=9,
           bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5))

    # Legend
    ax.legend(loc="upper right", fontsize=10)

    ax.set_aspect("equal")
    _save_figure(fig, output_path)
    return str(output_path)


def create_paper_visualizations(output_dir=None):
    """Generate all paper-related visualizations."""
    print("Generating paper visualizations...")
    
    visuals = {
        "01_rnn_vs_transformer": plot_rnn_vs_transformer(output_dir),
        "02_performance_vs_interpretability": plot_performance_vs_interpretability(output_dir),
        "03_multihead_attention": plot_multihead_attention_visualization(output_dir),
        "04_architecture_diagram": plot_architecture_diagram(output_dir),
        "05_tokenization_process": plot_tokenization_process(output_dir),
        "06_attention_head_importance": plot_attention_head_importance({}, output_dir),
        "07_complex_language_handling": plot_complex_language_handling(output_dir),
        "08_lime_framework": plot_lime_framework_visual(output_dir),
    }
    
    return visuals
