#!/usr/bin/env python3
"""
Generate all paper visualizations at once.
Creates 13 comprehensive figures for the research paper.
"""

import sys
from pathlib import Path
import time

# Add model directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))

from model.paper_visualizations import create_paper_visualizations
from model.advanced_paper_visualizations import create_advanced_visualizations


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("RESEARCH PAPER VISUALIZATIONS GENERATOR")
    print("Opening the Black Box: Explainable Sentiment Analysis")
    print("=" * 70)
    print()
    
    output_dir = Path(__file__).resolve().parent / "visuls"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Phase 1: Core paper visualizations
    print("🎨 Phase 1: Core Paper Visualizations")
    print("-" * 70)
    try:
        core_visuals = create_paper_visualizations(output_dir)
        for name, path in core_visuals.items():
            print(f"  ✓ {name}: {path}")
        print(f"  Generated {len(core_visuals)} visualizations")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1
    
    print()
    
    # Phase 2: Advanced visualizations
    print("🎨 Phase 2: Advanced Theory & Metrics")
    print("-" * 70)
    try:
        advanced_visuals = create_advanced_visualizations(output_dir)
        for name, path in advanced_visuals.items():
            print(f"  ✓ {name}: {path}")
        print(f"  Generated {len(advanced_visuals)} visualizations")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1
    
    print()
    
    elapsed_time = time.time() - start_time
    
    print("=" * 70)
    print(f"✓ COMPLETE: {len(core_visuals) + len(advanced_visuals)} visualizations generated")
    print(f"⏱  Time elapsed: {elapsed_time:.2f}s")
    print("=" * 70)
    print()
    
    print("Generated Visualizations:")
    print()
    print("CORE FIGURES (From Paper):")
    print("  01. RNN vs Transformer Architecture")
    print("  02. Performance vs Interpretability Trade-off")
    print("  03. Multi-Head Attention Mechanism")
    print("  04. DistilBERT Architecture Diagram")
    print("  05. WordPiece Tokenization Process")
    print("  06. Attention Head Importance Ranking")
    print("  07. Complex Language Handling (Negation, Sarcasm)")
    print("  08. LIME Framework Visualization")
    print()
    print("ADVANCED VISUALIZATIONS:")
    print("  09. AtteFa Faithfulness Metric Validation")
    print("  10. Global vs Local Interpretability")
    print("  11. SHAP vs LIME Comparison")
    print("  12. Explanation Quality Evaluation Metrics")
    print("  13. Adversarial Robustness Analysis")
    print()
    print(f"Location: {output_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
