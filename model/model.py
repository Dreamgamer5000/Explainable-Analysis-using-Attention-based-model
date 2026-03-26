import torch
import plotext as pt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers_interpret import SequenceClassificationExplainer

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Loading Model: {MODEL_NAME}...\n")

# Force the entire script to use the CPU to bypass the Mac MPS bug
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the model, force eager attention for the heatmap, and push to CPU
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    attn_implementation="eager"
).to(device)

# Set up the explainer (it will use the CPU model)
explainer = SequenceClassificationExplainer(model, tokenizer)

# Set up the pipeline, explicitly telling it to use the CPU (device=-1)
analyzer = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer, 
    top_k=None, 
    device=-1
)

# ==========================================
# 2. PROBABILITY TABLE
# ==========================================
def print_probability_table(text):
    def get_scores(scores):
        pos = next(item['score'] for item in scores if item['label'] == 'POSITIVE')
        neg = next(item['score'] for item in scores if item['label'] == 'NEGATIVE')
        return pos, neg

    # --- Overall ---
    overall_result = analyzer(text)[0] 
    o_pos, o_neg = get_scores(overall_result)
    
    print("--- OVERALL PROBABILITY ---")
    print(f"Positive: {o_pos * 100:.1f}%")
    print(f"Negative: {o_neg * 100:.1f}%\n")

    # --- Word-by-word ---
    print("--- WORD-BY-WORD PROBABILITY ---")
    print(f"{'Word':<15} | {'Positive %':<12} | {'Negative %':<12}")
    print("-" * 45)
    
    words = text.split()
    for word in words:
        clean_word = word.strip(".,!?\"'")
        if not clean_word:
            continue
            
        res = analyzer(clean_word)[0]
        w_pos, w_neg = get_scores(res)
        print(f"{clean_word:<15} | {w_pos * 100:>9.1f}% | {w_neg * 100:>9.1f}%")
    print("\n")

# ==========================================
# 3. WORD ATTRIBUTION BAR CHART
# ==========================================
def plot_word_attributions_terminal(text):
    word_attributions = explainer(text)
    
    # Filter out special tokens
    filtered_attrs = [(w, s) for w, s in word_attributions if w not in ["[CLS]", "[SEP]"]]
    words = [item[0] for item in filtered_attrs]
    scores = [item[1] for item in filtered_attrs]
    
    pt.clf() 
    pt.bar(words, scores)
    pt.title("Integrated Gradients: Word Attribution Scores")
    pt.ylabel("Impact Score")
    pt.theme('clear')
    pt.show()
    print("\n")

# ==========================================
# 4. ATTENTION HEATMAP
# ==========================================
def plot_attention_heatmap_terminal(text):
    # Ensure our inputs are also forced to the CPU to match the model
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions[-1] 
    avg_attention = attentions[0].mean(dim=0).cpu().numpy()
    
    matrix_data = avg_attention.tolist()
    
    pt.clf()
    pt.matrix_plot(matrix_data)
    pt.title("Transformer Attention Heatmap (Final Layer)")
    pt.theme('clear')
    pt.show()
    print("\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    test_statement = "The pacing was a bit slow, but the cinematic shots were absolutely gorgeous."
    
    print("="*60)
    print("1. PROBABILITY TABLE")
    print("="*60)
    print_probability_table(test_statement)
    
    print("="*60)
    print("2. WORD ATTRIBUTION BAR CHART")
    print("="*60)
    plot_word_attributions_terminal(test_statement)
    
    print("="*60)
    print("3. ATTENTION HEATMAP")
    print("="*60)
    plot_attention_heatmap_terminal(test_statement)
