from transformers import pipeline

print("Loading Original Model...")
original_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    top_k=None
)

print("Loading Custom IMDB Model...")
custom_analyzer = pipeline(
    "sentiment-analysis", 
    model="./my_custom_imdb_model", 
    top_k=None
)

def get_positive_score(scores):
    """Helper function to grab just the POSITIVE percentage."""
    return next(item['score'] for item in scores if item['label'] == 'POSITIVE')

def compare_models(text):
    print(f"\n{'='*60}")
    print(f"REVIEW: '{text}'")
    print(f"{'='*60}")

    # --- 1. OVERALL COMPARISON ---
    orig_scores = original_analyzer(text)[0]
    cust_scores = custom_analyzer(text)[0]
    
    orig_pos = get_positive_score(orig_scores)
    cust_pos = get_positive_score(cust_scores)
    
    print("--- OVERALL POSITIVE PROBABILITY ---")
    print(f"Original Model : {orig_pos * 100:>6.1f}%")
    print(f"Custom Model   : {cust_pos * 100:>6.1f}%\n")

    # --- 2. WORD-BY-WORD COMPARISON ---
    print("--- WORD-BY-WORD POSITIVE PROBABILITY ---")
    print(f"{'Word':<15} | {'Original %':<12} | {'Custom %':<12}")
    print("-" * 45)
    
    words = text.split()
    for word in words:
        clean_word = word.strip(".,!?\"'")
        if not clean_word:
            continue
            
        w_orig = get_positive_score(original_analyzer(clean_word)[0])
        w_cust = get_positive_score(custom_analyzer(clean_word)[0])
        
        print(f"{clean_word:<15} | {w_orig * 100:>9.1f}% | {w_cust * 100:>9.1f}%")

# --- Run the Test ---
# Let's use a tricky, nuanced review to really test the difference!
test_reviews = [
    "The pacing was a bit slow, but the cinematic shots were absolutely gorgeous.",
    "Oh sure, because watching paint dry is exactly what I wanted to pay twenty dollars for."
]

for review in test_reviews:
    compare_models(review)
