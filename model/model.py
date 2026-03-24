from transformers import pipeline

analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None 
)

def format_scores(scores):
    pos_score = next(item['score'] for item in scores if item['label'] == 'POSITIVE')
    neg_score = next(item['score'] for item in scores if item['label'] == 'NEGATIVE')
    return pos_score, neg_score

def analyze_full_and_word_wise(text):
    print(f"\n{'='*50}")
    print(f"REVIEW: '{text}'")
    print(f"{'='*50}")

    overall_result = analyzer(text)[0] 
    pos_prob, neg_prob = format_scores(overall_result)
    
    print("--- OVERALL PROBABILITY ---")
    print(f"Positive: {pos_prob * 100:.1f}%")
    print(f"Negative: {neg_prob * 100:.1f}%\n")

    print("--- WORD-BY-WORD PROBABILITY ---")
    words = text.split()
    
    print(f"{'Word':<15} | {'Positive %':<12} | {'Negative %':<12}")
    print("-" * 45)
    
    for word in words:
        clean_word = word.strip(".,!?\"'")
        if not clean_word:
            continue
            
        word_result = analyzer(clean_word)[0]
        w_pos, w_neg = format_scores(word_result)
        
        print(f"{clean_word:<15} | {w_pos * 100:>9.1f}% | {w_neg * 100:>9.1f}%")

test_reviews = [
    "An absolute triumph of cinema. The acting was brilliant, and the visuals were stunning.",
    "A complete waste of time. The plot was poorly written, and the dialogue felt incredibly wooden.",
    "The special effects were amazing, but the storyline was terribly boring and dragged on forever.",
    "Absolutely loved it! Highly recommended.",
    "Oh sure, because watching paint dry is exactly what I wanted to pay twenty dollars for. Absolute garbage.",
    "The movie was okay, but the villain was completely terrifying and awesome."
]

for review in test_reviews:
    analyze_full_and_word_wise(review)
