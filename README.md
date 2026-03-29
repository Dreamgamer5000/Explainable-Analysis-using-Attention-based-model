# Movie Review Sentiment Analyzer

This project is a Flask web app that analyzes movie reviews and shows overall sentiment plus token-level sentiment scores.

1. Overall sentiment probabilities.
2. Token-level sentiment probabilities.
3. Sentence-by-sentence trajectory.
4. Aspect radar scoring.

The base model is Hugging Face DistilBERT sentiment:
`distilbert-base-uncased-finetuned-sst-2-english`.

## Design Decisions: Hardcoded Words

The project uses two sets of hardcoded word lists for efficiency and interpretability:

1. **Stopwords** — Common English words (via scikit-learn) plus domain-specific terms like "film", "movie", "watch". These are filtered during word frequency analysis in `/api/scatter` so that only meaningful, sentiment-bearing words are counted.

2. **Aspect Keywords** — Domain-specific keywords map sentences to cinematic aspects (Acting, Directing, Plot, Cinematography, Soundtrack). This keyword-based approach is lightweight, requires no pre-training, and produces deterministic aspect detection without downloading a 1.6GB model.

Both lists are intentionally kept simple and malleable in the code so you can edit them based on your use case.

## Setup and Run

1. Create a virtual environment:

```bash
python3 -m venv .venv
```

2. Activate the virtual environment (fish shell):

```bash
source .venv/bin/activate.fish
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Start the server:

```bash
python server.py
```

5. Open the app in your browser:

```bash
http://127.0.0.1:5000/
```
