# Movie Review Sentiment Analyzer

This project is a Flask web app that analyzes movie reviews and shows overall sentiment plus token-level sentiment scores.

The model used is `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face Transformers.

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
