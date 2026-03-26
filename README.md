# Explainable-Analysis-using-Attention-based-model

## Installation

Open your terminal or command prompt and run the following command to install the required libraries:

```bash
pip install transformers torch flask
```


```bash
python server.py
```

## API

### Health check

```bash
curl http://127.0.0.1:5000/health
```

### Analyze movie review

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
	-H "Content-Type: application/json" \
	-d '{"review":"The movie was amazing but a bit too long."}'
```

### Response shape

```json
{
	"review": "The movie was amazing but a bit too long.",
	"overall": {
		"positive_probability": 0.999,
		"negative_probability": 0.001
	},
	"tokens": [
		{
			"token": "movie",
			"positive_probability": 0.95,
			"negative_probability": 0.05
		}
	]
}
```
