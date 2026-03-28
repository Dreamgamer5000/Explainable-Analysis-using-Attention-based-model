from flask import Flask, jsonify, request, send_from_directory

from model.model import analyze_review, analyze_scatter

app = Flask(__name__, static_folder="frontend", static_url_path="/frontend")


@app.get("/")
def frontend_index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health_check():
    return jsonify({"status": "ok"}), 200


@app.post("/api/analyze")
def analyze_movie_review():
    payload = request.get_json(silent=True) or {}
    review = payload.get("review")
    explain_method = payload.get("explain_method", "auto")

    if not isinstance(review, str) or not review.strip():
        return (
            jsonify({"error": "Invalid input. Provide a non-empty 'review' string in JSON body."}),
            400,
        )

    if explain_method not in {"auto", "lime", "shap"}:
        return (
            jsonify({"error": "Invalid explain_method. Use one of: auto, lime, shap."}),
            400,
        )

    result = analyze_review(review.strip(), explain_method=explain_method)
    return jsonify(result), 200


@app.post("/api/scatter")
def scatter_word_analysis():
    payload = request.get_json(silent=True) or {}
    reviews = payload.get("reviews")

    if not isinstance(reviews, list) or len(reviews) < 2:
        return (
            jsonify({"error": "Provide a 'reviews' array with at least 2 entries."}),
            400,
        )

    data = analyze_scatter(reviews)
    return jsonify({"scatter": data}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
