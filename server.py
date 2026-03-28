from flask import Flask, jsonify, request, send_from_directory

from model.model import analyze_review

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

    if not isinstance(review, str) or not review.strip():
        return (
            jsonify(
                {
                    "error": "Invalid input. Provide a non-empty 'review' string in JSON body.",
                }
            ),
            400,
        )

    result = analyze_review(review.strip())
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
