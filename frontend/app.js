const reviewInput = document.getElementById("reviewInput");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");
const resultSection = document.getElementById("resultSection");
const positiveScore = document.getElementById("positiveScore");
const negativeScore = document.getElementById("negativeScore");
const tokenOutput = document.getElementById("tokenOutput");

function setStatus(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.style.color = isError ? "#b91c1c" : "#374151";
}

function asPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

function tokenTextColor(positiveProbability, negativeProbability) {
    const score = positiveProbability - negativeProbability;
    const normalized = (score + 1) / 2;
    const hue = normalized * 120;
    return `hsl(${hue}, 78%, 38%)`;
}

function cleanTokenForMatch(text) {
    return text.replace(/^[.,!?"'()\[\]{}]+|[.,!?"'()\[\]{}]+$/g, "");
}

function renderTokens(tokens, reviewText) {
    tokenOutput.innerHTML = "";
    const fragments = reviewText.split(/(\s+)/);
    let tokenIndex = 0;

    for (const fragment of fragments) {
        const element = document.createElement("span");
        element.className = "token-word";
        element.textContent = fragment;

        const cleanedFragment = cleanTokenForMatch(fragment);
        const currentToken = tokens[tokenIndex];

        if (cleanedFragment && currentToken && cleanedFragment === currentToken.token) {
            element.style.color = tokenTextColor(
                currentToken.positive_probability,
                currentToken.negative_probability
            );
            tokenIndex += 1;
        }

        tokenOutput.appendChild(element);
    }
}

async function analyzeReview() {
    const review = reviewInput.value.trim();

    if (!review) {
        resultSection.classList.add("hidden");
        setStatus("Please enter a movie review first.", true);
        return;
    }

    analyzeButton.disabled = true;
    setStatus("Analyzing review...");

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Failed to analyze review.");
        }

        positiveScore.textContent = asPercent(data.overall.positive_probability);
        negativeScore.textContent = asPercent(data.overall.negative_probability);
        renderTokens(data.tokens, review);

        resultSection.classList.remove("hidden");
        setStatus("Analysis complete.");
    } catch (error) {
        resultSection.classList.add("hidden");
        setStatus(error.message, true);
    } finally {
        analyzeButton.disabled = false;
    }
}

analyzeButton.addEventListener("click", analyzeReview);
reviewInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        analyzeReview();
    }
});
