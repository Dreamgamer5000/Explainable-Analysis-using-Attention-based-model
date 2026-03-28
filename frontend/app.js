const reviewInput = document.getElementById("reviewInput");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");
const resultSection = document.getElementById("resultSection");
const positiveScore = document.getElementById("positiveScore");
const negativeScore = document.getElementById("negativeScore");
const tokenOutput = document.getElementById("tokenOutput");
const tokenTooltip = document.getElementById("tokenTooltip");
const tooltipToken = document.getElementById("tooltipToken");
const tooltipPositive = document.getElementById("tooltipPositive");
const tooltipNegative = document.getElementById("tooltipNegative");

let activeTokenElement = null;

function setStatus(message, type = "info") {
    statusMessage.textContent = message;
    statusMessage.classList.remove("is-error", "is-success");

    if (type === "error") {
        statusMessage.classList.add("is-error");
    }

    if (type === "success") {
        statusMessage.classList.add("is-success");
    }
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
    const fragmentNode = document.createDocumentFragment();
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
            element.classList.add("has-score");
            element.dataset.token = currentToken.token;
            element.dataset.positive = String(currentToken.positive_probability);
            element.dataset.negative = String(currentToken.negative_probability);
            tokenIndex += 1;
        }

        fragmentNode.appendChild(element);
    }

    tokenOutput.appendChild(fragmentNode);
}

function animateScoreValue(element) {
    element.classList.remove("score-pop");
    requestAnimationFrame(() => {
        element.classList.add("score-pop");
        setTimeout(() => {
            element.classList.remove("score-pop");
        }, 200);
    });
}

function showTooltip(target) {
    const token = target.dataset.token;
    const positive = Number(target.dataset.positive || 0);
    const negative = Number(target.dataset.negative || 0);

    tooltipToken.textContent = token;
    tooltipPositive.textContent = `Positive: ${asPercent(positive)}`;
    tooltipNegative.textContent = `Negative: ${asPercent(negative)}`;
    tokenTooltip.classList.remove("hidden");
    tokenTooltip.classList.add("show");
    tokenTooltip.setAttribute("aria-hidden", "false");
}

function hideTooltip() {
    tokenTooltip.classList.remove("show");
    tokenTooltip.setAttribute("aria-hidden", "true");
    activeTokenElement = null;

    setTimeout(() => {
        if (!activeTokenElement) {
            tokenTooltip.classList.add("hidden");
        }
    }, 120);
}

function positionTooltip(event) {
    const viewportPadding = 12;
    const tooltipRect = tokenTooltip.getBoundingClientRect();

    let left = event.clientX;
    let top = event.clientY - 16;

    const minLeft = viewportPadding + tooltipRect.width / 2;
    const maxLeft = window.innerWidth - viewportPadding - tooltipRect.width / 2;

    if (left < minLeft) {
        left = minLeft;
    }

    if (left > maxLeft) {
        left = maxLeft;
    }

    if (top - tooltipRect.height < viewportPadding) {
        top = event.clientY + tooltipRect.height + 16;
    }

    tokenTooltip.style.left = `${left}px`;
    tokenTooltip.style.top = `${top}px`;
}

async function analyzeReview() {
    const review = reviewInput.value.trim();

    if (!review) {
        resultSection.classList.add("hidden");
        resultSection.classList.remove("is-visible");
        hideTooltip();
        setStatus("Please enter a movie review first.", "error");
        return;
    }

    analyzeButton.disabled = true;
    hideTooltip();
    setStatus("Analyzing review...", "info");

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
        animateScoreValue(positiveScore);
        animateScoreValue(negativeScore);
        renderTokens(data.tokens, review);

        resultSection.classList.remove("hidden");
        requestAnimationFrame(() => {
            resultSection.classList.add("is-visible");
        });

        setStatus("Analysis complete.", "success");
    } catch (error) {
        resultSection.classList.add("hidden");
        resultSection.classList.remove("is-visible");
        hideTooltip();
        setStatus(error.message, "error");
    } finally {
        analyzeButton.disabled = false;
    }
}

analyzeButton.addEventListener("click", analyzeReview);

tokenOutput.addEventListener("mouseover", (event) => {
    const target = event.target.closest(".token-word.has-score");

    if (!target || target === activeTokenElement) {
        return;
    }

    activeTokenElement = target;
    showTooltip(target);
});

tokenOutput.addEventListener("mousemove", (event) => {
    if (!activeTokenElement) {
        return;
    }

    positionTooltip(event);
});

tokenOutput.addEventListener("mouseout", (event) => {
    if (!activeTokenElement) {
        return;
    }

    const nextTarget = event.relatedTarget;
    if (nextTarget && nextTarget.closest(".token-word.has-score") === activeTokenElement) {
        return;
    }

    hideTooltip();
});

reviewInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        analyzeReview();
    }
});
