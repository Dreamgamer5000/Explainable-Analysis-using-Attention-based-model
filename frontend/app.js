// ── DOM refs ──────────────────────────────────────────────────────────────────
const reviewInput = document.getElementById("reviewInput");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");
const resultSection = document.getElementById("resultSection");
const explainMethod = document.getElementById("explainMethod");
const tokenViewMode = document.getElementById("tokenViewMode");
const positiveScore = document.getElementById("positiveScore");
const negativeScore = document.getElementById("negativeScore");
const tokenOutput = document.getElementById("tokenOutput");
const tokenModeNote = document.getElementById("tokenModeNote");
const explanationSummary = document.getElementById("explanationSummary");
const summaryMethodLabel = document.getElementById("summaryMethodLabel");
const topPositiveWords = document.getElementById("topPositiveWords");
const topNegativeWords = document.getElementById("topNegativeWords");
const tokenTooltip = document.getElementById("tokenTooltip");
const tooltipToken = document.getElementById("tooltipToken");
const tooltipPositive = document.getElementById("tooltipPositive");
const tooltipNegative = document.getElementById("tooltipNegative");
const tooltipAttribution = document.getElementById("tooltipAttribution");

const trajectoryBlock = document.getElementById("trajectoryBlock");
const aspectsBlock = document.getElementById("aspectsBlock");
const scatterChartWrap = document.getElementById("scatterChartWrap");

let activeTokenElement = null;
let latestAnalyzePayload = null;

// ── Chart.js defaults ─────────────────────────────────────────────────────────
Chart.defaults.color = "#9ca7c3";
Chart.defaults.font.family = "Inter, Segoe UI, Arial, sans-serif";

// Chart instances (kept so we can destroy before re-creating)
let trajectoryChartInstance = null;
let aspectsChartInstance = null;
let scatterChartInstance = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(message, type = "info") {
    statusMessage.textContent = message;
    statusMessage.classList.remove("is-error", "is-success");
    if (type === "error") statusMessage.classList.add("is-error");
    if (type === "success") statusMessage.classList.add("is-success");
}

function setScatterStatus(message, type = "info") {
    const el = document.getElementById("scatterStatus");
    el.textContent = message;
    el.classList.remove("is-error", "is-success");
    if (type === "error") el.classList.add("is-error");
    if (type === "success") el.classList.add("is-success");
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

function attributionTextColor(normalizedAttribution) {
    const value = Number(normalizedAttribution || 0);
    const intensity = Math.min(1, Math.abs(value));
    if (value === 0) return "hsl(220, 10%, 62%)";
    const hue = value > 0 ? 142 : 352;
    const saturation = 68 + intensity * 18;
    const lightness = 56 - intensity * 20;
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function cleanTokenForMatch(text) {
    return text.replace(/^[.,!?"'()\[\]{}]+|[.,!?"'()\[\]{}]+$/g, "");
}

// ── Token rendering ───────────────────────────────────────────────────────────
function renderTokens(tokens, reviewText, explanation, mode) {
    tokenOutput.innerHTML = "";
    const fragments = reviewText.split(/(\s+)/);
    const fragmentNode = document.createDocumentFragment();
    const explanationTokens = explanation && Array.isArray(explanation.tokens) ? explanation.tokens : [];
    let tokenIndex = 0;

    for (const fragment of fragments) {
        const element = document.createElement("span");
        element.className = "token-word";
        element.textContent = fragment;

        const cleanedFragment = cleanTokenForMatch(fragment);
        const currentToken = tokens[tokenIndex];

        if (cleanedFragment && currentToken && cleanedFragment === currentToken.token) {
            const explanationToken = explanationTokens[tokenIndex];
            const hasAttribution = explanationToken && Number.isFinite(Number(explanationToken.normalized_attribution));

            if (mode === "attribution" && hasAttribution) {
                element.style.color = attributionTextColor(explanationToken.normalized_attribution);
                const sign = Number(explanationToken.normalized_attribution) > 0 ? "influence-positive" : "influence-negative";
                if (Number(explanationToken.normalized_attribution) !== 0) {
                    element.classList.add(sign);
                }
            } else {
                element.style.color = tokenTextColor(
                    currentToken.positive_probability,
                    currentToken.negative_probability
                );
            }

            element.classList.add("has-score");
            element.dataset.token = currentToken.token;
            element.dataset.positive = String(currentToken.positive_probability);
            element.dataset.negative = String(currentToken.negative_probability);
            element.dataset.attribution = String(explanationToken ? explanationToken.attribution : 0);
            element.dataset.normalizedAttribution = String(explanationToken ? explanationToken.normalized_attribution : 0);
            element.dataset.explainMethod = explanation && explanation.method ? explanation.method : "unavailable";
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
        setTimeout(() => element.classList.remove("score-pop"), 200);
    });
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function showTooltip(target) {
    const token = target.dataset.token;
    const positive = Number(target.dataset.positive || 0);
    const negative = Number(target.dataset.negative || 0);
    const attribution = Number(target.dataset.attribution || 0);
    const explainMethod = target.dataset.explainMethod || "unavailable";

    tooltipToken.textContent = token;
    tooltipPositive.textContent = `Positive: ${asPercent(positive)}`;
    tooltipNegative.textContent = `Negative: ${asPercent(negative)}`;
    if (explainMethod !== "unavailable") {
        const sign = attribution > 0 ? "supports positive" : attribution < 0 ? "supports negative" : "neutral";
        tooltipAttribution.textContent = `Attribution (${explainMethod.toUpperCase()}): ${attribution.toFixed(4)} (${sign})`;
        tooltipAttribution.classList.remove("hidden");
    } else {
        tooltipAttribution.classList.add("hidden");
    }
    tokenTooltip.classList.remove("hidden");
    tokenTooltip.classList.add("show");
    tokenTooltip.setAttribute("aria-hidden", "false");
}

function renderSummaryList(container, words, emptyText) {
    container.innerHTML = "";
    if (!words || words.length === 0) {
        const li = document.createElement("li");
        li.className = "summary-empty";
        li.textContent = emptyText;
        container.appendChild(li);
        return;
    }

    for (const word of words) {
        const li = document.createElement("li");
        const token = document.createElement("span");
        token.textContent = word.token;
        const score = document.createElement("span");
        score.textContent = Number(word.score).toFixed(4);
        li.appendChild(token);
        li.appendChild(score);
        container.appendChild(li);
    }
}

function renderExplanationSummary(explanation) {
    if (!explanation || explanation.method === "unavailable") {
        explanationSummary.classList.add("hidden");
        return;
    }

    explanationSummary.classList.remove("hidden");
    summaryMethodLabel.textContent = `Method: ${explanation.method.toUpperCase()} | Predicted class: ${explanation.class_label} (${asPercent(explanation.class_probability)})`;
    renderSummaryList(topPositiveWords, explanation.summary.top_positive, "No strongly positive contributing words.");
    renderSummaryList(topNegativeWords, explanation.summary.top_negative, "No strongly negative contributing words.");
}

function renderCurrentTokenView() {
    if (!latestAnalyzePayload) return;
    renderTokens(
        latestAnalyzePayload.tokens,
        latestAnalyzePayload.review,
        latestAnalyzePayload.explanation,
        tokenViewMode.value
    );
    if (tokenViewMode.value === "attribution") {
        tokenModeNote.textContent = "Attribution mode is active. Word colors reflect contribution strength to the predicted class.";
    } else {
        tokenModeNote.textContent = "Probability mode is active. Word colors reflect standalone token sentiment probabilities.";
    }
}

function hideTooltip() {
    tokenTooltip.classList.remove("show");
    tokenTooltip.setAttribute("aria-hidden", "true");
    activeTokenElement = null;
    setTimeout(() => {
        if (!activeTokenElement) tokenTooltip.classList.add("hidden");
    }, 120);
}

function positionTooltip(event) {
    const viewportPadding = 12;
    const tooltipRect = tokenTooltip.getBoundingClientRect();

    let left = event.clientX;
    let top = event.clientY - 16;

    const minLeft = viewportPadding + tooltipRect.width / 2;
    const maxLeft = window.innerWidth - viewportPadding - tooltipRect.width / 2;

    if (left < minLeft) left = minLeft;
    if (left > maxLeft) left = maxLeft;
    if (top - tooltipRect.height < viewportPadding) top = event.clientY + tooltipRect.height + 16;

    tokenTooltip.style.left = `${left}px`;
    tokenTooltip.style.top = `${top}px`;
}

// ── Chart: Sentiment Trajectory ───────────────────────────────────────────────
function renderTrajectoryChart(trajectory) {
    if (!trajectory || trajectory.length < 2) {
        trajectoryBlock.classList.add("hidden");
        return;
    }

    trajectoryBlock.classList.remove("hidden");

    if (trajectoryChartInstance) trajectoryChartInstance.destroy();

    const labels = trajectory.map((_, i) => `S${i + 1}`);
    const scores = trajectory.map(d => d.score);

    const ctx = document.getElementById("trajectoryChart").getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, "rgba(54, 217, 123, 0.28)");
    gradient.addColorStop(0.5, "rgba(79, 140, 255, 0.08)");
    gradient.addColorStop(1, "rgba(255, 111, 141, 0.28)");

    trajectoryChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [{
                data: scores,
                borderColor: "#4f8cff",
                backgroundColor: gradient,
                borderWidth: 2.5,
                pointBackgroundColor: scores.map(s => s >= 0 ? "#36d97b" : "#ff6f8d"),
                pointBorderColor: "transparent",
                pointRadius: 6,
                pointHoverRadius: 8,
                tension: 0.4,
                fill: true,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const d = trajectory[items[0].dataIndex];
                            return d.sentence.length > 70
                                ? d.sentence.slice(0, 70) + "…"
                                : d.sentence;
                        },
                        label: (item) => `Score: ${item.raw.toFixed(3)}`,
                    },
                    titleFont: { size: 12 },
                    padding: 10,
                },
            },
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    grid: { color: "rgba(255,255,255,0.07)" },
                    ticks: {
                        color: "#9ca7c3",
                        callback: (v) => v.toFixed(1),
                        stepSize: 0.5,
                    },
                },
                x: {
                    grid: { color: "rgba(255,255,255,0.07)" },
                    ticks: { color: "#9ca7c3" },
                },
            },
        },
    });
}

// ── Chart: Aspect Radar ───────────────────────────────────────────────────────
function renderAspectsChart(aspects) {
    const labels = Object.keys(aspects);
    const hasAny = labels.some(k => aspects[k] !== null);

    if (!hasAny) {
        aspectsBlock.classList.add("hidden");
        return;
    }

    aspectsBlock.classList.remove("hidden");

    if (aspectsChartInstance) aspectsChartInstance.destroy();

    const data = labels.map(k => aspects[k] !== null ? aspects[k] : 0);
    const hasData = labels.map(k => aspects[k] !== null);

    const ctx = document.getElementById("aspectsChart").getContext("2d");

    aspectsChartInstance = new Chart(ctx, {
        type: "radar",
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: "rgba(79, 140, 255, 0.15)",
                borderColor: "#4f8cff",
                borderWidth: 2,
                pointBackgroundColor: data.map((v, i) =>
                    !hasData[i] ? "#4a5068" : v >= 0 ? "#36d97b" : "#ff6f8d"
                ),
                pointBorderColor: "transparent",
                pointRadius: 5,
                pointHoverRadius: 7,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => {
                            if (!hasData[item.dataIndex]) return "No mentions detected";
                            return `Score: ${item.raw.toFixed(3)}`;
                        },
                    },
                    padding: 10,
                },
            },
            scales: {
                r: {
                    min: -1,
                    max: 1,
                    grid: { color: "rgba(255,255,255,0.1)" },
                    angleLines: { color: "rgba(255,255,255,0.1)" },
                    pointLabels: { color: "#e6e9f2", font: { size: 13, weight: "600" } },
                    ticks: {
                        backdropColor: "transparent",
                        color: "#9ca7c3",
                        stepSize: 0.5,
                        callback: (v) => v.toFixed(1),
                    },
                },
            },
        },
    });
}

// ── Chart: Polarized Word Scatter ─────────────────────────────────────────────
function renderScatterChart(scatter) {
    if (!scatter || scatter.length === 0) {
        setScatterStatus("No word data to display.", "error");
        return;
    }

    scatterChartWrap.classList.remove("hidden");

    if (scatterChartInstance) scatterChartInstance.destroy();

    // Color each point by how positive vs negative it is
    const pointColors = scatter.map(d => {
        const total = d.positive_freq + d.negative_freq;
        const posRatio = d.positive_freq / total;
        const r = Math.round(255 * (1 - posRatio));
        const g = Math.round(217 * posRatio);
        const b = Math.round(123 * posRatio + 141 * (1 - posRatio));
        return `rgba(${r}, ${g}, ${b}, 0.80)`;
    });

    // Scale point radius by total frequency
    const maxTotal = Math.max(...scatter.map(d => d.positive_freq + d.negative_freq));
    const pointRadii = scatter.map(d => {
        const total = d.positive_freq + d.negative_freq;
        return Math.max(4, Math.min(14, 4 + (total / maxTotal) * 10));
    });

    const ctx = document.getElementById("scatterChart").getContext("2d");

    scatterChartInstance = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [{
                data: scatter.map(d => ({
                    x: d.negative_freq,
                    y: d.positive_freq,
                    word: d.word,
                })),
                backgroundColor: pointColors,
                pointRadius: pointRadii,
                pointHoverRadius: pointRadii.map(r => r + 3),
                pointBorderColor: "transparent",
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => {
                            const d = item.raw;
                            return [
                                `"${d.word}"`,
                                `Positive reviews: ${d.y}`,
                                `Negative reviews: ${d.x}`,
                            ];
                        },
                        title: () => "",
                    },
                    padding: 10,
                    displayColors: false,
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Frequency in Negative reviews →",
                        color: "#ff6f8d",
                        font: { size: 12, weight: "600" },
                    },
                    grid: { color: "rgba(255,255,255,0.07)" },
                    ticks: { color: "#9ca7c3" },
                    min: 0,
                },
                y: {
                    title: {
                        display: true,
                        text: "Frequency in Positive reviews →",
                        color: "#36d97b",
                        font: { size: 12, weight: "600" },
                    },
                    grid: { color: "rgba(255,255,255,0.07)" },
                    ticks: { color: "#9ca7c3" },
                    min: 0,
                },
            },
        },
    });
}

// ── Main analyze flow ─────────────────────────────────────────────────────────
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
    setStatus("Analyzing review…", "info");

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                review,
                explain_method: explainMethod.value,
            }),
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Failed to analyze review.");

        positiveScore.textContent = asPercent(data.overall.positive_probability);
        negativeScore.textContent = asPercent(data.overall.negative_probability);
        animateScoreValue(positiveScore);
        animateScoreValue(negativeScore);
        latestAnalyzePayload = {
            review,
            tokens: data.tokens,
            explanation: data.explanation,
        };
        renderCurrentTokenView();
        renderExplanationSummary(data.explanation);
        renderTrajectoryChart(data.trajectory);
        renderAspectsChart(data.aspects);

        resultSection.classList.remove("hidden");
        requestAnimationFrame(() => resultSection.classList.add("is-visible"));

        setStatus("Analysis complete.", "success");
    } catch (error) {
        resultSection.classList.add("hidden");
        resultSection.classList.remove("is-visible");
        latestAnalyzePayload = null;
        explanationSummary.classList.add("hidden");
        hideTooltip();
        setStatus(error.message, "error");
    } finally {
        analyzeButton.disabled = false;
    }
}

// ── Scatter batch analyze ─────────────────────────────────────────────────────
async function analyzeScatter() {
    const text = document.getElementById("scatterInput").value.trim();

    if (!text) {
        setScatterStatus("Please paste at least two reviews separated by ---.", "error");
        return;
    }

    const reviews = text.split(/---/).map(r => r.trim()).filter(Boolean);

    if (reviews.length < 2) {
        setScatterStatus("Separate reviews with --- on its own line. Need at least 2 reviews.", "error");
        return;
    }

    const scatterButton = document.getElementById("scatterButton");
    scatterButton.disabled = true;
    setScatterStatus(`Analyzing ${reviews.length} reviews…`, "info");

    try {
        const response = await fetch("/api/scatter", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reviews }),
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Failed to analyze batch.");

        renderScatterChart(data.scatter);
        setScatterStatus(
            `Plotted ${data.scatter.length} words from ${reviews.length} reviews. Hover points for details.`,
            "success"
        );
    } catch (error) {
        setScatterStatus(error.message, "error");
    } finally {
        scatterButton.disabled = false;
    }
}

// ── Event listeners ───────────────────────────────────────────────────────────
analyzeButton.addEventListener("click", analyzeReview);
document.getElementById("scatterButton").addEventListener("click", analyzeScatter);

tokenOutput.addEventListener("mouseover", (event) => {
    const target = event.target.closest(".token-word.has-score");
    if (!target || target === activeTokenElement) return;
    activeTokenElement = target;
    showTooltip(target);
});

tokenOutput.addEventListener("mousemove", (event) => {
    if (!activeTokenElement) return;
    positionTooltip(event);
});

tokenOutput.addEventListener("mouseout", (event) => {
    if (!activeTokenElement) return;
    const nextTarget = event.relatedTarget;
    if (nextTarget && nextTarget.closest(".token-word.has-score") === activeTokenElement) return;
    hideTooltip();
});

reviewInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") analyzeReview();
});

tokenViewMode.addEventListener("change", () => {
    renderCurrentTokenView();
});
