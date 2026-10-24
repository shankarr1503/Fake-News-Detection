// Same origin when served by the API (http://localhost:8000/); fall back to the
// local API when the page is opened directly from disk (needs CORS_ORIGINS=null on the API).
const API_URL = location.protocol === 'file:' ? 'http://localhost:8000/predict' : '/predict';

// Give up on a request after this long instead of leaving the page stuck.
const REQUEST_TIMEOUT_MS = 60_000;
// Below this, a prediction is barely better than a coin flip.
const LOW_CONFIDENCE = 0.6;

const form = document.getElementById('news-form');
const titleInput = document.getElementById('title');
const contentInput = document.getElementById('content');
const checkBtn = document.getElementById('check-btn');
const loader = document.getElementById('loader');
const statusEl = document.getElementById('status');
const errorBanner = document.getElementById('error-banner');
const resultsPanel = document.getElementById('results-panel');
const summaryEl = document.getElementById('summary');
const scene = document.getElementById('scene');
const stampText = document.getElementById('stamp-text');
const methodNumber = document.getElementById('method-number');
const modelsEl = document.querySelector('.models');
const hero = document.getElementById('hero');
const root = document.documentElement;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const fields = [
    {
        input: titleInput,
        name: 'title',
        label: 'headline',
        countEl: document.getElementById('title-count'),
        errorEl: document.getElementById('title-error'),
    },
    {
        input: contentInput,
        name: 'text',
        label: 'article text',
        countEl: document.getElementById('content-count'),
        errorEl: document.getElementById('content-error'),
    },
];

let inFlight = false;

form.addEventListener('submit', (event) => {
    event.preventDefault();
    analyze();
});

// Ctrl+Enter / Cmd+Enter submits from the textarea, where plain Enter adds a new line.
contentInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        form.requestSubmit();
    }
});

for (const field of fields) {
    field.input.addEventListener('input', () => {
        updateCount(field);
        if (field.input.value.trim()) setFieldError(field, '');
    });
    updateCount(field);
}

setUpTilt();
setUpScrollDepth();
setUpReveal();
setUpCardTilt();

async function analyze() {
    if (inFlight) return;

    const title = titleInput.value.trim();
    const text = contentInput.value.trim();
    if (!validate({ title, text })) return;

    setLoading(true);
    hideError();
    resultsPanel.hidden = true;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
        let response;
        try {
            response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, text }),
                signal: controller.signal,
            });
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('The request timed out. The server may be busy; please try again.');
            }
            throw new Error('Could not reach the API. Check your connection and that the backend is running.');
        }

        const data = await response.json().catch(() => null);
        if (!response.ok) {
            throw new Error(describeHttpError(response, data));
        }
        if (!data || !data.ml || !data.ai) {
            throw new Error('The API returned an unexpected response.');
        }

        // Hide the loader first: removing it shifts the page, which would throw off the scroll to the results.
        setLoading(false);
        showResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        clearTimeout(timeout);
        setLoading(false);
    }
}

function validate(values) {
    let firstInvalid = null;
    for (const field of fields) {
        const missing = !values[field.name];
        setFieldError(field, missing ? `Please enter the ${field.label}.` : '');
        if (missing && !firstInvalid) firstInvalid = field.input;
    }
    if (firstInvalid) firstInvalid.focus();
    return !firstInvalid;
}

function setFieldError(field, message) {
    field.errorEl.textContent = message;
    if (message) {
        field.input.setAttribute('aria-invalid', 'true');
    } else {
        field.input.removeAttribute('aria-invalid');
    }
}

function updateCount(field) {
    const max = field.input.maxLength;
    const length = field.input.value.length;
    field.countEl.textContent = `${length.toLocaleString()} / ${max.toLocaleString()}`;
    field.countEl.classList.toggle('near-limit', length >= max * 0.9);
}

function setLoading(loading) {
    inFlight = loading;
    checkBtn.disabled = loading;
    checkBtn.setAttribute('aria-busy', String(loading));
    loader.hidden = !loading;
    // Only set here; the result or error that follows replaces the message.
    if (loading) {
        statusEl.textContent = 'Analyzing the article…';
        setScene('analyzing');
    }
}

function setScene(state, label = '') {
    scene.dataset.state = state;
    stampText.textContent = label;
}

function showError(message) {
    // The banner is role="alert", so it announces itself.
    statusEl.textContent = '';
    setScene('idle');
    errorBanner.textContent = message;
    errorBanner.hidden = false;
}

function hideError() {
    errorBanner.hidden = true;
    errorBanner.textContent = '';
}

function describeHttpError(response, data) {
    const detail = formatDetail(data);
    switch (response.status) {
        case 413:
            return 'The article is too long for the server to accept. Please shorten it and try again.';
        case 422:
            return `Please check your input: ${detail || 'the server rejected it.'}`;
        case 429:
            return 'Too many requests. Please wait a moment and try again.';
        case 502:
        case 503:
        case 504:
            return 'The service is temporarily unavailable. Please try again shortly.';
        default:
            return `The API returned an error (${response.status}${detail ? `: ${detail}` : ''}).`;
    }
}

function formatDetail(data) {
    if (!data || !data.detail) return '';
    if (typeof data.detail === 'string') return data.detail;
    if (!Array.isArray(data.detail)) return '';
    return data.detail
        .map((d) => {
            const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : '';
            return field && field !== 'body' ? `${field}: ${d.msg}` : d.msg;
        })
        .join('; ');
}

function showResults(data) {
    // Start the bars and towers from zero and force a layout, so they grow to their new size on every result.
    for (const prefix of ['ml', 'ai']) {
        document.getElementById(`${prefix}-bar`).style.width = '0%';
        document.getElementById(`${prefix}-tower`).style.setProperty('--level', '0');
    }
    resultsPanel.hidden = false;
    methodNumber.textContent = '03';
    modelsEl.classList.remove('is-revealing');
    void resultsPanel.offsetWidth;
    modelsEl.classList.add('is-revealing');

    updateResult('ml', data.ml);
    updateResult('ai', data.ai);
    setScene(...stampFor(data.ml, data.ai));
    summaryEl.textContent = summarize(data.ml, data.ai);
    statusEl.textContent = `Results ready. ${summaryEl.textContent}`;
    resultsPanel.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
}

// What the 3D stack's stamp shows: [scene state, stamp text].
function stampFor(ml, ai) {
    const available = [ml, ai].filter((r) => r.confidence !== null);
    if (available.length === 0) return ['none', 'No model'];
    if (available.some((r) => r.label !== available[0].label)) return ['mixed', 'Split'];
    if (available.every((r) => r.confidence < LOW_CONFIDENCE)) return ['mixed', 'Unsure'];
    return available[0].label === 'REAL' ? ['real', 'Real'] : ['fake', 'Fake'];
}

function summarize(ml, ai) {
    const available = [ml, ai].filter((r) => r.confidence !== null);
    if (available.length === 0) {
        return 'No models are loaded on the server, so no prediction could be made.';
    }
    if (available.length === 1) {
        return `Only one model is available. It predicts ${available[0].label}.`;
    }
    if (ml.label === ai.label) {
        return `Both models predict ${ml.label}.`;
    }
    return 'The models disagree. Treat this article with extra care.';
}

function updateResult(prefix, data) {
    const labelEl = document.getElementById(`${prefix}-label`);
    const confEl = document.getElementById(`${prefix}-conf`);
    const meterEl = document.getElementById(`${prefix}-meter`);
    const barEl = document.getElementById(`${prefix}-bar`);
    const lowEl = document.getElementById(`${prefix}-low`);
    const captionEl = document.getElementById(`${prefix}-caption`);
    const towerEl = document.getElementById(`${prefix}-tower`);

    const kind = data.label === 'REAL' ? 'real' : data.label === 'FAKE' ? 'fake' : 'unavailable';
    labelEl.textContent = kind === 'unavailable' ? 'Unavailable' : data.label;
    labelEl.className = `model-verdict ${kind}`;
    towerEl.classList.remove('real', 'fake', 'unavailable');
    towerEl.classList.add(kind);
    towerEl.style.setProperty('--level', data.confidence === null ? '0' : String(data.confidence));
    barEl.className = `meter-fill ${kind}`;

    if (data.confidence === null || kind === 'unavailable') {
        confEl.textContent = 'Model not loaded on the server';
        barEl.style.width = '0%';
        meterEl.setAttribute('aria-valuenow', '0');
        meterEl.hidden = true;
        captionEl.hidden = true;
        lowEl.hidden = true;
        return;
    }

    const percent = Math.round(data.confidence * 1000) / 10;
    confEl.textContent = `${percent.toFixed(1)}%`;
    meterEl.hidden = false;
    captionEl.hidden = false;
    meterEl.setAttribute('aria-valuenow', String(percent));
    barEl.style.width = `${percent}%`;
    lowEl.hidden = data.confidence >= LOW_CONFIDENCE;
}

// Tilt the 3D stack towards the pointer, and share the pointer position with the CSS as --px/--py
// (the headline layers and the grid floor use it). Skipped for touch screens and reduced motion.
function setUpTilt() {
    if (reducedMotion || !window.matchMedia('(pointer: fine)').matches) return;

    let frame = 0;
    window.addEventListener('pointermove', (event) => {
        if (frame) return;
        frame = requestAnimationFrame(() => {
            frame = 0;
            const x = event.clientX / window.innerWidth - 0.5;
            const y = event.clientY / window.innerHeight - 0.5;
            scene.style.setProperty('--rx', `${(-y * 14).toFixed(2)}deg`);
            scene.style.setProperty('--rz', `${(x * 18).toFixed(2)}deg`);
            root.style.setProperty('--px', x.toFixed(3));
            root.style.setProperty('--py', y.toFixed(3));
        });
    });
}

// --scroll goes from 0 to 1 as the hero scrolls out of view; the headline and the stack turn with it.
function setUpScrollDepth() {
    if (reducedMotion) return;

    let frame = 0;
    const update = () => {
        frame = 0;
        const rect = hero.getBoundingClientRect();
        const progress = Math.min(Math.max(-rect.top / rect.height, 0), 1);
        root.style.setProperty('--scroll', progress.toFixed(3));
    };
    window.addEventListener(
        'scroll',
        () => {
            if (!frame) frame = requestAnimationFrame(update);
        },
        { passive: true },
    );
    update();
}

// Sections fold up into place as they scroll into view.
function setUpReveal() {
    const targets = document.querySelectorAll('.section .section-label, .section .section-body');
    if (reducedMotion || !('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    observer.unobserve(entry.target);
                }
            }
        },
        { threshold: 0.12 },
    );
    for (const target of targets) {
        target.classList.add('reveal');
        observer.observe(target);
    }
}

// Result cards lean towards the pointer.
function setUpCardTilt() {
    if (reducedMotion || !window.matchMedia('(pointer: fine)').matches) return;

    for (const card of document.querySelectorAll('.model')) {
        card.addEventListener('pointermove', (event) => {
            const rect = card.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width - 0.5;
            const y = (event.clientY - rect.top) / rect.height - 0.5;
            card.style.setProperty('--card-y', `${(x * 7).toFixed(2)}deg`);
            card.style.setProperty('--card-x', `${(-y * 7).toFixed(2)}deg`);
        });
        card.addEventListener('pointerleave', () => {
            card.style.setProperty('--card-y', '0deg');
            card.style.setProperty('--card-x', '0deg');
        });
    }
}
