// Same origin when served by the API (http://localhost:8000/); fall back to the
// local API when the page is opened directly from disk (needs CORS_ORIGINS=null on the API).
const API_URL = location.protocol === 'file:' ? 'http://localhost:8000/predict' : '/predict';

const checkBtn = document.getElementById('check-btn');
const loader = document.getElementById('loader');
const resultsPanel = document.getElementById('results-panel');

checkBtn.addEventListener('click', async () => {
    const title = document.getElementById('title').value.trim();
    const text = document.getElementById('content').value.trim();

    if (!title || !text) {
        alert('Please provide both title and content.');
        return;
    }

    // UI Reset
    resultsPanel.style.display = 'none';
    loader.style.display = 'block';
    checkBtn.disabled = true;

    try {
        let response;
        try {
            response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, text })
            });
        } catch (error) {
            throw new Error('Could not reach the API. Make sure the backend is running.');
        }

        const data = await response.json().catch(() => null);
        if (!response.ok) {
            throw new Error(`The API returned an error (${response.status}): ${formatDetail(data)}`);
        }

        // Update UI
        updateResult('ml', data.ml);
        updateResult('ai', data.ai);
        resultsPanel.style.display = 'grid';
    } catch (error) {
        alert(error.message);
    } finally {
        loader.style.display = 'none';
        checkBtn.disabled = false;
    }
});

function formatDetail(data) {
    if (!data || !data.detail) return 'Unknown error';
    if (typeof data.detail === 'string') return data.detail;
    return data.detail.map(d => `${d.loc.slice(-1)[0]}: ${d.msg}`).join('; ');
}

function updateResult(prefix, data) {
    const labelEl = document.getElementById(`${prefix}-label`);
    const confEl = document.getElementById(`${prefix}-conf`);
    const barEl = document.getElementById(`${prefix}-bar`);

    labelEl.textContent = data.label;

    if (data.confidence === null) {
        labelEl.style.color = 'var(--text-muted)';
        confEl.textContent = 'Model not loaded';
        barEl.style.width = '0%';
        return;
    }

    const color = data.label === 'REAL' ? 'var(--real)' : 'var(--fake)';
    labelEl.style.color = color;

    const confPercent = (data.confidence * 100).toFixed(1);
    confEl.textContent = `Confidence: ${confPercent}%`;
    barEl.style.width = `${confPercent}%`;
    barEl.style.backgroundColor = color;
}
