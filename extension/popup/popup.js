// popup/popup.js
(async () => {
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');

  // Get the URL of the active tab
  const [{ url }] = await chrome.tabs.query({ active: true, currentWindow: true });

  const apiUrl = 'http://127.0.0.1:8000/api/check'; // change to production when deployed
  try {
    const resp = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json(); // {verdict, score, stage, reasons, latency_ms, note}

    // Render result
    statusEl.textContent = data.verdict.toUpperCase();
    statusEl.style.color = data.verdict === 'phishing' ? '#ff4a4a' : '#4caf50';
    resultEl.innerHTML = `
      <strong>Score:</strong> ${(data.score * 100).toFixed(1)} %<br>
      <strong>Stage:</strong> ${data.stage}<br>
      <strong>Reasons:</strong> ${data.reasons.join(', ') || '‑'}<br>
      <small>latency ${data.latency_ms} ms</small>`;

    // Send to content script for optional in‑page banner
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, {
      type: 'SHOW_RESULT',
      payload: { verdict: data.verdict, score: data.score }
    });
  } catch (e) {
    statusEl.textContent = 'ERROR';
    statusEl.style.color = '#d32f2f';
    resultEl.textContent = e.message;
  }
})();
