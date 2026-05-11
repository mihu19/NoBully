// NoBully Popup Script

const COLORS = {
  safe:    '#00e87a',
  warn:    '#ffb020',
  danger:  '#ff3355',
  muted:   '#4a5270',
};

function getStatusMeta(analysis) {
  if (!analysis) return { label: 'UNKNOWN', color: COLORS.muted, blocked: false };
  if (analysis.should_block || analysis.toxicity_percent >= 65) {
    return { label: 'BLOCKED', color: COLORS.danger, blocked: true };
  }
  if (analysis.toxicity_percent >= 35) {
    return { label: 'CAUTION', color: COLORS.warn, blocked: false };
  }
  return { label: 'SAFE', color: COLORS.safe, blocked: false };
}

function setRing(pct, color) {
  const circumference = 188;
  const offset = circumference - (pct / 100) * circumference;
  const ring = document.getElementById('ringFill');
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = color;
}

function renderAnalysis(analysis, url) {
  // Show results panel
  document.getElementById('loadingState').classList.add('hidden');
  document.getElementById('idleState').classList.add('hidden');
  document.getElementById('resultsView').classList.remove('hidden');

  const meta = getStatusMeta(analysis);

  // Status card accent
  document.getElementById('statusCard').style.setProperty('--status-color', meta.color);
  document.getElementById('statusDot').style.background = meta.color;
  document.getElementById('statusDot').style.boxShadow = `0 0 6px ${meta.color}`;
  document.getElementById('statusText').textContent = meta.label;
  document.getElementById('statusText').style.color = meta.color;

  // URL
  try {
    const u = new URL(url);
    document.getElementById('pageUrl').textContent = u.hostname + u.pathname.slice(0, 20);
  } catch {
    document.getElementById('pageUrl').textContent = url.slice(0, 35);
  }

  // Ring
  const tox = analysis.toxicity_percent || 0;
  document.getElementById('ringPct').textContent = tox + '%';
  setRing(tox, meta.color);

  // Stats
  document.getElementById('severityVal').textContent =
    (analysis.severity_percent !== undefined ? analysis.severity_percent + '%' : '—');
  document.getElementById('badWordVal').textContent =
    (analysis.bad_word_count !== undefined ? analysis.bad_word_count : '—');
  document.getElementById('decisionVal').textContent = meta.blocked ? '🚫 Blocked' : '✅ Allowed';
  document.getElementById('decisionVal').style.color = meta.color;

  // Flagged words
  const container = document.getElementById('flaggedTags');
  container.innerHTML = '';
  const words = analysis.flagged_words || [];
  if (words.length === 0) {
    container.innerHTML = '<span class="no-flags">None detected</span>';
  } else {
    words.slice(0, 12).forEach(w => {
      const tag = document.createElement('span');
      tag.className = 'flagged-tag';
      tag.textContent = w;
      container.appendChild(tag);
    });
    if (words.length > 12) {
      const more = document.createElement('span');
      more.className = 'flagged-tag';
      more.style.background = 'transparent';
      more.style.borderColor = 'transparent';
      more.style.color = 'var(--muted)';
      more.textContent = `+${words.length - 12} more`;
      container.appendChild(more);
    }
  }
}

function showIdle() {
  document.getElementById('loadingState').classList.add('hidden');
  document.getElementById('idleState').classList.remove('hidden');
  document.getElementById('resultsView').classList.add('hidden');
}

// Main init
async function init() {
  // Load enabled state
  chrome.storage.sync.get('extensionEnabled', (res) => {
    const enabled = res.extensionEnabled !== false; // default true
    document.getElementById('enabledToggle').checked = enabled;
    document.getElementById('toggleLabel').textContent = enabled ? 'ON' : 'OFF';
  });

  // Toggle handler
  document.getElementById('enabledToggle').addEventListener('change', (e) => {
    const enabled = e.target.checked;
    chrome.storage.sync.set({ extensionEnabled: enabled });
    document.getElementById('toggleLabel').textContent = enabled ? 'ON' : 'OFF';
  });

  // Get current tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url) { showIdle(); return; }

  const currentUrl = tab.url;

  // Lookup history for this tab
  chrome.storage.local.get('analysisHistory', (result) => {
    const history = result.analysisHistory || [];

    // Find most recent entry matching this tab by tabId or URL
    const match = [...history].reverse().find(
      (h) => h.tabId === tab.id || h.url === currentUrl
    );

    if (match) {
      renderAnalysis(match, match.url || currentUrl);
    } else {
      showIdle();
    }
  });

  // Dashboard button
  document.getElementById('dashBtn').addEventListener('click', () => {
    chrome.tabs.create({ url: chrome.runtime.getURL('dashboard.html') });
    window.close();
  });

  // Clear history
  document.getElementById('clearBtn').addEventListener('click', () => {
    chrome.storage.local.set({ analysisHistory: [] }, () => {
      showIdle();
    });
  });
}

init();
