// NoBully Dashboard Script

let allHistory = [];
let currentFilter = 'all';
let lastSeenHistoryKey = '';
const HISTORY_API = '/moderator/api/history?limit=500';
const CLEAR_API = '/moderator/api/history';

function getFlagCount(entry) {
  return entry.negative_word_count ?? entry.bad_word_count ?? 0;
}

function isBlocked(entry) {
  return Boolean(entry.blocked ?? entry.should_block ?? ((entry.toxicity_percent || 0) >= 65));
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function getStatus(entry) {
  if (isBlocked(entry)) {
    return { label: 'BLOCKED', color: '#ff3355', filter: 'blocked' };
  }
  if ((entry.toxicity_percent || 0) >= 35) {
    return { label: 'CAUTION', color: '#ffb020', filter: 'caution' };
  }
  return { label: 'SAFE', color: '#00e87a', filter: 'safe' };
}

function formatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  return d.toLocaleDateString();
}

function formatUrl(url) {
  if (!url) return { host: 'Unknown', path: '' };
  try {
    const u = new URL(url);
    return { host: u.hostname, path: u.pathname.slice(0, 40) || '/' };
  } catch {
    return { host: url.slice(0, 40), path: '' };
  }
}

function scoreBarColor(pct) {
  if (pct >= 65) return '#ff3355';
  if (pct >= 35) return '#ffb020';
  return '#00e87a';
}

function showToast(msg, duration = 2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), duration);
}

function historyKey(history) {
  return history
    .slice(-10)
    .map(entry => `${entry.url || ''}|${entry.timestamp || ''}|${getFlagCount(entry)}|${entry.toxicity_percent || 0}`)
    .join('~');
}

function detectNewBlockedEntry(previousHistory, nextHistory) {
  const previousKeys = new Set(previousHistory.map(entry => `${entry.url || ''}|${entry.timestamp || ''}|${getFlagCount(entry)}|${entry.toxicity_percent || 0}`));
const newEntries = nextHistory.filter(entry => !previousKeys.has(`${entry.url || ''}|${entry.timestamp || ''}|${getFlagCount(entry)}|${entry.toxicity_percent || 0}`));
  return [...newEntries].reverse().find(entry => getStatus(entry).filter === 'blocked') || null;
}

// ── Stats ─────────────────────────────────────────────────────────────────────

function renderStats(history) {
  const total = history.length;
  const blocked = history.filter(h => getStatus(h).filter === 'blocked').length;
  const safe = history.filter(h => getStatus(h).filter === 'safe').length;
  const flaggedWords = history.reduce((sum, h) => sum + getFlagCount(h), 0);
  const avgTox = total
    ? Math.round(history.reduce((sum, h) => sum + (h.toxicity_percent || 0), 0) / total)
    : 0;

  document.getElementById('statTotal').textContent = total;
  document.getElementById('statTotalSub').textContent = total === 1 ? '1 page' : `${total} pages`;
  document.getElementById('statBlocked').textContent = blocked;
  document.getElementById('statBlockedPct').textContent =
    total ? `${Math.round((blocked / total) * 100)}% block rate` : '0% block rate';
  document.getElementById('statAvgTox').textContent = avgTox + '%';
  document.getElementById('statFlaggedWords').textContent = flaggedWords;
  document.getElementById('statFlaggedWordsSub').textContent =
    total ? `${Math.round(flaggedWords / total)} avg per scan` : '0 total matches';
  document.getElementById('statSafe').textContent = safe;
  document.getElementById('statSafePct').textContent =
    total ? `${Math.round((safe / total) * 100)}% safe rate` : '0% safe rate';

  document.getElementById('footerTimestamp').textContent =
    `Last updated: ${new Date().toLocaleTimeString()}`;
}

// ── Table ─────────────────────────────────────────────────────────────────────

function buildTable(history) {
  const filtered = currentFilter === 'all'
    ? history
    : history.filter(h => getStatus(h).filter === currentFilter);

  const count = filtered.length;
  document.getElementById('historyCount').textContent =
    count === 0 ? '' : `${count} result${count !== 1 ? 's' : ''}`;

  const container = document.getElementById('tableInner');

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">${
          currentFilter === 'all' ? 'No scans yet' : `No ${currentFilter} pages`
        }</div>
        <div class="empty-sub">${
          currentFilter === 'all'
            ? 'Browse some pages and results will appear here.'
            : 'Try a different filter.'
        }</div>
      </div>`;
    return;
  }

  const rows = [...filtered].reverse().map(entry => {
    const st = getStatus(entry);
    const tox = entry.toxicity_percent || 0;
    const sev = entry.severity_percent || 0;
    const { host, path } = formatUrl(entry.url);
    const words = (entry.flagged_words || []).slice(0, 5);
    const extraWords = (entry.flagged_words || []).length - 5;

    return `
      <tr class="row">
        <td>
          <div class="row-status">
            <div class="row-dot" style="background:${st.color}; box-shadow: 0 0 5px ${st.color}"></div>
            <span class="row-status-label" style="color:${st.color}">${st.label}</span>
          </div>
        </td>
        <td>
          <div class="row-url">
            ${escHtml(host)}
            <small>${escHtml(path)}</small>
          </div>
        </td>
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar-label" style="color:${scoreBarColor(tox)}">${tox}%</div>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width:${tox}%; background:${scoreBarColor(tox)}"></div>
            </div>
          </div>
        </td>
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar-label" style="color:${scoreBarColor(sev)}">${sev}%</div>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width:${sev}%; background:${scoreBarColor(sev)}"></div>
            </div>
          </div>
        </td>
        <td>
          <div class="row-words">
            ${words.map(w => `<span class="word-tag">${escHtml(w)}</span>`).join('')}
            ${extraWords > 0 ? `<span class="word-tag" style="opacity:0.5">+${extraWords}</span>` : ''}
            ${words.length === 0 ? '<span style="color:var(--muted);font-size:10px;">none</span>' : ''}
          </div>
        </td>
        <td class="row-time">${formatTime(entry.timestamp)}</td>
      </tr>`;
  }).join('');

  container.innerHTML = `
    <table class="history-table">
      <thead>
        <tr>
          <th>Status</th>
          <th>Page</th>
          <th>Toxicity</th>
          <th>Severity</th>
          <th>Flagged Words</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Filters ───────────────────────────────────────────────────────────────────

document.querySelectorAll('.filter-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-chip').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    buildTable(allHistory);
  });
});

// ── Actions ───────────────────────────────────────────────────────────────────

document.getElementById('clearAllBtn').addEventListener('click', async () => {
  if (!confirm('Clear all analysis history? This cannot be undone.')) return;
  await fetch(CLEAR_API, { method: 'DELETE' });
  allHistory = [];
  renderStats([]);
  renderAlerts([]);
  buildTable([]);
  showToast('History cleared.');
});

document.getElementById('exportBtn').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(allHistory, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nobully-history-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported ' + allHistory.length + ' records.');
});

// ── Alerts ───────────────────────────────────────────────────────────────────

function renderAlerts(history) {
  const blocked = history.filter(h => getStatus(h).filter === 'blocked');
  const container = document.getElementById('alertList');
  if (blocked.length === 0) {
    container.innerHTML = '<div class="no-alerts">No blocked pages yet.</div>';
    return;
  }
  container.innerHTML = [...blocked].reverse().slice(0, 5).map(entry => {
    const { host } = formatUrl(entry.url);
    return `
      <div class="alert-item">
        <div class="alert-left">
          <div class="alert-dot"></div>
          <div>
            <div class="alert-url">${escHtml(host)}</div>
            <div class="alert-meta">${formatTime(entry.timestamp)} &mdash; ${getFlagCount(entry)} flagged words</div>
          </div>
        </div>
        <span class="alert-badge">Toxicity ${entry.toxicity_percent || 0}%</span>
      </div>`;
  }).join('');
}

// ── Load ──────────────────────────────────────────────────────────────────────

async function loadData() {
  try {
    const previousHistory = allHistory;
    const res = await fetch(HISTORY_API, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allHistory = await res.json();

    renderStats(allHistory);
    renderAlerts(allHistory);
    buildTable(allHistory);

    const key = historyKey(allHistory);
    if (key !== lastSeenHistoryKey) {
      const newBlocked = detectNewBlockedEntry(previousHistory, allHistory);
      if (newBlocked) {
        const { host } = formatUrl(newBlocked.url);
        showToast(`Blocked page detected: ${host}`);
      }
      lastSeenHistoryKey = key;
    }
  } catch {
    showToast('Dashboard cannot reach backend API.');
  }
}

loadData();

// Live refresh every 5 seconds in case content.js sends new data
setInterval(loadData, 5000);