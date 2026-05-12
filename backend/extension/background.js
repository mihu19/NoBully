const DEFAULT_TIMEOUT_MS = 30000;
const ACTIVE_SCAN_DEBOUNCE_MS = 120;
const ANALYSIS_CACHE_TTL_MS = 30000;
const ANALYSIS_CACHE_LIMIT = 300;

let activeScanTimer = null;
let analysisCache = new Map();
let analysisInFlight = false;

async function postAnalyzeRequest(apiUrl, payload) {
  const cacheKey = JSON.stringify([
    apiUrl,
    payload.text,
    payload.severity_threshold_percent,
    payload.negative_word_threshold,
    Boolean(payload.fast_mode),
  ]);
  const cached = analysisCache.get(cacheKey);

  if (cached && Date.now() - cached.createdAt < ANALYSIS_CACHE_TTL_MS) {
    return cached.result;
  }

  const timeoutMs = payload.fast_mode ? 12000 : DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    analysisInFlight = true;
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`API request failed with ${response.status}: ${body}`);
    }

    const result = await response.json();

    if (analysisCache.size > ANALYSIS_CACHE_LIMIT) {
      analysisCache = new Map(
        Array.from(analysisCache.entries()).slice(-Math.floor(ANALYSIS_CACHE_LIMIT / 2))
      );
    }

    analysisCache.set(cacheKey, {
      createdAt: Date.now(),
      result,
    });
    return result;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The local analysis API timed out.");
    }

    throw error;
  } finally {
    analysisInFlight = false;
    clearTimeout(timeoutId);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "BULLY_ANALYZE_TEXT") {
    return false;
  }

  if (sender.tab && sender.tab.active === false) {
    sendResponse({ ok: false, inactive: true });
    return false;
  }

  if (message.payload?.fast_mode && analysisInFlight) {
    sendResponse({ ok: false, busy: true });
    return false;
  }

  postAnalyzeRequest(message.apiUrl, message.payload)
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));

  return true;
});

function sendScanMessage(tabId) {
  if (!tabId) {
    return;
  }

  chrome.tabs.sendMessage(
    tabId,
    { type: "BULLY_SCAN_VISIBLE_NOW" },
    () => {
      // Some tabs do not allow content scripts; ignore those quietly.
      void chrome.runtime.lastError;
    }
  );
}

function scanActiveTabSoon(delay = ACTIVE_SCAN_DEBOUNCE_MS) {
  clearTimeout(activeScanTimer);
  activeScanTimer = setTimeout(() => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const activeTab = tabs && tabs[0];
      sendScanMessage(activeTab?.id);
    });
  }, delay);
}

chrome.tabs.onActivated.addListener(({ tabId }) => {
  sendScanMessage(tabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === "complete") {
    sendScanMessage(tabId);
  }
});

chrome.windows.onFocusChanged.addListener((windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) {
    return;
  }

  scanActiveTabSoon(0);
});

chrome.runtime.onStartup.addListener(() => scanActiveTabSoon(0));
chrome.runtime.onInstalled.addListener(() => scanActiveTabSoon(0));
