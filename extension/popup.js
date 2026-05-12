const DEFAULT_SETTINGS = {
  enabled: true,
  apiUrl: "http://127.0.0.1:8000/analyze",
  severityThresholdPercent: 65,
  negativeWordThreshold: 30,
};

const fields = {
  enabled: document.getElementById("enabled"),
  apiUrl: document.getElementById("apiUrl"),
  severityThresholdPercent: document.getElementById(
    "severityThresholdPercent"
  ),
  negativeWordThreshold: document.getElementById("negativeWordThreshold"),
};
const form = document.getElementById("settingsForm");
const apiStatus = document.getElementById("apiStatus");
const testButton = document.getElementById("testButton");
const lastResult = document.getElementById("lastResult");
const lastDecision = document.getElementById("lastDecision");
const lastSeverity = document.getElementById("lastSeverity");
const lastToxicity = document.getElementById("lastToxicity");
const lastNegativeWords = document.getElementById("lastNegativeWords");
const lastUrl = document.getElementById("lastUrl");

function storageGet(defaults) {
  return new Promise((resolve) => {
    chrome.storage.local.get(defaults, resolve);
  });
}

function storageSet(values) {
  return new Promise((resolve) => {
    chrome.storage.local.set(values, resolve);
  });
}

function toHealthUrl(apiUrl) {
  const url = new URL(apiUrl);
  url.pathname = "/health";
  url.search = "";
  url.hash = "";
  return url.toString();
}

function setApiStatus(label, className) {
  apiStatus.textContent = label;
  apiStatus.className = `status ${className}`;
}

function readFormSettings() {
  return {
    enabled: fields.enabled.checked,
    apiUrl: fields.apiUrl.value.trim() || DEFAULT_SETTINGS.apiUrl,
    severityThresholdPercent: Number(fields.severityThresholdPercent.value),
    negativeWordThreshold: Number(fields.negativeWordThreshold.value),
  };
}

function writeFormSettings(settings) {
  fields.enabled.checked = Boolean(settings.enabled);
  fields.apiUrl.value = settings.apiUrl;
  fields.severityThresholdPercent.value = settings.severityThresholdPercent;
  fields.negativeWordThreshold.value = settings.negativeWordThreshold;
}

async function loadSettings() {
  const settings = await storageGet(DEFAULT_SETTINGS);
  writeFormSettings({ ...DEFAULT_SETTINGS, ...settings });
}

async function saveSettings() {
  const settings = readFormSettings();
  await storageSet(settings);
  setApiStatus("Saved", "status-ok");
}

async function testApi() {
  const settings = readFormSettings();

  try {
    setApiStatus("Checking", "status-idle");
    const response = await fetch(toHealthUrl(settings.apiUrl));

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    setApiStatus("Online", "status-ok");
  } catch (error) {
    setApiStatus("Offline", "status-error");
  }
}

function renderLastAnalysis(analysis) {
  if (!analysis) {
    lastResult.hidden = true;
    return;
  }

  lastResult.hidden = false;
  lastDecision.textContent = analysis.blocked ? "Blocked" : "Allowed";
  lastSeverity.textContent = `${analysis.severity_percent || 0}%`;
  lastToxicity.textContent = `${analysis.toxicity_percent || 0}%`;
  lastNegativeWords.textContent = `${analysis.negative_word_count || 0}/${
    analysis.negative_word_threshold || DEFAULT_SETTINGS.negativeWordThreshold
  }`;
  lastUrl.textContent = analysis.url || "";
  lastUrl.title = analysis.url || "";
}

async function loadLastAnalysis() {
  const { bullyLastAnalysis } = await storageGet({ bullyLastAnalysis: null });
  renderLastAnalysis(bullyLastAnalysis);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await saveSettings();
});

testButton.addEventListener("click", testApi);

document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  await loadLastAnalysis();
  await testApi();
});
