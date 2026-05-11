// NoBully Background Service Worker
// Handles cross-tab communication and storage

chrome.runtime.onInstalled.addListener(() => {
  console.log("[NoBully] Extension installed");
  chrome.storage.sync.set({ extensionVersion: "1.0.0", extensionEnabled: true });
});

chrome.commands.onCommand.addListener((command) => {
  if (command === "toggle-extension") {
    chrome.storage.sync.get("extensionEnabled", (res) => {
      const next = !(res.extensionEnabled !== false);
      chrome.storage.sync.set({ extensionEnabled: next });
      console.log("[NoBully] Toggled via shortcut:", next ? "ON" : "OFF");
    });
  }
});

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("[NoBully] Message received:", request);

  if (request.action === "logAnalysis") {
    chrome.storage.local.get("analysisHistory", (result) => {
      const history = result.analysisHistory || [];
      history.push({
        ...request.data,
        url: sender.tab?.url || "",
        tabId: sender.tab?.id,
        timestamp: new Date().toISOString(),
      });
      // Keep only last 100 analyses
      if (history.length > 100) history.shift();
      chrome.storage.local.set({ analysisHistory: history });
    });
  }

  sendResponse({ status: "ok" });
  return true; // keep message channel open for async
});
