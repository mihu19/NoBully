// NoBully Content Filter - Content Script
// This runs on every webpage and analyzes content

const API_URL = "http://127.0.0.1:5000/analyze"; // Change this to cloud URL when deployed
const TOXICITY_THRESHOLD = 65;
const BAD_WORD_THRESHOLD = 30;

console.log("[NoBully] Content script loaded");

// Extract visible text from page
const skippedTags = new Set([
  "SCRIPT", "STYLE", "TEXTAREA", "INPUT", "SELECT", "OPTION",
  "CODE", "PRE", "NOSCRIPT", "SVG", "CANVAS",
]);

function canReadTextNode(textNode) {
  const parent = textNode.parentElement;
  const text = textNode.textContent || "";
  if (!parent || skippedTags.has(parent.tagName)) return false;
  if (parent.closest("[hidden], [aria-hidden='true']")) return false;
  return Boolean(text.trim());
}

function parseTextFromHtml(root = document.body || document.documentElement) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(textNode) {
      return canReadTextNode(textNode) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const parts = [];
  while (walker.nextNode()) {
    parts.push(walker.currentNode.textContent.trim());
  }
  return parts.join("\n").slice(0, 5000);
}

// Show blocking overlay
function showBlockOverlay(analysis) {
  const overlay = document.createElement("div");
  overlay.id = "nobully-block-overlay";
  overlay.innerHTML = `
    <div style="
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.93);
      display: flex; align-items: center; justify-content: center;
      z-index: 999999;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
      <div style="
        background: #0d0f1c;
        border: 1px solid #ff3355;
        padding: 44px 40px;
        border-radius: 14px;
        max-width: 480px;
        width: 90%;
        text-align: center;
        box-shadow: 0 0 60px rgba(255,51,85,0.2), 0 20px 60px rgba(0,0,0,0.6);
      ">
        <div style="font-size: 42px; margin-bottom: 16px;">🛡️</div>
        <h1 style="color: #ff3355; margin: 0 0 12px; font-size: 24px; font-weight: 800; letter-spacing: -0.5px;">
          Page Blocked
        </h1>
        <p style="color: #8892aa; font-size: 14px; margin: 0 0 24px; line-height: 1.6;">
          NoBully detected toxic or harmful content and blocked this page for your safety.
        </p>
        <div style="
          background: #131626; border: 1px solid #1e2438;
          padding: 16px 20px; border-radius: 10px; margin: 0 0 24px;
          text-align: left; font-family: 'IBM Plex Mono', monospace; font-size: 12px;
        ">
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2438;margin-bottom:6px;">
            <span style="color:#4a5270">Toxicity Score</span>
            <span style="color:#ff3355;font-weight:600">${analysis.toxicity_percent}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2438;margin-bottom:6px;">
            <span style="color:#4a5270">Severity Score</span>
            <span style="color:#ffb020;font-weight:600">${analysis.severity_percent}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;">
            <span style="color:#4a5270">Flagged Words</span>
            <span style="color:#c8cfe8;font-weight:600">${analysis.bad_word_count}</span>
          </div>
        </div>
        <p style="color: #4a5270; font-size: 11px; margin: 0 0 20px;">
          A report has been sent to your moderation dashboard.
        </p>
        <button onclick="window.history.back()" style="
          background: #ff3355; color: white; border: none;
          padding: 12px 32px; border-radius: 8px; font-size: 14px;
          font-weight: 600; cursor: pointer; letter-spacing: 0.02em;
        ">
          ← Go Back
        </button>
      </div>
    </div>
  `;

  const existing = document.getElementById("nobully-block-overlay");
  if (existing) existing.remove();

  document.body.appendChild(overlay);
  document.body.style.overflow = "hidden";
}

// Send alert to moderator
async function sendAlertToModerator(analysis) {
  try {
    const moderatorUrl = "http://127.0.0.1:3000/api/alert";
    const alert = {
      url: window.location.href,
      timestamp: new Date().toISOString(),
      toxicity_percent: analysis.toxicity_percent,
      severity_percent: analysis.severity_percent,
      bad_word_count: analysis.bad_word_count,
      flagged_words: analysis.flagged_words,
      page_title: document.title,
    };
    console.log("[NoBully] Sending alert to moderator:", alert);
    await fetch(moderatorUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(alert),
    }).catch(() => {
      console.log("[NoBully] Moderator server not available");
    });
  } catch (error) {
    console.error("[NoBully] Error sending alert:", error);
  }
}

// Log analysis to background (for popup + dashboard)
function logAnalysis(analysis) {
  chrome.runtime.sendMessage({
    action: "logAnalysis",
    data: {
      ...analysis,
      url: window.location.href,
      page_title: document.title,
    },
  }).catch(() => {}); // ignore if background not ready
}

// Check if extension is enabled
async function isEnabled() {
  return new Promise((resolve) => {
    chrome.storage.sync.get("extensionEnabled", (res) => {
      resolve(res.extensionEnabled !== false);
    });
  });
}

// Analyze page
async function analyzePage() {
  try {
    if (!(await isEnabled())) {
      console.log("[NoBully] Extension disabled — skipping analysis");
      return;
    }

    const pageText = parseTextFromHtml();
    if (!pageText.trim()) {
      console.log("[NoBully] No text found on page");
      return;
    }

    console.log("[NoBully] Analyzing page text…");

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: pageText }),
    });

    if (!response.ok) {
      console.error("[NoBully] API error:", response.status);
      return;
    }

    const analysis = await response.json();
    console.log("[NoBully] Analysis result:", analysis);

    // Always log to storage (for popup + dashboard)
    logAnalysis(analysis);

    if (analysis.should_block) {
      console.log("[NoBully] Page BLOCKED");
      showBlockOverlay(analysis);
      await sendAlertToModerator(analysis);
    } else {
      console.log(
        "[NoBully] Page allowed — Toxicity: " + analysis.toxicity_percent +
        "%, Bad words: " + analysis.bad_word_count
      );
    }
  } catch (error) {
    console.error("[NoBully] Error analyzing page:", error);
  }
}

// Run analysis when page loads
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", analyzePage);
} else {
  analyzePage();
}
