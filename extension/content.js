// NoBully Content Filter - Content Script
// This runs on every webpage and analyzes content

const API_URL = "http://127.0.0.1:5000/analyze"; // Change this to cloud URL when deployed
const TOXICITY_THRESHOLD = 65;
const BAD_WORD_THRESHOLD = 30;

console.log("[NoBully] Content script loaded");

// Extract visible text from page (same logic as filterHTML.js)
const skippedTags = new Set([
  "SCRIPT",
  "STYLE",
  "TEXTAREA",
  "INPUT",
  "SELECT",
  "OPTION",
  "CODE",
  "PRE",
  "NOSCRIPT",
  "SVG",
  "CANVAS",
]);

function canReadTextNode(textNode) {
  const parent = textNode.parentElement;
  const text = textNode.textContent || "";

  if (!parent || skippedTags.has(parent.tagName)) {
    return false;
  }

  if (parent.closest("[hidden], [aria-hidden='true']")) {
    return false;
  }

  return Boolean(text.trim());
}

function parseTextFromHtml(root = document.body || document.documentElement) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(textNode) {
      return canReadTextNode(textNode)
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT;
    },
  });
  const parts = [];
  const maxTextLength = 5000;

  while (walker.nextNode()) {
    parts.push(walker.currentNode.textContent.trim());
  }

  return parts.join("\n").slice(0, maxTextLength);
}

// Show blocking overlay
function showBlockOverlay(analysis) {
  // Create overlay
  const overlay = document.createElement("div");
  overlay.id = "nobully-block-overlay";
  overlay.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.9);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 999999;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
      <div style="
        background: white;
        padding: 40px;
        border-radius: 12px;
        max-width: 500px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      ">
        <h1 style="color: #d32f2f; margin: 0 0 20px 0; font-size: 28px;">
          🚫 Page Blocked
        </h1>
        <p style="color: #333; font-size: 16px; margin: 20px 0;">
          This page contains toxic or unsafe content and has been blocked for your safety.
        </p>
        <div style="
          background: #f5f5f5;
          padding: 20px;
          border-radius: 8px;
          margin: 20px 0;
          text-align: left;
        ">
          <p style="margin: 5px 0; font-size: 14px;">
            <strong>Toxicity Score:</strong> ${analysis.toxicity_percent}%
          </p>
          <p style="margin: 5px 0; font-size: 14px;">
            <strong>Severity Score:</strong> ${analysis.severity_percent}%
          </p>
          <p style="margin: 5px 0; font-size: 14px;">
            <strong>Flagged Words:</strong> ${analysis.bad_word_count}
          </p>
        </div>
        <p style="color: #666; font-size: 13px; margin: 20px 0 0 0;">
          A report has been sent to our moderation team.
        </p>
        <button onclick="window.history.back()" style="
          background: #1976d2;
          color: white;
          border: none;
          padding: 12px 30px;
          border-radius: 6px;
          font-size: 16px;
          cursor: pointer;
          margin-top: 20px;
        ">
          Go Back
        </button>
      </div>
    </div>
  `;

  // Remove existing overlay if any
  const existing = document.getElementById("nobully-block-overlay");
  if (existing) {
    existing.remove();
  }

  document.body.appendChild(overlay);
  document.body.style.overflow = "hidden"; // Prevent scrolling
}

// Send alert to moderator
async function sendAlertToModerator(analysis) {
  try {
    const moderatorUrl = "http://127.0.0.1:3000/api/alert"; // Will be set up later
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

    // Note: This will fail until moderator server is running
    await fetch(moderatorUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(alert),
    }).catch(() => {
      console.log("[NoBully] Moderator server not available (will try later)");
    });
  } catch (error) {
    console.error("[NoBully] Error sending alert:", error);
  }
}

// Analyze page
async function analyzePage() {
  try {
    const pageText = parseTextFromHtml();

    if (!pageText.trim()) {
      console.log("[NoBully] No text found on page");
      return;
    }

    console.log("[NoBully] Analyzing page text...");

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: pageText }),
      timeout: 30000, // 30 second timeout
    });

    if (!response.ok) {
      console.error("[NoBully] API error:", response.status);
      return;
    }

    const analysis = await response.json();
    console.log("[NoBully] Analysis result:", analysis);

    // Decision logic: BLOCK if toxicity > 65% AND bad_words >= 30
    if (analysis.should_block) {
      console.log("[NoBully] Page BLOCKED");
      showBlockOverlay(analysis);
      await sendAlertToModerator(analysis);
    } else {
      console.log("[NoBully] Page allowed - Toxicity: " + analysis.toxicity_percent + "%, Bad words: " + analysis.bad_word_count);
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
