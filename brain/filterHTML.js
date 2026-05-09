(function () {
  const apiUrl =
    window.BULLY_EXECUTE_API_URL || "http://127.0.0.1:8000/analyze";
  const maxTextLength = 5000;
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

    while (walker.nextNode()) {
      parts.push(walker.currentNode.textContent.trim());
    }

    return parts.join("\n").slice(0, maxTextLength);
  }

  async function sendTextToExecute(text) {
    if (!text.trim()) {
      return null;
    }

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`execute.py request failed with ${response.status}`);
    }

    return response.json();
  }

  async function run() {
    try {
      await sendTextToExecute(parseTextFromHtml());
    } catch (error) {
      console.warn("Could not send page text to execute.py.", error);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run, { once: true });
  } else {
    run();
  }
})();
