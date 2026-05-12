(function () {
  const apiUrl =
    window.BULLY_EXECUTE_API_URL || "http://127.0.0.1:8000/analyze";
  const maxTextLength = 20000;
  const maxHtmlLength = 100000;
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
  const leetCharacters = {
    0: "o",
    1: "i",
    "!": "i",
    "|": "i",
    3: "e",
    4: "a",
    "@": "a",
    5: "s",
    $: "s",
    7: "t",
    "+": "t",
    8: "b",
  };
  const negativeWordScanPattern = /[A-Za-z0-9@$!|+]+(?:[._*'-]+[A-Za-z0-9@$!|+]+)*/g;

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

  function parseHtmlSnapshot() {
    const root = document.documentElement;

    if (!root) {
      return "";
    }

    return root.outerHTML.slice(0, maxHtmlLength);
  }

  function createPageSnapshot() {
    return {
      url: window.location.href,
      title: document.title,
      html: parseHtmlSnapshot(),
      text: parseTextFromHtml(),
    };
  }

  async function sendSnapshotToExecute(snapshot) {
    if (!snapshot.html.trim() && !snapshot.text.trim()) {
      return null;
    }

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(snapshot),
    });

    if (!response.ok) {
      throw new Error(`execute.py request failed with ${response.status}`);
    }

    return response.json();
  }

  function publishAnalysisResult(result) {
    window.__BULLY_LAST_ANALYSIS__ = result;
    window.dispatchEvent(
      new CustomEvent("bully-analysis-complete", { detail: result })
    );
  }

  function installBlurStyles() {
    if (document.getElementById("bully-blur-styles")) {
      return;
    }

    const style = document.createElement("style");
    style.id = "bully-blur-styles";
    style.textContent = `
      .bully-blurred-word {
        filter: blur(5px);
        user-select: none;
      }
    `;
    document.documentElement.appendChild(style);
  }

  function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function normalizeNegativeWordText(value) {
    return String(value || "")
      .toLowerCase()
      .replace(/[0183457!|@$+]/g, (character) => leetCharacters[character] || character)
      .replace(/[^a-z0-9]+/g, "");
  }

  function squashRepeatedCharacters(value) {
    return value.replace(/(.)\1{2,}/g, "$1$1");
  }

  function buildNegativeWordKeys(value) {
    const normalized = normalizeNegativeWordText(value);
    const keys = new Set();

    if (!normalized) {
      return keys;
    }

    keys.add(normalized);
    keys.add(squashRepeatedCharacters(normalized));

    if (normalized.length >= 4) {
      keys.add(normalized.replace(/(.)\1+/g, "$1"));
    }

    return keys;
  }

  function buildPhrasePattern(phrase) {
    const parts = String(phrase || "").toLowerCase().match(/[a-z0-9@$!|+]+/g) || [];

    if (parts.length < 2) {
      return null;
    }

    const pattern = parts.map(escapeRegex).join("[\\s._*'-]+");
    return new RegExp(`\\b${pattern}\\b`, "gi");
  }

  function buildBlurPolicy(terms) {
    const wordKeys = new Set();
    const phrasePatterns = [];

    Array.from(new Set(terms.map((term) => String(term || "").trim())))
      .filter((term) => term.length > 1)
      .forEach((term) => {
        const parts = term.match(/[A-Za-z0-9@$!|+]+/g) || [];

        if (parts.length > 1) {
          const phrasePattern = buildPhrasePattern(term);

          if (phrasePattern) {
            phrasePatterns.push(phrasePattern);
          }

          return;
        }

        buildNegativeWordKeys(term).forEach((key) => wordKeys.add(key));
      });

    return { wordKeys, phrasePatterns };
  }

  function canBlurTextNode(textNode) {
    const parent = textNode.parentElement;

    if (!parent || skippedTags.has(parent.tagName)) {
      return false;
    }

    if (parent.closest("[data-bully-blurred='true'], [hidden], [aria-hidden='true']")) {
      return false;
    }

    return Boolean(textNode.textContent && textNode.textContent.trim());
  }

  function hasNegativeWordKey(candidate, wordKeys) {
    for (const key of buildNegativeWordKeys(candidate)) {
      if (wordKeys.has(key)) {
        return true;
      }
    }

    return false;
  }

  function mergeRanges(ranges) {
    if (!ranges.length) {
      return [];
    }

    const sortedRanges = ranges.sort((first, second) => first[0] - second[0]);
    const mergedRanges = [sortedRanges[0]];

    for (const range of sortedRanges.slice(1)) {
      const previousRange = mergedRanges[mergedRanges.length - 1];

      if (range[0] <= previousRange[1]) {
        previousRange[1] = Math.max(previousRange[1], range[1]);
      } else {
        mergedRanges.push(range);
      }
    }

    return mergedRanges;
  }

  function findBlurRanges(text, policy) {
    const ranges = [];
    let match;

    negativeWordScanPattern.lastIndex = 0;

    while ((match = negativeWordScanPattern.exec(text)) !== null) {
      if (hasNegativeWordKey(match[0], policy.wordKeys)) {
        ranges.push([match.index, match.index + match[0].length]);
      }
    }

    policy.phrasePatterns.forEach((pattern) => {
      pattern.lastIndex = 0;

      while ((match = pattern.exec(text)) !== null) {
        ranges.push([match.index, match.index + match[0].length]);
      }
    });

    return mergeRanges(ranges);
  }

  function blurTextNode(textNode, policy) {
    const text = textNode.textContent || "";
    const ranges = findBlurRanges(text, policy);
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;

    if (!ranges.length) {
      return;
    }

    ranges.forEach(([wordStart, wordEnd]) => {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex, wordStart)));
      const span = document.createElement("span");
      span.className = "bully-blurred-word";
      span.dataset.bullyBlurred = "true";
      span.textContent = text.slice(wordStart, wordEnd);
      fragment.appendChild(span);
      lastIndex = wordEnd;
    });

    fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    textNode.parentNode.replaceChild(fragment, textNode);
  }

  function blurNegativeWords(terms) {
    const policy = buildBlurPolicy(terms);

    if (
      (!policy.wordKeys.size && !policy.phrasePatterns.length) ||
      !document.body
    ) {
      return;
    }

    installBlurStyles();

    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(textNode) {
        return canBlurTextNode(textNode)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    });
    const textNodes = [];

    while (walker.nextNode()) {
      textNodes.push(walker.currentNode);
    }

    textNodes.forEach((textNode) => blurTextNode(textNode, policy));
  }

  function addStat(parent, label, value) {
    const item = document.createElement("div");
    const labelElement = document.createElement("span");
    const valueElement = document.createElement("strong");

    labelElement.textContent = label;
    valueElement.textContent = value;
    item.append(labelElement, valueElement);
    parent.appendChild(item);
  }

  function blockPage(result) {
    const head = document.createElement("head");
    const title = document.createElement("title");
    const style = document.createElement("style");
    const body = document.createElement("body");
    const main = document.createElement("main");
    const panel = document.createElement("section");
    const heading = document.createElement("h1");
    const message = document.createElement("p");
    const stats = document.createElement("div");
    const button = document.createElement("button");

    title.textContent = "Page blocked";
    style.textContent = `
      :root {
        color-scheme: light;
        font-family: Arial, Helvetica, sans-serif;
        background: #f8fafc;
        color: #172033;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        background: #f8fafc;
      }

      main {
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
      }

      section {
        width: min(560px, 100%);
        border: 1px solid #d8dee8;
        border-radius: 8px;
        background: #ffffff;
        padding: 28px;
        box-shadow: 0 18px 45px rgba(23, 32, 51, 0.12);
      }

      h1 {
        margin: 0 0 12px;
        font-size: 28px;
        line-height: 1.2;
      }

      p {
        margin: 0;
        color: #4c5870;
        font-size: 16px;
        line-height: 1.5;
      }

      .stats {
        display: grid;
        gap: 10px;
        margin: 22px 0;
      }

      .stats div {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        border-bottom: 1px solid #edf0f5;
        padding-bottom: 10px;
      }

      .stats div:last-child {
        border-bottom: 0;
        padding-bottom: 0;
      }

      .stats span {
        color: #5d687f;
      }

      .stats strong {
        color: #172033;
      }

      button {
        border: 0;
        border-radius: 6px;
        background: #172033;
        color: #ffffff;
        cursor: pointer;
        font: inherit;
        min-height: 44px;
        padding: 0 18px;
      }

      button:hover {
        background: #29344b;
      }
    `;

    heading.textContent = "Page blocked";
    message.textContent =
      result.message ||
      "This page was blocked because it may contain harmful content.";
    stats.className = "stats";

    addStat(stats, "Severity", `${result.severity_percent || 0}%`);
    addStat(stats, "Toxicity", `${result.toxicity_percent || 0}%`);
    addStat(
      stats,
      "Negative words",
      `${result.negative_word_count || 0}/${result.negative_word_threshold || 30}`
    );

    button.textContent = "Go back";
    button.addEventListener("click", () => {
      window.history.back();
    });

    head.append(title, style);
    panel.append(heading, message, stats, button);
    main.appendChild(panel);
    body.appendChild(main);
    document.documentElement.replaceChildren(head, body);
  }

  async function run() {
    try {
      const result = await sendSnapshotToExecute(createPageSnapshot());

      if (!result) {
        return;
      }

      publishAnalysisResult(result);
      blurNegativeWords([
        ...(result.blur_words || result.negative_word_matches || []),
      ]);

      if (result.blocked) {
        blockPage(result);
      }
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
