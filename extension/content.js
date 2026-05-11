(function () {
  if (window.__BULLY_FILTER_INSTALLED__) {
    return;
  }

  window.__BULLY_FILTER_INSTALLED__ = true;

  const DEFAULT_SETTINGS = {
    enabled: true,
    apiUrl: "http://127.0.0.1:8000/analyze",
    severityThresholdPercent: 65,
    negativeWordThreshold: 30,
  };
  const maxBatchTextLength = 2500;
  const maxTextNodesPerScan = 90;
  const viewportMargin = 350;
  const scanDebounceMs = 120;
  const mutationDebounceMs = 180;
  const maxResultAgeMs = 12000;
  const cacheLimit = 2500;
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

  let settings = { ...DEFAULT_SETTINGS };
  let scanTimer = null;
  let inFlight = false;
  let pendingScan = false;
  let knownBlurTerms = new Set();
  let sentTextKeys = new Set();
  let observedRoots = new WeakSet();
  let latestAnalysis = null;
  let cachedPolicy = null;
  let policyVersion = 0;
  let cachedPolicyVersion = -1;
  let lastApiWarningAt = 0;
  let apiUnavailableUntil = 0;
  let viewGeneration = 0;
  let currentUrl = window.location.href;

  function getStorageArea() {
    if (typeof chrome === "undefined" || !chrome.storage?.local) {
      return null;
    }

    return chrome.storage.local;
  }

  function readSettings() {
    const storage = getStorageArea();

    if (!storage) {
      return Promise.resolve(DEFAULT_SETTINGS);
    }

    return new Promise((resolve) => {
      storage.get(DEFAULT_SETTINGS, (items) => {
        resolve({ ...DEFAULT_SETTINGS, ...items });
      });
    });
  }

  function writeLastAnalysis(result) {
    const storage = getStorageArea();

    if (!storage) {
      return Promise.resolve();
    }

    return new Promise((resolve) => {
      storage.set(
        {
          bullyLastAnalysis: {
            ...result,
            url: window.location.href,
            title: document.title,
            checkedAt: new Date().toISOString(),
          },
        },
        resolve
      );
    });
  }

  function installBlurStyles(root = document) {
    const target = root === document ? document.documentElement : root;

    if (!target || target.querySelector?.("#bully-blur-styles")) {
      return;
    }

    const rootDocument = target.ownerDocument || document;
    const style = rootDocument.createElement("style");
    style.id = "bully-blur-styles";
    style.textContent = `
      .bully-blurred-word {
        filter: blur(6px);
        user-select: none;
      }
    `;
    target.appendChild(style);
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

  function buildBlurPolicy() {
    if (cachedPolicy && cachedPolicyVersion === policyVersion) {
      return cachedPolicy;
    }

    const wordKeys = new Set();
    const phrasePatterns = [];

    knownBlurTerms.forEach((term) => {
      const cleanTerm = String(term || "").trim();
      const parts = cleanTerm.match(/[A-Za-z0-9@$!|+]+/g) || [];

      if (cleanTerm.length < 2) {
        return;
      }

      if (parts.length > 1) {
        const phrasePattern = buildPhrasePattern(cleanTerm);

        if (phrasePattern) {
          phrasePatterns.push(phrasePattern);
        }

        return;
      }

      buildNegativeWordKeys(cleanTerm).forEach((key) => wordKeys.add(key));
    });

    cachedPolicy = { wordKeys, phrasePatterns };
    cachedPolicyVersion = policyVersion;
    return cachedPolicy;
  }

  function textCacheKey(text) {
    return text.toLowerCase().replace(/\s+/g, " ").trim().slice(0, 500);
  }

  function rememberTextKey(key) {
    if (!key) {
      return false;
    }

    if (sentTextKeys.has(key)) {
      return false;
    }

    if (sentTextKeys.size > cacheLimit) {
      sentTextKeys = new Set(Array.from(sentTextKeys).slice(-Math.floor(cacheLimit / 2)));
    }

    sentTextKeys.add(key);
    return true;
  }

  function resetForNavigation() {
    if (window.location.href === currentUrl) {
      return;
    }

    currentUrl = window.location.href;
    viewGeneration += 1;
    sentTextKeys.clear();
    knownBlurTerms.clear();
    policyVersion += 1;
  }

  function markViewChanged() {
    resetForNavigation();
    viewGeneration += 1;
  }

  function cleanVisibleText(text) {
    return String(text || "").replace(/\s+/g, " ").trim();
  }

  function isElementVisible(element) {
    if (!element || skippedTags.has(element.tagName)) {
      return false;
    }

    if (
      element.closest?.(
        "[data-bully-blurred='true'], [hidden], [aria-hidden='true']"
      )
    ) {
      return false;
    }

    const elementWindow = element.ownerDocument?.defaultView || window;
    const style = elementWindow.getComputedStyle(element);

    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      style.opacity === "0"
    ) {
      return false;
    }

    const rects = element.getClientRects();

    if (!rects.length) {
      return false;
    }

    const width =
      elementWindow.innerWidth || element.ownerDocument.documentElement.clientWidth;
    const height =
      elementWindow.innerHeight || element.ownerDocument.documentElement.clientHeight;

    for (const rect of rects) {
      if (
        rect.bottom >= -viewportMargin &&
        rect.right >= -viewportMargin &&
        rect.top <= height + viewportMargin &&
        rect.left <= width + viewportMargin
      ) {
        return true;
      }
    }

    return false;
  }

  function textNodeIntersectsViewport(textNode) {
    const rootDocument = textNode.ownerDocument || document;
    const nodeWindow = rootDocument.defaultView || window;
    const width = nodeWindow.innerWidth || rootDocument.documentElement.clientWidth;
    const height = nodeWindow.innerHeight || rootDocument.documentElement.clientHeight;
    const range = rootDocument.createRange();

    try {
      range.selectNodeContents(textNode);

      for (const rect of range.getClientRects()) {
        if (
          rect.width > 0 &&
          rect.height > 0 &&
          rect.bottom >= -viewportMargin &&
          rect.right >= -viewportMargin &&
          rect.top <= height + viewportMargin &&
          rect.left <= width + viewportMargin
        ) {
          return true;
        }
      }
    } finally {
      range.detach();
    }

    return false;
  }

  function canReadTextNode(textNode) {
    const parent = textNode.parentElement;
    const text = cleanVisibleText(textNode.textContent);

    if (!parent || !text || skippedTags.has(parent.tagName)) {
      return false;
    }

    return isElementVisible(parent) && textNodeIntersectsViewport(textNode);
  }

  function addElementChain(element, containers) {
    let currentElement = element;
    let depth = 0;
    const viewportArea =
      (window.innerWidth || document.documentElement.clientWidth) *
      (window.innerHeight || document.documentElement.clientHeight);

    while (
      currentElement &&
      currentElement.nodeType === Node.ELEMENT_NODE &&
      currentElement !== document.body &&
      currentElement !== document.documentElement &&
      depth < 4
    ) {
      if (!skippedTags.has(currentElement.tagName)) {
        const rect = currentElement.getBoundingClientRect();
        const area = Math.max(1, rect.width * rect.height);

        if (depth === 0 || area <= viewportArea * 2.5) {
          containers.add(currentElement);
        }
      }

      if (currentElement.shadowRoot) {
        installBlurStyles(currentElement.shadowRoot);
        observeRoot(currentElement.shadowRoot);
        const shadowElement = currentElement.shadowRoot.elementFromPoint?.(
          window.innerWidth / 2,
          window.innerHeight / 2
        );

        if (shadowElement && shadowElement !== currentElement) {
          containers.add(shadowElement);
        }
      }

      currentElement = currentElement.parentElement;
      depth += 1;
    }
  }

  function collectViewportContainers() {
    const containers = new Set();
    const width = window.innerWidth || document.documentElement.clientWidth;
    const height = window.innerHeight || document.documentElement.clientHeight;
    const xFractions = [0.12, 0.28, 0.44, 0.6, 0.76, 0.92];
    const yFractions = [0.08, 0.2, 0.34, 0.48, 0.62, 0.76, 0.9];

    observeRoot(document.body || document.documentElement);
    installBlurStyles();

    for (const xFraction of xFractions) {
      for (const yFraction of yFractions) {
        const x = Math.max(0, Math.min(width - 1, Math.round(width * xFraction)));
        const y = Math.max(0, Math.min(height - 1, Math.round(height * yFraction)));
        const element = document.elementFromPoint(x, y);

        if (element) {
          addElementChain(element, containers);
        }
      }
    }

    if (!containers.size && document.body) {
      containers.add(document.body);
    }

    return containers;
  }

  function collectVisibleTextNodes(limit = maxTextNodesPerScan) {
    const containers = collectViewportContainers();
    const textNodes = [];
    const seenTextNodes = new WeakSet();

    for (const container of containers) {
      const containerDocument = container.ownerDocument || document;
      const walker = containerDocument.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
        acceptNode(textNode) {
          if (seenTextNodes.has(textNode)) {
            return NodeFilter.FILTER_REJECT;
          }

          return canReadTextNode(textNode)
            ? NodeFilter.FILTER_ACCEPT
            : NodeFilter.FILTER_REJECT;
        },
      });

      while (walker.nextNode()) {
        seenTextNodes.add(walker.currentNode);
        textNodes.push(walker.currentNode);

        if (textNodes.length >= limit) {
          return textNodes;
        }
      }
    }

    return textNodes;
  }

  function buildNewVisibleTextBatch(textNodes) {
    const parts = [];
    let length = 0;

    for (const textNode of textNodes) {
      const text = cleanVisibleText(textNode.textContent);

      if (text.length < 2) {
        continue;
      }

      const key = textCacheKey(text);

      if (!rememberTextKey(key)) {
        continue;
      }

      parts.push(text);
      length += text.length + 1;

      if (length >= maxBatchTextLength) {
        break;
      }
    }

    return parts.join("\n");
  }

  function sendRuntimeMessage(message) {
    return new Promise((resolve, reject) => {
      if (typeof chrome === "undefined" || !chrome.runtime?.sendMessage) {
        reject(new Error("Chrome runtime messaging is not available."));
        return;
      }

      chrome.runtime.sendMessage(message, (response) => {
        const runtimeError = chrome.runtime.lastError;

        if (runtimeError) {
          reject(new Error(runtimeError.message));
          return;
        }

        resolve(response);
      });
    });
  }

  async function sendTextToExecute(text) {
    const response = await sendRuntimeMessage({
      type: "BULLY_ANALYZE_TEXT",
      apiUrl: settings.apiUrl,
      payload: {
        url: window.location.href,
        title: document.title,
        text,
        fast_mode: true,
        severity_threshold_percent: Number(settings.severityThresholdPercent),
        negative_word_threshold: Number(settings.negativeWordThreshold),
      },
    });

    if (response?.inactive) {
      return { skipped: true, reason: "inactive" };
    }

    if (response?.busy) {
      return { skipped: true, reason: "busy" };
    }

    if (!response?.ok) {
      throw new Error(response?.error || "The local analysis API did not respond.");
    }

    return response.result;
  }

  function publishAnalysisResult(result) {
    latestAnalysis = result;
    window.__BULLY_LAST_ANALYSIS__ = result;
    window.dispatchEvent(
      new CustomEvent("bully-analysis-complete", { detail: result })
    );
  }

  function addKnownBlurTerms(terms) {
    let changed = false;

    (terms || []).forEach((term) => {
      const cleanTerm = String(term || "").trim();

      if (cleanTerm.length < 2 || knownBlurTerms.has(cleanTerm)) {
        return;
      }

      knownBlurTerms.add(cleanTerm);
      changed = true;
    });

    if (changed) {
      policyVersion += 1;
    }
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
    const rootDocument = textNode.ownerDocument || document;
    const fragment = rootDocument.createDocumentFragment();
    let lastIndex = 0;

    if (!ranges.length || !textNode.parentNode) {
      return;
    }

    ranges.forEach(([wordStart, wordEnd]) => {
      fragment.appendChild(rootDocument.createTextNode(text.slice(lastIndex, wordStart)));

      const span = rootDocument.createElement("span");
      span.className = "bully-blurred-word";
      span.dataset.bullyBlurred = "true";
      span.textContent = text.slice(wordStart, wordEnd);
      fragment.appendChild(span);
      lastIndex = wordEnd;
    });

    fragment.appendChild(rootDocument.createTextNode(text.slice(lastIndex)));
    textNode.parentNode.replaceChild(fragment, textNode);
  }

  function blurVisibleKnownTerms() {
    blurKnownTermsInTextNodes(collectVisibleTextNodes());
  }

  function blurKnownTermsInTextNodes(textNodes) {
    const policy = buildBlurPolicy();

    if (!policy.wordKeys.size && !policy.phrasePatterns.length) {
      return;
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
    if (window.top !== window) {
      return;
    }

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
    `;

    heading.textContent = "Page blocked";
    message.textContent =
      result.message ||
      "This page was blocked because it may contain harmful content.";
    stats.className = "stats";

    addStat(stats, "Severity", `${result.severity_percent || 0}%`);
    addStat(
      stats,
      "Negative words",
      `${result.negative_word_count || 0}/${result.negative_word_threshold || 30}`
    );

    button.textContent = "Go back";
    button.addEventListener("click", () => window.history.back());

    head.append(title, style);
    panel.append(heading, message, stats, button);
    main.appendChild(panel);
    body.appendChild(main);
    document.documentElement.replaceChildren(head, body);
  }

  async function processVisibleBatch() {
    resetForNavigation();

    if (!settings.enabled || document.hidden || Date.now() < apiUnavailableUntil) {
      return;
    }

    if (inFlight) {
      pendingScan = true;
      return;
    }

    const textNodes = collectVisibleTextNodes();
    blurKnownTermsInTextNodes(textNodes);
    const text = buildNewVisibleTextBatch(textNodes);

    if (!text.trim()) {
      return;
    }

    inFlight = true;
    const requestGeneration = viewGeneration;
    const requestUrl = window.location.href;
    const requestStartedAt = Date.now();

    try {
      const result = await sendTextToExecute(text);

      if (result?.skipped) {
        if (result.reason === "busy") {
          pendingScan = true;
        }

        return;
      }

      if (
        document.hidden ||
        requestUrl !== window.location.href ||
        requestGeneration !== viewGeneration ||
        Date.now() - requestStartedAt > maxResultAgeMs
      ) {
        pendingScan = true;
        return;
      }

      publishAnalysisResult(result);
      addKnownBlurTerms(result.blur_words || result.negative_word_matches || []);
      blurKnownTermsInTextNodes(collectVisibleTextNodes());
      await writeLastAnalysis(result);

      if (result.blocked) {
        blockPage(result);
      }
    } catch (error) {
      apiUnavailableUntil = Date.now() + 5000;

      if (Date.now() - lastApiWarningAt > 15000) {
        lastApiWarningAt = Date.now();
        console.warn(
          "Could not analyze visible text with the safety filter. Make sure the local API is running at the configured URL.",
          error
        );
      }
    } finally {
      inFlight = false;

      if (pendingScan) {
        pendingScan = false;
        scheduleScan(scanDebounceMs);
      }
    }
  }

  function scheduleScan(delay = scanDebounceMs) {
    if (!settings.enabled) {
      return;
    }

    window.clearTimeout(scanTimer);
    scanTimer = window.setTimeout(processVisibleBatch, delay);
  }

  function observeRoot(root) {
    if (!root || observedRoots.has(root)) {
      return;
    }

    observedRoots.add(root);

    const observer = new MutationObserver(() => {
      scheduleScan(mutationDebounceMs);
    });
    observer.observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  function installEventListeners() {
    window.addEventListener("scroll", () => {
      markViewChanged();
      scheduleScan(scanDebounceMs);
    }, {
      passive: true,
      capture: true,
    });
    window.addEventListener("resize", () => {
      markViewChanged();
      scheduleScan(scanDebounceMs);
    }, {
      passive: true,
    });
    window.addEventListener("focus", () => {
      markViewChanged();
      scheduleScan(0);
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        markViewChanged();
        scheduleScan(0);
      }
    });

    if (typeof chrome !== "undefined" && chrome.storage?.onChanged) {
      chrome.storage.onChanged.addListener((changes, areaName) => {
        if (areaName !== "local") {
          return;
        }

        if (
          changes.enabled ||
          changes.apiUrl ||
          changes.severityThresholdPercent ||
          changes.negativeWordThreshold
        ) {
          readSettings().then((updatedSettings) => {
            settings = updatedSettings;
            sentTextKeys.clear();
            markViewChanged();
            scheduleScan(0);
          });
        }
      });
    }

    if (typeof chrome !== "undefined" && chrome.runtime?.onMessage) {
      chrome.runtime.onMessage.addListener((message) => {
        if (message?.type !== "BULLY_SCAN_VISIBLE_NOW") {
          return false;
        }

        apiUnavailableUntil = 0;
        markViewChanged();
        blurVisibleKnownTerms();
        scheduleScan(0);
        return false;
      });
    }
  }

  async function run() {
    settings = await readSettings();

    if (!settings.enabled) {
      return;
    }

    installBlurStyles();
    observeRoot(document.body || document.documentElement);
    installEventListeners();
    scheduleScan(0);
    window.setInterval(() => scheduleScan(scanDebounceMs), 2500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run, { once: true });
  } else {
    run();
  }
})();
