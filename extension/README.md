# Browser extension

This folder contains the Chromium browser extension for NoBully.

## Contents

```text
extension/
├── background.js
├── content.js
├── manifest.json
├── popup.css
├── popup.html
└── popup.js
```

## Purpose

The extension scans visible text on web pages, sends that text to the local NoBully backend, receives an analysis result, and then blurs harmful words or blocks the page when needed.

## File overview

- `manifest.json` defines the extension as a Manifest V3 Chromium extension.
- `background.js` runs as the service worker. It sends API requests to the backend and handles scan messages.
- `content.js` runs inside web pages. It collects visible text, observes page changes, blurs harmful words, and blocks pages when necessary.
- `popup.html` defines the extension popup interface.
- `popup.css` styles the popup interface.
- `popup.js` saves settings, tests the backend health endpoint, and displays the latest analysis result.

## Installation

1. Open a Chromium-based browser.
2. Go to `chrome://extensions`.
3. Enable developer mode.
4. Click `Load unpacked`.
5. Select this `extension/` folder.

## Default backend URL

```text
http://127.0.0.1:8000/analyze
```

The backend must be running before the extension can analyze pages.

## Settings

The popup lets you configure:

- enabled or disabled state
- API URL
- severity threshold percentage
- negative-word threshold

## Notes

The extension uses local browser storage for settings and the latest scan result. It is designed to work with the local FastAPI backend in `backend/`.
