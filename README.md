# NoBully

NoBully is a browser-based cyberbullying and harmful-language detection project. It scans text from web pages, sends the page text to a local FastAPI backend, runs the text through trained machine learning models, and then either allows the page, blurs detected harmful words, or blocks the page when the configured safety thresholds are exceeded.

The project has three main parts:

- `backend/` contains the Python API, model inference code, training utilities, moderation dashboard, and saved model files.
- `extension/` contains the Chromium browser extension that scans pages and communicates with the backend.
- `backend/modpage/` contains a local dashboard for reviewing recent moderation events.

## How it works

1. The browser extension runs on visited pages.
2. The content script collects visible text from the page.
3. The background service worker sends the text to the backend `/analyze` endpoint.
4. The backend loads the trained BERT, LSTM, and polish-layer models.
5. The backend returns a safety result with toxicity, severity, flagged words, and a block decision.
6. The extension blurs matched harmful words or replaces the page with a block screen.
7. The backend stores recent moderation events in memory for the dashboard.

## Architecture

```text
browser page
   |
   v
extension/content.js
   |
   v
extension/background.js
   |
   v
backend/brain/api_server.py
   |
   v
backend/brain/execute.py
   |
   v
backend/brain/models/
```

## Main features

- Real-time visible-page scanning through a Chromium extension.
- Local FastAPI backend for text analysis.
- Custom trained BERT model for toxicity classification.
- LSTM classifier used together with the BERT model.
- Polish layer that helps reduce false positives in harmless contexts.
- Configurable severity and negative-word thresholds.
- Word blurring for detected harmful terms.
- Full-page blocking when content passes the configured danger threshold.
- Popup UI for enabling/disabling the filter, changing the API URL, testing the API, and seeing the last result.
- Local moderation dashboard for scan history and basic statistics.

## Repository structure

```text
NoBully/
├── README.md
├── backend/
│   ├── README.md
│   ├── requirements.txt
│   ├── brain/
│   │   ├── README.md
│   │   ├── api_server.py
│   │   ├── execute.py
│   │   ├── filterHTML.js
│   │   ├── get_data.py
│   │   ├── polish.py
│   │   ├── train.py
│   │   └── models/
│   │       ├── README.md
│   │       ├── lstm_classifier.pt
│   │       ├── polish_layer.pt
│   │       └── bred_bert/
│   │           ├── README.md
│   │           ├── config.json
│   │           ├── model.safetensors
│   │           ├── tokenizer.json
│   │           └── tokenizer_config.json
│   └── modpage/
│       ├── README.md
│       ├── dashboard.html
│       ├── dashboard.js
│       └── icons/
│           ├── README.md
│           ├── icon16.png
│           ├── icon32.png
│           ├── icon48.png
│           └── icon128.png
└── extension/
    ├── README.md
    ├── background.js
    ├── content.js
    ├── manifest.json
    ├── popup.css
    ├── popup.html
    └── popup.js
```

## Requirements

- Python 3.11 or newer
- pip
- A Chromium-based browser such as Chrome, Edge, Brave, or Chromium
- Enough disk space and memory to load the included PyTorch models

## Backend setup

From the repository root:

```bash
cd backend
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python -m uvicorn brain.api_server:app --reload --host 127.0.0.1 --port 8000
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

Check that it is running:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok"}
```

## Extension setup

1. Open your Chromium browser.
2. Go to `chrome://extensions` or the equivalent extensions page.
3. Enable developer mode.
4. Click `Load unpacked`.
5. Select the `extension/` folder.
6. Click the NoBully extension icon.
7. Make sure the API URL is set to:

```text
http://127.0.0.1:8000/analyze
```

8. Click the test button in the popup to confirm that the backend is online.

## Using the project

Start the backend first, then load or enable the extension. After that, open any website. The extension will scan visible text, send it to the backend, and apply the result.

When harmful content is detected, NoBully can:

- blur matched words on the page
- store the result as the latest analysis
- block the page if the result exceeds the threshold
- add the scan result to the local moderation history

## Backend API

### GET `/health`

Checks whether the backend is online.

### POST `/analyze`

Analyzes a page snapshot or plain text.

Example request:

```json
{
  "text": "text to analyze",
  "url": "https://example.com",
  "title": "Example page",
  "fast_mode": true,
  "severity_threshold_percent": 65,
  "negative_word_threshold": 30
}
```

Example response fields:

```json
{
  "blocked": false,
  "block_reasons": [],
  "severity_percent": 0,
  "toxicity_percent": 0,
  "flagged_words": [],
  "negative_word_count": 0,
  "negative_word_matches": [],
  "blur_words": [],
  "threshold_percent": 65,
  "negative_word_threshold": 30,
  "analyzed_character_count": 0,
  "analyzed_word_count": 0,
  "chunks_analyzed": 0,
  "message": "Page allowed",
  "moderator_warning_sent": false,
  "moderator_warning_error": null
}
```

### GET `/dashboard`

Opens the local moderation dashboard.

```text
http://127.0.0.1:8000/dashboard
```

### GET `/moderator/api/history`

Returns recent moderation events stored in memory.

### GET `/moderator/api/stats`

Returns basic dashboard statistics.

### DELETE `/moderator/api/history`

Clears the in-memory moderation history.

## Model files

The backend uses saved models from `backend/brain/models/`.

- `bred_bert/` stores the transformer model and tokenizer files.
- `lstm_classifier.pt` stores the LSTM classifier checkpoint.
- `polish_layer.pt` stores the additional correction layer used to reduce false positives.

Do not delete these files unless you plan to retrain or replace the models.

## Training

The training code is in `backend/brain/train.py`. It trains the BERT model, trains the LSTM model, and then trains the polish layer.

The training script expects dataset folders such as:

```text
backend/brain/data/
backend/brain/curated_data/
```

These folders may not be included in the repository. Add the required CSV files before running training.

Run training from the `backend/brain` folder:

```bash
cd backend/brain
python train.py
```

The polish layer can also be trained separately:

```bash
python polish.py
```

## Moderator dashboard

The moderation dashboard is served by the backend at:

```text
http://127.0.0.1:8000/dashboard
```

It shows recent page scans, block/safe/caution status, toxicity values, flagged word counts, and summary statistics. The history is stored in memory, so it resets when the backend process restarts.

## Configuration

The extension popup lets the user configure:

- whether the filter is enabled
- the backend API URL
- the severity threshold percentage
- the negative-word threshold

Default API URL:

```text
http://127.0.0.1:8000/analyze
```

Default severity threshold:

```text
65
```

Default negative-word threshold:

```text
30
```

## Troubleshooting

### The popup says the API is offline

Make sure the backend is running:

```bash
cd backend
python -m uvicorn brain.api_server:app --reload --host 127.0.0.1 --port 8000
```

Then check:

```text
http://127.0.0.1:8000/health
```

### The extension does not scan pages

Reload the extension from the browser extensions page, refresh the website, and make sure the filter is enabled in the popup.

### The backend cannot find model files

Make sure these paths exist:

```text
backend/brain/models/bred_bert/
backend/brain/models/lstm_classifier.pt
backend/brain/models/polish_layer.pt
```

### The dashboard is empty

Open some pages while the extension and backend are running. The dashboard only shows events from the current backend session.

