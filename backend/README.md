# Backend

This folder contains the Python side of NoBully. It runs the FastAPI server, loads the machine learning models, analyzes text from the browser extension, and serves the moderation dashboard.

## Contents

```text
backend/
├── requirements.txt
├── brain/
└── modpage/
```

## Main responsibilities

- Start the local API server.
- Receive page text through `/analyze`.
- Load and run the saved models from `brain/models/`.
- Return toxicity, severity, flagged words, and block decisions.
- Store recent moderation events in memory.
- Serve the dashboard from `modpage/`.

## Setup

Run these commands from this folder:

```bash
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

Start the server:

```bash
python -m uvicorn brain.api_server:app --reload --host 127.0.0.1 --port 8000
```

## Important URLs

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/analyze
http://127.0.0.1:8000/dashboard
```

## Notes

The backend must be running before the browser extension can analyze pages. The moderation history is stored only in memory, so it is cleared when the backend restarts.
