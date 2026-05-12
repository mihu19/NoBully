# Bully Page Safety Filter Extension

## Start the local API

From the project root:

```powershell
pip install -r requirements.txt
python -m uvicorn brain.api_server:app --reload --host 127.0.0.1 --port 8000
```

## Load in Chrome or Edge

1. Open `chrome://extensions` or `edge://extensions`.
2. Turn on Developer mode.
3. Choose Load unpacked.
4. Select this folder: `extension`.
5. Keep the Python API running while browsing.

## Use the popup

Open the extension popup to enable or disable the filter, test the API connection,
and change the severity or model-detected negative-word thresholds.

The extension scans visible text as you scroll, watches newly added page content,
and runs in frames where Chrome or Edge allows content-script injection.
