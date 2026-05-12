# Brain

This folder contains the main analysis logic for NoBully. It includes the API server, model inference code, model training code, text cleaning logic, and helper scripts.

## Main files

```text
brain/
├── api_server.py
├── execute.py
├── filterHTML.js
├── get_data.py
├── polish.py
├── train.py
└── models/
```

## File overview

- `api_server.py` defines the FastAPI app and the `/analyze`, `/health`, `/dashboard`, and moderator API endpoints.
- `execute.py` loads the BERT, LSTM, and polish-layer models, cleans text, splits page content into chunks, calculates toxicity, and decides whether a page should be blocked.
- `filterHTML.js` is an older or standalone browser-side page filter script that can collect page text, send it to the backend, blur detected words, and block the page.
- `get_data.py` is intended for dataset collection or import workflows.
- `polish.py` trains and loads the polish layer used to reduce false positives and adjust raw model probabilities.
- `train.py` trains the BERT classifier, LSTM classifier, and polish layer.
- `models/` stores the saved model artifacts required for inference.

## Running the API

Run from the `backend/` folder:

```bash
python -m uvicorn brain.api_server:app --reload --host 127.0.0.1 --port 8000
```

## Running training

Training requires CSV datasets in folders such as:

```text
brain/data/
brain/curated_data/
```

Then run:

```bash
python train.py
```

## Notes

This folder is the core of the backend. Avoid moving model paths or renaming files unless the corresponding paths in the Python code are also updated.
