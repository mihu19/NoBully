# NoBully

**NoBully** is an AI-powered cyberbullying detection tool. It analyzes text on web pages in real time using custom trained BERT and LSTM models and flags harmful content right in your browser.

The project is split into two components:
- **`brain/`** — A FastAPI backend that runs inference using custom trained BERT and LSTM models
- **`extension/`** — A browser extension that intercepts page content and calls the backend's `/analyze` endpoint

---

## Architecture

```
Browser Extension  ──►  FastAPI Backend (/analyze)  ──►  Custom-Trained Models
   (extension/)                (brain/)                        (PyTorch)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- A Chromium-based browser (Chrome, Edge, Brave, etc.)
- *(Optional)* [ngrok](https://ngrok.com/) for exposing the backend over the internet

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
python -m uvicorn brain.api_server:app --reload --host 127.0.0.1 --port 8000
```

Wait until you see:

```
INFO:     Application startup complete.
```

The API will be available at `http://127.0.0.1:8000`.

### 3. (Optional) Expose via ngrok

If you want the extension to reach the backend from any machine or network:

```bash
ngrok http 8000
```

Copy the generated public URL (e.g. `https://xxxx.ngrok.io`) — you'll need it in step 5.

### 4. Load the Browser Extension

1. Open your browser and navigate to the extensions page (e.g. `chrome://extensions`)
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `/extension` folder from this repository

### 5. Configure the Extension

1. Click the NoBully extension icon in your browser toolbar
2. In the extension UI, paste your API URL into the input field:
   - Local: `http://127.0.0.1:8000/analyze`
   - ngrok: `https://xxxx.ngrok.io/analyze`
3. Save


## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| NLP models | Custom-trained BERT & LSTM (trained on HuggingFace datasets) |
| Deep learning | PyTorch |
| Data processing | pandas, datasets |
| HTTP client | httpx |
| Extension | Vanilla JS, HTML, CSS |

---

## Project Structure


---

## Moderator Page

>  *Coming soon.*

---

