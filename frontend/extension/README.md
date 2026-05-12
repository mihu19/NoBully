# 🚀 NoBully Browser Extension Setup Guide

## What You Have Now

✅ **Flask API** (`new_api.py`) - Running on http://127.0.0.1:5000
- ✓ Loads your trained AI model
- ✓ Analyzes text for toxicity and bad words
- ✓ Returns: toxicity %, severity %, flagged words, and block decision

✅ **Browser Extension Files** (Ready to load in Chrome)
- `manifest.json` - Extension configuration
- `content.js` - Analyzes every webpage
- `background.js` - Handles cross-tab communication

## Installing the Extension in Chrome

### Step 1: Open Extensions Page
1. Open Chrome
2. Type in address bar: `chrome://extensions/`
3. Toggle **"Developer mode"** (top right corner)

### Step 2: Load Extension
1. Click **"Load unpacked"** button
2. Navigate to: `e:\FAF\SDA\NoBully.worktrees\copilot-worktree-2026-05-11T09-24-45\extension`
3. Click **"Select Folder"**

### Step 3: Verify Installation
- You should see "NoBully - Content Safety Filter" in your extensions list
- The extension is now active on all websites

## How It Works

```
User visits website
         ↓
Extension loads page content
         ↓
Extract visible text (skips scripts, styles, etc.)
         ↓
Send to API: POST /analyze with text
         ↓
Python AI model analyzes
         ↓
Returns: {
  toxicity_percent: 75,
  severity_percent: 70,
  bad_word_count: 5,
  flagged_words: ["word1", "word2"],
  should_block: false/true
}
         ↓
Check: if toxicity > 65% AND bad_words >= 30
         ↓
If YES → Block page + Show warning + Send alert
If NO → Allow page normally
```

## Testing the Extension

### Test Case 1: Normal Content (Should Allow)
1. Visit a normal website (like example.com)
2. Open Chrome DevTools (F12 → Console)
3. You should see: `[NoBully] Page allowed - Toxicity: XX%`

### Test Case 2: Toxic Content (Should Block)
Create a test HTML file with toxic content:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>This is a very toxic page</h1>
    <p>You are stupid and dumb and worthless you should die</p>
    <p>This contains many bad words and offensive language</p>
</body>
</html>
```

Save as `test.html`, then:
1. Open the file in Chrome (drag & drop into browser)
2. If toxicity > 65% AND bad words >= 30, you'll see the block overlay
3. Check Console (F12) for analysis logs

## What's Next

### Phase 1: Test Locally (You are here ✓)
- [x] Flask API running
- [x] Extension installed in Chrome
- [x] Test on sample pages

### Phase 2: Create Moderator Dashboard
- [ ] Create Node.js server (`moderator/moderator.js`)
- [ ] Create web UI to see blocked pages
- [ ] Setup SQLite database for alerts
- [ ] Test end-to-end

### Phase 3: Deploy
- [ ] Deploy Flask API to cloud (Heroku/Railway)
- [ ] Update API_URL in extension to cloud endpoint
- [ ] Deploy moderator dashboard
- [ ] Create Chrome Web Store listing

## Troubleshooting

### Extension not working?
1. Check Console (F12) for errors
2. Verify API URL is correct (http://127.0.0.1:5000)
3. Reload extension: chrome://extensions → click reload
4. Check if Flask API is running

### Page analysis slow?
- First analysis loads the model (slow)
- Subsequent requests are fast
- This is normal on first request

### API connection refused?
1. Make sure Flask server is running:
   ```bash
   cd brain
   python new_api.py
   ```
2. Verify port 5000 is not blocked by firewall
3. Try health check: Open http://127.0.0.1:5000/ in browser

## File Structure

```
NoBully/
├── brain/
│   ├── execute.py (Your AI model)
│   ├── new_api.py (Flask API - NEW)
│   ├── models/ (Trained model files)
│   └── ...
└── extension/ (NEW)
    ├── manifest.json (Extension config)
    ├── content.js (Page analyzer)
    ├── background.js (Service worker)
    └── README.md (This file)
```

## API Endpoint Reference

### POST /analyze
**Send:** `{ "text": "page text here" }`

**Receive:**
```json
{
  "toxicity_percent": 75,
  "severity_percent": 70,
  "bad_word_count": 5,
  "flagged_words": ["word1", "word2"],
  "should_block": false
}
```

### GET /health
Quick health check to verify API is running

### GET /
Shows API documentation

## Configuration

### Change API URL
If you deploy Flask API to cloud, update `content.js`:
```javascript
const API_URL = "https://your-api.com/analyze"; // Change this line
```

### Change Thresholds
In `content.js`:
```javascript
const TOXICITY_THRESHOLD = 65;    // Block if > 65%
const BAD_WORD_THRESHOLD = 30;    // Block if >= 30 bad words
```

## Security Notes

⚠️ **Current Setup (Local Development Only)**
- Extension sends page text to local API (on your computer)
- Data does NOT leave your machine
- Perfect for testing and development

🔒 **For Production**
- Use HTTPS for API (not HTTP)
- Add authentication tokens
- Use browser storage for sensitive data
- Add content security policies

## Questions?

Check the console logs (F12) for detailed information about what's happening.
Each log message starts with `[NoBully]` so you can search for them.

---
**Next Step**: Create the Moderator Dashboard (Phase 2)
