# Moderation dashboard

This folder contains the local moderation dashboard served by the backend.

## Contents

```text
modpage/
├── dashboard.html
├── dashboard.js
└── icons/
```

## Purpose

The dashboard shows recent analysis events from the backend. It can display total scans, blocked pages, safe pages, average toxicity, flagged word counts, and recent history entries.

## How to open it

Start the backend, then open:

```text
http://127.0.0.1:8000/dashboard
```

## Data source

The dashboard reads data from the backend moderator API endpoints:

```text
/moderator/api/history
/moderator/api/stats
```

## Notes

The history is stored in backend memory. It resets when the FastAPI server restarts.
