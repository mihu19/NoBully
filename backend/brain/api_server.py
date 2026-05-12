import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if __package__:
    from .execute import (
        DEFAULT_INFERENCE_CONFIG,
        DEFAULT_MODEL_PATHS,
        DEFAULT_PAGE_SAFETY_CONFIG,
        LoadedModels,
        PageSafetyAnalysis,
        PageSafetyConfig,
        analyze_page_snapshot,
        load_models,
        predict_base_toxicity_probability,
    )
else:
    from execute import (
        DEFAULT_INFERENCE_CONFIG,
        DEFAULT_MODEL_PATHS,
        DEFAULT_PAGE_SAFETY_CONFIG,
        LoadedModels,
        PageSafetyAnalysis,
        PageSafetyConfig,
        analyze_page_snapshot,
        load_models,
        predict_base_toxicity_probability,
    )


MODERATOR_WARNING_URL_ENV = "MODERATOR_WARNING_URL"

app = FastAPI(
    title="Bully Page Safety API",
    version="1.0.0",
    description="Runs the local toxicity model against browser page snapshots.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_local_extension_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


class AnalyzeRequest(BaseModel):
    html: str | None = None
    text: str | None = None
    url: str | None = None
    title: str | None = None
    fast_mode: bool = False
    severity_threshold_percent: int | None = Field(default=None, ge=0, le=100)
    negative_word_threshold: int | None = Field(default=None, ge=0)


class AnalyzeResponse(BaseModel):
    blocked: bool
    block_reasons: list[str]
    severity_percent: int
    toxicity_percent: int
    flagged_words: list[str]
    negative_word_count: int
    negative_word_matches: list[str]
    blur_words: list[str]
    threshold_percent: int
    negative_word_threshold: int
    analyzed_character_count: int
    analyzed_word_count: int
    chunks_analyzed: int
    message: str
    moderator_warning_sent: bool
    moderator_warning_error: str | None = None


@lru_cache(maxsize=1)
def get_models() -> LoadedModels:
    return load_models(DEFAULT_MODEL_PATHS, DEFAULT_INFERENCE_CONFIG)


def warm_model() -> None:
    models = get_models()
    predict_base_toxicity_probability(
        "warmup page text",
        models,
        DEFAULT_INFERENCE_CONFIG,
    )


@app.on_event("startup")
async def preload_models_on_startup() -> None:
    warm_model()


def build_safety_config(request: AnalyzeRequest) -> PageSafetyConfig:
    overrides = {}

    if request.severity_threshold_percent is not None:
        overrides["severity_threshold_percent"] = request.severity_threshold_percent

    if request.negative_word_threshold is not None:
        overrides["negative_word_threshold"] = request.negative_word_threshold

    if request.fast_mode:
        overrides.update(
            {
                "max_text_characters": 8000,
                "page_chunk_word_count": 90,
                "max_page_chunks": 4,
                "max_returned_negative_words": 40,
                "model_negative_word_min_probability": 0.60,
                "max_negative_word_saliency_chunks": 1,
            }
        )

    if not overrides:
        return DEFAULT_PAGE_SAFETY_CONFIG

    return replace(DEFAULT_PAGE_SAFETY_CONFIG, **overrides)


@lru_cache(maxsize=512)
def analyze_text_snapshot_cached(
    text: str,
    severity_threshold_percent: int,
    negative_word_threshold: int,
    fast_mode: bool,
) -> PageSafetyAnalysis:
    request = AnalyzeRequest(
        text=text,
        severity_threshold_percent=severity_threshold_percent,
        negative_word_threshold=negative_word_threshold,
        fast_mode=fast_mode,
    )
    safety_config = build_safety_config(request)
    return analyze_page_snapshot(
        html_snapshot=None,
        text_snapshot=text,
        models=get_models(),
        inference_config=DEFAULT_INFERENCE_CONFIG,
        safety_config=safety_config,
    )


async def send_moderator_warning(
    request: AnalyzeRequest,
    analysis: PageSafetyAnalysis,
) -> tuple[bool, str | None]:
    moderator_url = os.getenv(MODERATOR_WARNING_URL_ENV)

    if not moderator_url:
        return False, None

    payload = {
        "url": request.url,
        "title": request.title,
        "blocked": analysis.blocked,
        "block_reasons": analysis.block_reasons,
        "severity_percent": analysis.severity_percent,
        "toxicity_percent": analysis.toxicity_percent,
        "negative_word_count": analysis.negative_word_count,
        "negative_word_matches": analysis.negative_word_matches,
        "blur_words": analysis.blur_words,
        "flagged_words": analysis.flagged_words,
        "message": analysis.message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(moderator_url, json=payload)
            response.raise_for_status()
    except Exception as error:
        return False, str(error)

    return True, None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if not (request.html and request.html.strip()) and not (
        request.text and request.text.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Send either 'html' or 'text' in the JSON body.",
        )

    try:
        safety_config = build_safety_config(request)

        if request.text and request.text.strip() and not (
            request.html and request.html.strip()
        ):
            analysis = analyze_text_snapshot_cached(
                request.text,
                safety_config.severity_threshold_percent,
                safety_config.negative_word_threshold,
                request.fast_mode,
            )
        else:
            analysis = analyze_page_snapshot(
                html_snapshot=request.html,
                text_snapshot=request.text,
                models=get_models(),
                inference_config=DEFAULT_INFERENCE_CONFIG,
                safety_config=safety_config,
            )
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    warning_sent = False
    warning_error = None

    if analysis.blocked:
        warning_sent, warning_error = await send_moderator_warning(request, analysis)

    response_data = asdict(analysis)
    response_data["moderator_warning_sent"] = warning_sent
    response_data["moderator_warning_error"] = warning_error
    return AnalyzeResponse(**response_data)
