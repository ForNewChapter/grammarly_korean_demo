# 서버 시작점
# 웹 요청을 받아서 교정 파이프라인에 넘기고, 결과를 돌려주는 역할만 한다.

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model_loader import ModelLoader, IS_CLOUD_RUN
from correction_pipeline import CorrectionPipeline

LOGGER = logging.getLogger("grammarly_korean")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_BACKEND_NAME = os.getenv("BACKEND_NAME") or socket.gethostname()
DEFAULT_BACKEND_BRANCH = os.getenv("BACKEND_BRANCH")
DEFAULT_BACKEND_VERSION = os.getenv("BACKEND_VERSION")


class CorrectRequest(BaseModel):
    text: str = Field(min_length=0, max_length=5000)
    cursor: Optional[int] = None
    mode: str = "realtime"


# ── FastAPI 앱 ─────────────────────────────────────────────

app = FastAPI(title="Korean On-Device Grammarly Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 엔진 싱글턴 ───────────────────────────────────────────

PIPELINE: Optional[CorrectionPipeline] = None
ENGINE_INIT_LOCK = threading.Lock()
ENGINE_WARMUP_THREAD: Optional[threading.Thread] = None
ENGINE_INITIALIZING = False
ENGINE_INIT_ERROR: Optional[str] = None


def get_pipeline() -> CorrectionPipeline:
    global PIPELINE, ENGINE_INITIALIZING, ENGINE_INIT_ERROR
    if PIPELINE is not None:
        return PIPELINE
    with ENGINE_INIT_LOCK:
        if PIPELINE is not None:
            return PIPELINE
        ENGINE_INITIALIZING = True
        ENGINE_INIT_ERROR = None
        try:
            LOGGER.info("Initializing correction engine")
            models = ModelLoader()
            PIPELINE = CorrectionPipeline(models)
            LOGGER.info("Correction engine initialized")
            return PIPELINE
        except Exception as exc:
            PIPELINE = None
            ENGINE_INIT_ERROR = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Correction engine initialization failed")
            raise
        finally:
            ENGINE_INITIALIZING = False


def _warm_engine_background() -> None:
    try:
        get_pipeline()
    except Exception:
        pass


def schedule_engine_warmup() -> None:
    global ENGINE_WARMUP_THREAD
    if PIPELINE is not None:
        return
    with ENGINE_INIT_LOCK:
        if PIPELINE is not None:
            return
        if ENGINE_WARMUP_THREAD is not None and ENGINE_WARMUP_THREAD.is_alive():
            return
        ENGINE_WARMUP_THREAD = threading.Thread(
            target=_warm_engine_background,
            name="engine-warmup",
            daemon=True,
        )
        ENGINE_WARMUP_THREAD.start()


# ── 라우트 ─────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    if IS_CLOUD_RUN:
        LOGGER.info("Cloud Run detected; deferring heavy engine startup")
        schedule_engine_warmup()
        return
    get_pipeline()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(APP_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(APP_DIR / "styles.css", media_type="text/css")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "backendName": DEFAULT_BACKEND_NAME,
        "backendBranch": DEFAULT_BACKEND_BRANCH,
        "backendVersion": DEFAULT_BACKEND_VERSION,
        "engineReady": PIPELINE is not None,
        "engineInitializing": ENGINE_INITIALIZING,
        "engineError": ENGINE_INIT_ERROR,
        "cloudRun": IS_CLOUD_RUN,
    }


@app.post("/api/correct")
def correct(req: CorrectRequest) -> Dict[str, Any]:
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model engine is not ready: {exc}") from exc
    text = req.text or ""
    cursor = len(text) if req.cursor is None else req.cursor
    return pipeline.run(full_text=text, cursor=cursor, mode=req.mode)
