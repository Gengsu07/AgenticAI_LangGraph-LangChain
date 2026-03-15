"""
LangServe + FastAPI Production Server for MSME VPS Deployment
=============================================================
Best practices implementation for scalable LLM applications.

Key features:
- Structured configuration via environment variables
- Health check & readiness endpoints
- CORS middleware for frontend integration
- Rate limiting to protect VPS resources
- Centralized error handling
- Multiple chain routes (translation + general assistant)
- Async-ready with proper uvicorn configuration
- Graceful shutdown support

Run with:
    Development : python serve.py
    Production  : gunicorn serve:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langserve import add_routes

# ──────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────
load_dotenv()

# --- General Settings ---
APP_TITLE = os.getenv("APP_TITLE", "LangChain LLM API Server")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "Production-grade API server using LangChain + LangServe for MSME workloads",
)

# --- Server Settings ---
HOST = os.getenv("SERVER_HOST", "0.0.0.0")  # 0.0.0.0 to accept external traffic on VPS
PORT = int(os.getenv("SERVER_PORT", "8000"))
WORKERS = int(os.getenv("SERVER_WORKERS", "2"))  # Match VPS CPU cores (2 is safe for 2-core VPS)
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# --- CORS Settings (adjust for your frontend domain) ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# --- Rate Limiting (simple in-memory, use Redis for multi-worker) ---
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))  # requests per minute

# --- LLM Settings ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "Gemma2-9b-It")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ──────────────────────────────────────────────────────────────
# 3. LOGGING SETUP
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("langserve")

# ──────────────────────────────────────────────────────────────
# 4. VALIDATE REQUIRED SETTINGS
# ──────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file or environment variables."
    )

# ──────────────────────────────────────────────────────────────
# 5. LLM MODEL INITIALIZATION
# ──────────────────────────────────────────────────────────────
model = ChatGroq(
    model=LLM_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)
logger.info("LLM model initialized: %s (temp=%.1f, max_tokens=%d)", LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS)

# ──────────────────────────────────────────────────────────────
# 6. CHAINS DEFINITION
# ──────────────────────────────────────────────────────────────
parser = StrOutputParser()

# --- Chain 1: Translation Chain (original) ---
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate the following text into {language}. "
               "Provide only the translation, no explanations."),
    ("user", "{text}"),
])
translation_chain = translation_prompt | model | parser

# --- Chain 2: General Assistant Chain ---
assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for a small-to-medium business. "
               "Provide clear, concise, and actionable answers. "
               "If the question is about business operations, give practical advice."),
    ("user", "{question}"),
])
assistant_chain = assistant_prompt | model | parser

# --- Chain 3: Summarizer Chain ---
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer. Summarize the following text into key bullet points. "
               "Keep it concise and focus on the most important information."),
    ("user", "{text}"),
])
summarizer_chain = summarizer_prompt | model | parser

logger.info("All chains initialized successfully")

# ──────────────────────────────────────────────────────────────
# 7. RATE LIMITER (Simple In-Memory)
# ──────────────────────────────────────────────────────────────
# NOTE: For multi-worker production, replace with Redis-based rate limiter
# (e.g., slowapi with Redis backend)
_request_log: dict[str, list[float]] = {}


def check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is within rate limits."""
    now = time.time()
    window = 60  # 1 minute

    if client_ip not in _request_log:
        _request_log[client_ip] = []

    # Clean old entries
    _request_log[client_ip] = [t for t in _request_log[client_ip] if now - t < window]

    if len(_request_log[client_ip]) >= RATE_LIMIT_RPM:
        return False

    _request_log[client_ip].append(now)
    return True


# ──────────────────────────────────────────────────────────────
# 8. FASTAPI APP WITH LIFESPAN (startup/shutdown)
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    logger.info("🚀 Server starting — %s v%s", APP_TITLE, APP_VERSION)
    logger.info("📍 Listening on %s:%d", HOST, PORT)
    logger.info("🔗 Docs available at http://%s:%d/docs", HOST, PORT)
    yield
    logger.info("🛑 Server shutting down gracefully...")


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
    openapi_url="/openapi.json",
)

# ──────────────────────────────────────────────────────────────
# 9. MIDDLEWARE
# ──────────────────────────────────────────────────────────────

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Logging & Rate Limiting Middleware ---
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Log requests, enforce rate limits, and track response time."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check (skip health endpoints)
    if request.url.path not in ("/health", "/ready", "/docs", "/redoc", "/openapi.json"):
        if not check_rate_limit(client_ip):
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

    response = await call_next(request)

    # Log response time
    duration = time.time() - start_time
    logger.info(
        "%s %s — %d — %.2fs — %s",
        request.method, request.url.path, response.status_code, duration, client_ip,
    )
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    return response


# ──────────────────────────────────────────────────────────────
# 10. HEALTH & READINESS ENDPOINTS
# ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check — returns 200 if server is running."""
    return {"status": "healthy", "version": APP_VERSION}


@app.get("/ready", tags=["System"])
async def readiness_check():
    """
    Readiness check — verifies the LLM connection is working.
    Useful for load balancers and monitoring on VPS.
    """
    try:
        # Quick test invoke to verify LLM connectivity
        test_result = await model.ainvoke("Say 'ok'")
        return {
            "status": "ready",
            "llm_model": LLM_MODEL,
            "llm_connected": True,
        }
    except Exception as e:
        logger.error("Readiness check failed: %s", str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "llm_model": LLM_MODEL,
                "llm_connected": False,
                "error": str(e),
            },
        )


@app.get("/", tags=["System"])
async def root():
    """API root — shows available endpoints."""
    return {
        "message": f"Welcome to {APP_TITLE}",
        "version": APP_VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ready": "/ready",
            "translation": "/translate/playground/",
            "assistant": "/assistant/playground/",
            "summarizer": "/summarize/playground/",
        },
    }


# ──────────────────────────────────────────────────────────────
# 11. LANGSERVE CHAIN ROUTES
# ──────────────────────────────────────────────────────────────
add_routes(app, translation_chain, path="/translate")
add_routes(app, assistant_chain, path="/assistant")
add_routes(app, summarizer_chain, path="/summarize")

logger.info("Chain routes registered: /translate, /assistant, /summarize")

# ──────────────────────────────────────────────────────────────
# 12. GLOBAL EXCEPTION HANDLER
# ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler to prevent exposing internals."""
    logger.error("Unhandled exception on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again later.",
            "path": request.url.path,
        },
    )


# ──────────────────────────────────────────────────────────────
# 13. ENTRYPOINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",         # String reference for auto-reload support
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        reload=True,         # Auto-reload in development (disable in production)
        workers=1,           # Use 1 worker with reload; use gunicorn for multi-worker
        timeout_keep_alive=30,  # Keep connections alive for 30s
    )
