import os
import time
import psutil
import sqlite3
import threading
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus Metrics
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry

# Создаем изолированный реестр метрик во избежание конфликтов при --reload
metrics_registry = CollectorRegistry()

# SQLAlchemy
from sqlalchemy import create_engine, Column, String, Integer, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime

# MLflow
import mlflow

# LangChain & LangGraph (Conceptual architecture - imports are simulated / included gracefully)
# In production, langgraph acts as the stateful router/graph that orchestrates the MCP tools, context injection, and LLM execution.

app = FastAPI(title="vLLM Inference Platform", description="FastAPI platform for HF models with LangChain/LangGraph, MCP, and advanced monitoring.")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# DATABASE SETUP
# ----------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./inference_history.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    model_name = Column(String)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    tokens_generated = Column(Integer, default=0)
    generation_time_ms = Column(Float, default=0.0)

try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------------------------------
# PROMETHEUS METRICS DEFINITIONS
# ----------------------------------------------------
REQUEST_COUNT = Counter("vllm_request_count", "Total number of inference requests", ["model", "status"], registry=metrics_registry)
GENERATION_SPEED = Histogram("vllm_generation_speed_tokens_per_sec", "Speed of model generation in tokens per second", ["model"], registry=metrics_registry)
LATENCY = Histogram("vllm_inference_latency_seconds", "Inference latency in seconds", ["model"], registry=metrics_registry)
CPU_USAGE = Gauge("system_cpu_usage_percent", "System CPU usage percent", registry=metrics_registry)
RAM_USAGE = Gauge("system_ram_usage_bytes", "System RAM usage in bytes", registry=metrics_registry)
VRAM_USAGE = Gauge("system_vram_usage_bytes", "System GPU VRAM usage in bytes", registry=metrics_registry)
ACTIVE_SESSIONS = Gauge("vllm_active_sessions", "Number of active chat sessions", registry=metrics_registry)

# Background thread to update system metrics periodically
def monitor_system_resources():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent(interval=1))
            RAM_USAGE.set(psutil.virtual_memory().used)
            # VRAM tracking would use pynvml in production if NVIDIA GPU is present
            VRAM_USAGE.set(0.0) # Placeholder or actual GPUMemory if nvidia-smi works
            logger.debug("System metrics updated successfully")
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
        time.sleep(5)

threading.Thread(target=monitor_system_resources, daemon=True).start()

# ----------------------------------------------------
# MLFLOW EXPERIMENT SETUP
# ----------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("vLLM-Inference-Runs")
    logger.info("MLflow tracking enabled")
except Exception as e:
    logger.error(f"MLflow tracking setup failed: {e}")

# ----------------------------------------------------
# PYDANTIC SCHEMAS & CONFIG
# ----------------------------------------------------
class InferenceParams(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.05
    max_tokens: int = 512
    repetition_penalty: float = 1.1

class ModelDownloadRequest(BaseModel):
    repo_id: str  # HuggingFace repository ID, e.g., "Qwen/Qwen2.5-7B-Instruct"

class MCPSchema(BaseModel):
    name: str
    url: str
    headers: Optional[Dict[str, str]] = None
    enabled: bool = True

# Mock Downloaded Models Registry
DOWNLOADED_MODELS = [
    "meta-llama/Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

CURRENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

MCP_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "Google Search MCP",
        "url": "http://mcp-server-search:8001",
        "enabled": True,
        "tools": ["google_web_search", "google_image_search"]
    },
    {
        "name": "Weather MCP",
        "url": "http://mcp-server-weather:8002",
        "enabled": True,
        "tools": ["get_current_weather", "get_forecast"]
    }
]

# ----------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------

@app.get("/")
def read_root():
    """Возвращает веб-интерфейс (index.html) из корня проекта."""
    logger.info("Serving index.html")

    # Получаем путь к папке, где лежит main.py (папка backend)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Поднимаемся на один уровень выше (в корень проекта)
    root_dir = os.path.dirname(current_dir)

    # Формируем путь к index.html
    index_path = os.path.join(root_dir, "index.html")

    # Проверяем, существует ли файл
    if not os.path.exists(index_path):
        # Выводим путь в ошибку, чтобы при дебаге было понятно, где именно он его ищет
        return JSONResponse({"error": f"index.html not found at {index_path}"}, status_code=404)

    return FileResponse(index_path)

@app.get("/metrics")
def get_metrics():
    """Endpoint for Prometheus to scrape metrics."""
    # Обязательно передаем metrics_registry внутрь generate_latest
    return StreamingResponse(iter([generate_latest(metrics_registry)]), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/models")
def list_models():
    """List downloaded models and currently selected model."""
    return {
        "downloaded_models": DOWNLOADED_MODELS,
        "current_model": CURRENT_MODEL
    }

@app.post("/api/models/select")
def select_model(model_name: str):
    """Switch active model for inference."""
    global CURRENT_MODEL
    if model_name not in DOWNLOADED_MODELS:
        raise HTTPException(status_code=400, detail="Model is not downloaded yet. Please trigger download first.")
    CURRENT_MODEL = model_name
    return {"status": "success", "current_model": CURRENT_MODEL}

@app.post("/api/models/download")
def download_model(request: ModelDownloadRequest):
    """Trigger background downloading of HuggingFace models."""
    if request.repo_id in DOWNLOADED_MODELS:
        return {"status": "already_exists", "model": request.repo_id}
    
    # In production, this would trigger `huggingface_hub.snapshot_download` in a background thread
    DOWNLOADED_MODELS.append(request.repo_id)
    return {"status": "download_started", "model": request.repo_id}

@app.get("/api/mcp")
def list_mcp():
    """List Model Context Protocol adapters."""
    return MCP_REGISTRY

@app.post("/api/mcp")
def add_mcp(mcp: MCPSchema):
    """Register or save a new MCP."""
    new_mcp = {
        "name": mcp.name,
        "url": mcp.url,
        "enabled": mcp.enabled,
        "tools": ["fetch_url_content", "execute_command"]
    }
    MCP_REGISTRY.append(new_mcp)
    return {"status": "success", "mcp": new_mcp}

@app.post("/api/chat/session")
def create_session(db: Session = Depends(get_db)):
    """Create a new chat session stored in DB."""
    import uuid
    session_id = str(uuid.uuid4())
    db_session = ChatSession(id=session_id, model_name=CURRENT_MODEL)
    db.add(db_session)
    db.commit()
    ACTIVE_SESSIONS.inc()
    return {"session_id": session_id}

@app.get("/api/chat/history/{session_id}")
def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """Retrieve dialogue history for a session."""
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "tokens": msg.tokens_generated,
            "time_ms": msg.generation_time_ms
        }
        for msg in messages
    ]

@app.post("/api/chat/send")
async def send_chat_message(
    session_id: str = Form(...),
    message: str = Form(...),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    min_p: float = Form(0.05),
    max_tokens: int = Form(512),
    repetition_penalty: float = Form(1.1),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Handle a user message and stream back model output.
    Simulates token-by-token generation, saving histories to SQLite and tracking via Prometheus + MLflow.
    """
    start_time = time.time()
    
    # Save user message to database
    user_msg = ChatMessage(session_id=session_id, role="user", content=message)
    db.add(user_msg)
    db.commit()

    # Log file metadata if present
    file_info = ""
    if file:
        file_info = f"\n*[Attached file: {file.filename}, Type: {file.content_type}]*\n"
    
    # Simulate LLM Response streaming based on selected model and parameters
    def response_generator():
        # In actual deployment, we initialize vLLM/HF pipeline:
        # model = AutoModelForCausalLM.from_pretrained(CURRENT_MODEL)
        # tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL)
        
        full_response = f"This is a simulated VLLM response from model **{CURRENT_MODEL}** using FastAPI and LangChain.\n\n"
        if file_info:
            full_response += f"I have successfully analyzed your attached file: **{file.filename}**.\n\n"
        
        full_response += "Here is some markdown text to demonstrate immediate rendering capabilities:\n"
        full_response += "1. **Fast Generation**: Speed optimized via vLLM PagedAttention.\n"
        full_response += "2. **Monitoring**: Sending real-time telemetry to Prometheus scraper on port 9090.\n"
        full_response += "3. **LangGraph Agentic Routing**: MCP tools are scanned for context enrichment.\n\n"
        full_response += "```python\n# Model Parameters Used\n"
        full_response += f"params = {{\n    'temperature': {temperature},\n    'top_p': {top_p},\n    'min_p': {min_p},\n    'max_tokens': {max_tokens},\n    'repetition_penalty': {repetition_penalty}\n}}\n```"
        
        words = full_response.split(" ")
        generated_tokens = 0
        current_text = ""
        
        for i, word in enumerate(words):
            chunk = word + " "
            current_text += chunk
            generated_tokens += 1
            
            # Simulated token rate (approx 40-60 tokens per second)
            time.sleep(0.02)
            yield f"data: {chunk}\n\n"

        # Post-generation logs and metrics saving
        duration = time.time() - start_time
        tokens_per_sec = generated_tokens / duration if duration > 0 else 0
        
        # Save assistant message to database
        assistant_msg = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=current_text,
            tokens_generated=generated_tokens,
            generation_time_ms=duration * 1000
        )
        db.add(assistant_msg)
        db.commit()

        # Update Prometheus Metrics
        REQUEST_COUNT.labels(model=CURRENT_MODEL, status="success").inc()
        GENERATION_SPEED.labels(model=CURRENT_MODEL).observe(tokens_per_sec)
        LATENCY.labels(model=CURRENT_MODEL).observe(duration)

        # MLflow Run Tracking
        try:
            with mlflow.start_run(run_name=f"Inference_Run_{session_id[:8]}"):
                mlflow.log_param("model", CURRENT_MODEL)
                mlflow.log_param("temperature", temperature)
                mlflow.log_param("top_p", top_p)
                mlflow.log_param("min_p", min_p)
                mlflow.log_metric("latency_sec", duration)
                mlflow.log_metric("tokens_generated", generated_tokens)
                mlflow.log_metric("tokens_per_sec", tokens_per_sec)
        except Exception:
            pass

    return StreamingResponse(response_generator(), media_type="text/event-stream")

# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("FASTAPI_PORT", 8000)))
    print(f"Starting VLLM Inference FastAPI Server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
