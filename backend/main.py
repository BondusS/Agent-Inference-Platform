import os
import json
import time
import logging
import threading
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from huggingface_hub import snapshot_download, scan_cache_dir

# --- LangChain & LangGraph ---
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Для реального инференса используем HuggingFacePipeline или кастомный vLLM клиент
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mcp import MCPRegistryManager

# Prometheus & SQLAlchemy (оставил как вы просили для сбора логов)
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from sqlalchemy import create_engine, Column, String, Integer, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import datetime
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папки
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
MCP_JSON_PATH = os.path.join(ROOT_DIR, "mcps.json")

# Инициализация приложения
app = FastAPI(title="vLLM Agentic Platform")
app.mount("/static", StaticFiles(directory=os.path.join(BACKEND_DIR, "static")), name="static")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

# --- База Данных для истории чата ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./inference_history.db")

# Добавляем check_same_thread только если это локальная SQLite база
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Метрики (Бэкенд) ---
metrics_registry = CollectorRegistry()
REQUEST_COUNT = Counter("vllm_request_count", "Requests", ["model"], registry=metrics_registry)


@app.on_event("startup")
def startup_event():
    # Создаем mcps.json, если его нет
    if not os.path.exists(MCP_JSON_PATH):
        with open(MCP_JSON_PATH, "w") as f:
            json.dump({}, f)

    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        import requests
        requests.get(MLFLOW_URI, timeout=1.0)
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("vLLM-Agentic-Runs")
    except Exception:
        pass


# --- Глобальное состояние ---
CURRENT_MODEL_ID = None
llm_pipeline = None
llm_chat = None
mcp_manager = None
graph_app = None


def rebuild_graph():
    """Собирает LangGraph с актуальной моделью и актуальными MCP тулами"""
    global graph_app, mcp_manager

    if mcp_manager is None:
        mcp_manager = MCPRegistryManager(MCP_JSON_PATH)

    tools = mcp_manager.get_langchain_tools()

    workflow = StateGraph(MessagesState)

    # Привязываем инструменты к модели (если модель загружена и тулы есть)
    if llm_chat and tools:
        model_with_tools = llm_chat.bind_tools(tools)
    else:
        model_with_tools = llm_chat

    def call_model(state: MessagesState):
        if not model_with_tools:
            return {"messages": [AIMessage(content="[System]: No model loaded in registry. Select a model first.")]}
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Узел LLM
    workflow.add_node("agent", call_model)

    # Если есть инструменты, добавляем узел инструментов и условные переходы (цикл)
    if tools:
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)
        # tools_condition проверяет, вернула ли модель "tool_calls". Если да -> идем в "tools". Иначе -> END
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
    else:
        # Прямой путь, если инструментов нет
        workflow.add_edge("agent", END)

    workflow.add_edge(START, "agent")
    graph_app = workflow.compile()
    logger.info(f"LangGraph rebuilt. Active tools: {len(tools)}")


def load_llm_into_memory(repo_id: str):
    """Инициализация локальной модели"""
    global CURRENT_MODEL_ID, llm_pipeline, llm_chat
    logger.info(f"Loading model {repo_id} into memory. This may take time...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, return_full_text=False)
        llm_pipeline = HuggingFacePipeline(pipeline=pipe)

        # Оборачиваем пайплайн в Chat модель, чтобы она умела использовать bind_tools
        llm_chat = ChatHuggingFace(llm=llm_pipeline)

        CURRENT_MODEL_ID = repo_id
        logger.info("Model loaded successfully!")

        # Пересобираем граф с новой моделью
        rebuild_graph()
    except Exception as e:
        logger.error(f"Error loading model: {e}")


# Инициализируем граф без модели при старте
rebuild_graph()


# --- LangGraph Agent Setup ---
def get_agent_graph():
    workflow = StateGraph(MessagesState)

    def call_model(state: MessagesState):
        if not llm_pipeline:
            return {"messages": [AIMessage(content="[System]: No model loaded in registry. Select a model first.")]}

        # Инференс через LangChain Pipeline
        response_text = llm_pipeline.invoke(state["messages"])
        return {"messages": [AIMessage(content=response_text)]}

    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()


graph_app = get_agent_graph()


# --- API Endpoints ---
@app.get("/")
def read_root():
    return FileResponse(os.path.join(ROOT_DIR, "index.html"))


@app.get("/metrics")
def get_metrics():
    return StreamingResponse(iter([generate_latest(metrics_registry)]), media_type=CONTENT_TYPE_LATEST)


# Модели
@app.get("/api/models")
def list_models():
    """Чтение скачанных моделей из HuggingFace Cache"""
    try:
        cache = scan_cache_dir()
        downloaded = [repo.repo_id for repo in cache.repos if repo.repo_type == "model"]
    except Exception:
        downloaded = []
    return {"downloaded_models": downloaded, "current_model": CURRENT_MODEL_ID}


class ModelDownloadRequest(BaseModel):
    repo_id: str


@app.post("/api/models/download")
def download_model(request: ModelDownloadRequest, bg_tasks: BackgroundTasks):
    def pull_hf(repo):
        logger.info(f"Starting snapshot download for {repo}")
        snapshot_download(repo_id=repo)
        logger.info(f"Finished downloading {repo}")

    bg_tasks.add_task(pull_hf, request.repo_id)
    return {"status": "started"}


@app.post("/api/models/select")
def select_model(model_name: str, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(load_llm_into_memory, model_name)
    return {"status": "loading_in_background"}


# --- MCP Коннекторы (JSON Editor) ---
@app.get("/api/mcp")
def get_mcp_config():
    """Отдает текущее содержимое mcps.json"""
    if not os.path.exists(MCP_JSON_PATH):
        return {}
    with open(MCP_JSON_PATH, "r") as f:
        return json.load(f)


@app.post("/api/mcp/bulk")
def save_mcp_config_bulk(data: Dict[str, Any]):
    """Перезаписывает mcps.json целиком (используется редактором из UI)"""
    with open(MCP_JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    # <-- НОВОЕ: При сохранении JSON сразу перечитываем тулы с серверов и обновляем агента
    global mcp_manager
    mcp_manager.load_config()
    rebuild_graph()

    return {"status": "success"}


# Чат и LangGraph Inference
@app.post("/api/chat/session")
def create_session(db: Session = Depends(get_db)):
    import uuid
    session_id = str(uuid.uuid4())
    db.add(ChatSession(id=session_id))
    db.commit()
    return {"session_id": session_id}


@app.post("/api/chat/send")
async def send_chat_message(
        session_id: str = Form(...),
        message: str = Form(...),
        db: Session = Depends(get_db)
):
    db.add(ChatMessage(session_id=session_id, role="user", content=message))
    db.commit()

    async def stream_langgraph():
        REQUEST_COUNT.labels(model=str(CURRENT_MODEL_ID)).inc()
        # Извлекаем историю из БД
        history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(
            ChatMessage.created_at).all()
        lc_messages = []
        for msg in history:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            else:
                lc_messages.append(AIMessage(content=msg.content))

        # Запускаем граф LangGraph в режиме потока (если модель поддерживает, иначе отдаст целиком)
        full_response = ""
        try:
            async for event in graph_app.astream_events({"messages": lc_messages}, version="v1"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    full_response += chunk
                    yield f"data: {chunk}\n\n"
                elif kind == "on_chain_end" and "agent" in event["name"] and not full_response:
                    # Fallback для синхронных не потоковых моделей LangChain
                    outputs = event["data"].get("output", {})
                    if "messages" in outputs:
                        full_response = outputs["messages"][-1].content
                        # Разбиваем по словам для эффекта стриминга на UI
                        for word in full_response.split(" "):
                            yield f"data: {word} \n\n"
                            time.sleep(0.05)
        except Exception as e:
            yield f"data: Error executing LangGraph: {str(e)}\n\n"
            full_response = f"Error: {str(e)}"

        # Сохранение ответа
        db.add(ChatMessage(session_id=session_id, role="assistant", content=full_response))
        db.commit()

    return StreamingResponse(stream_langgraph(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
