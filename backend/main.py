import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from models.matter import ChatRequest, ChatResponse, IntakeResult, MatterSubmission
from orchestrator.pipeline import CounselOSPipeline
from agents.chat import ChatAgent
from core.llm import get_llm_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Single pipeline instance — initialised at startup, shared across requests
pipeline: CounselOSPipeline | None = None
chat_agent: ChatAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, chat_agent
    logger.info("Initialising CounselOS pipeline...")
    pipeline = CounselOSPipeline()
    llm = get_llm_provider()
    chat_agent = ChatAgent(llm)
    logger.info("Pipeline ready.")
    yield
    logger.info("CounselOS shutting down.")


app = FastAPI(
    title="CounselOS API",
    description="Multi-agent AI Co-Worker for corporate legal matter intake.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "CounselOS"}


@app.post("/api/intake", response_model=IntakeResult)
def intake(submission: MatterSubmission) -> IntakeResult:
    """
    Submit a legal matter for AI-powered intake processing.
    Returns a structured intake result with classification, risk assessment,
    and attorney assignment recommendation.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    try:
        result = pipeline.run(submission)
        return result
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask a follow-up question about a matter's analysis.
    Requires the matter to have been previously processed.
    """
    result = CounselOSPipeline.get_cached_result(request.matter_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Matter {request.matter_id} not found in cache.")

    if chat_agent is None:
        raise HTTPException(status_code=503, detail="Chat agent not initialised.")

    try:
        return chat_agent.answer(
            message=request.message,
            result=result,
            conversation_history=request.conversation_history,
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
