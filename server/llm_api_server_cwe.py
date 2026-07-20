#!/usr/bin/env python3
"""OpenAI-compatible LLM API server with CWE detection."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
import uvicorn
import torch
import time
import json
import gc
import logging
from devcwe import LRModel
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_instance: Optional[LRModel] = None
model_config: Dict[str, Any] = {}
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS_DEFAULT", "4096"))


def cleanup_cuda_cache():
    """Clear Python and CUDA caches after each request."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 12000
    stream: Optional[bool] = False
    use_cwe_detection: Optional[bool] = None
    cwe_check_interval: Optional[int] = 5
    use_traceguard: Optional[bool] = None
    cwe_injection_mode: Optional[str] = "immediate"
    cwe_reconstruction_model_type: Optional[str] = "auto"
    top_k: Optional[int] = 10
    repetition_penalty: Optional[float] = 1.2
    do_sample: Optional[bool] = True


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("Initializing model...")
    init_model()
    logger.info("Model initialized")
    yield
    logger.info("Cleaning up resources...")
    global model_instance
    model_instance = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="CWE Detection LLM API",
    description="OpenAI-compatible API with CWE detection",
    version="1.0.0",
    lifespan=lifespan
)


def init_model():
    """Initialize the model."""
    global model_instance, model_config
    
    model_path = os.getenv("MODEL_PATH", "../model/deepseek_14B")
    device_id = int(os.getenv("DEVICE_ID", "4"))
    torch_dtype_str = os.getenv("TORCH_DTYPE", "float16")
    log_dir = os.getenv("CWE_LOG_DIR", "cwe_logs")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch_dtype = torch.float16 if torch_dtype_str == "float16" else torch.float32
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_dir = f"{log_dir}/session_{timestamp}"
    
    model_instance = LRModel(model_path, device, torch_dtype, log_dir=session_log_dir, max_injections_per_rule=3)
    
    model_config = {
        "model_path": model_path,
        "device": str(device),
        "torch_dtype": torch_dtype_str,
        "log_dir": session_log_dir,
    }
    
    logger.info(f"Model loaded: {model_path}")
    logger.info(f"CWE log directory: {session_log_dir}")
    logger.info(f"Default max_new_tokens: {DEFAULT_MAX_NEW_TOKENS}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CWE Detection LLM API Server",
        "status": "running",
        "model_loaded": model_instance is not None
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "lrmodel-cwe-detection",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cwe-detection",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")
    
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        use_cwe_detection = request.use_cwe_detection
        if use_cwe_detection is None:
            use_cwe_detection = request.use_traceguard if request.use_traceguard is not None else True
        
        request.use_cwe_detection = use_cwe_detection
        
        logger.info(f"Handling request: use_cwe_detection={use_cwe_detection}, use_traceguard={request.use_traceguard}")
        
        request_max_tokens = (
            request.max_tokens
            if (request.max_tokens is not None and request.max_tokens > 0)
            else DEFAULT_MAX_NEW_TOKENS
        )
        gen_kwargs = {
            'max_new_tokens': request_max_tokens,
            'temperature': request.temperature or 0.1,
            'top_k': request.top_k or 10,
            'top_p': request.top_p or 0.8,
            'repetition_penalty': request.repetition_penalty or 1.2,
            'do_sample': request.do_sample if request.do_sample is not None else True,
        }
        logger.info(f"Request max_new_tokens={request_max_tokens}")
        
        if request.stream:
            return StreamingResponse(
                stream_chat_completions(request, messages, gen_kwargs),
                media_type="text/event-stream"
            )
        
        return await non_stream_chat_completions(request, messages, gen_kwargs)
        
    except Exception as e:
        logger.error(f"Error while handling request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completions(
    request: ChatCompletionRequest,
    messages: List[Dict[str, str]],
    gen_kwargs: Dict[str, Any]
):
    """Stream a chat completion response."""
    completion_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    
    try:
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        
        logger.info(f"Starting streaming generation,use_cwe_detection={request.use_cwe_detection}, cwe_check_interval={request.cwe_check_interval}")
        
        full_response = ""
        
        for token in model_instance.inference(
            messages,
            gen_kwargs,
            use_cwe_detection=request.use_cwe_detection if request.use_cwe_detection is not None else True,
            cwe_check_interval=request.cwe_check_interval or 5,
            cwe_injection_mode=request.cwe_injection_mode or "immediate",
            cwe_reconstruction_model_type=request.cwe_reconstruction_model_type or "auto"
        ):
            full_response += token
            chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"content": token},
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0)
        
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        if request.use_cwe_detection and model_instance.cwe_detector:
            logger.info(f"CWE detection logs saved to: {model_instance.log_dir}")
        
        logger.info("Streaming generation completed")
        
    except Exception as e:
        logger.error(f"Streaming generation error: {str(e)}", exc_info=True)
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        cleanup_cuda_cache()


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model output
    """
    think_end = text.rfind('`</think>`')
    
    if think_end != -1:
        final_content = text[think_end + len('`</think>`'):].strip()
        if final_content:
            return final_content
    
    return text


async def non_stream_chat_completions(
    request: ChatCompletionRequest,
    messages: List[Dict[str, str]],
    gen_kwargs: Dict[str, Any]
) -> ChatCompletionResponse:
    """Generate a non-streaming chat completion response."""
    completion_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    
    try:
        full_response = ""
        for token in model_instance.inference(
            messages,
            gen_kwargs,
            use_cwe_detection=request.use_cwe_detection if request.use_cwe_detection is not None else True,
            cwe_check_interval=request.cwe_check_interval or 5,
            cwe_injection_mode=request.cwe_injection_mode or "immediate",
            cwe_reconstruction_model_type=request.cwe_reconstruction_model_type or "auto"
        ):
            full_response += token

        final_answer = extract_final_answer(full_response)
        
        if request.use_cwe_detection and model_instance.cwe_detector:
            logger.info(f"CWE detection logs saved to: {model_instance.log_dir}")
        
        prompt_tokens = sum(len(msg["content"].split()) for msg in messages)
        completion_tokens = len(final_answer.split())
        
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=final_answer),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    finally:
        cleanup_cuda_cache()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "cuda_available": torch.cuda.is_available(),
        "config": model_config
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the CWE Detection LLM API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "llm_api_server_cwe:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )