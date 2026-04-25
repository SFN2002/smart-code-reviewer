from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
from smart_code_reviewer.main import AsyncLifecycleManager
from smart_code_reviewer.config import ReviewConfig

app = FastAPI(title="Smart-Code-Reviewer API")
lifecycle: Optional[AsyncLifecycleManager] = None

class AnalyzeRequest(BaseModel):
    source: str
    language: str = "python"
    source_id: str

class SearchRequest(BaseModel):
    source: str
    top_k: int = 5

@app.on_event("startup")
async def startup():
    global lifecycle
    config = ReviewConfig.load()
    lifecycle = AsyncLifecycleManager(config)
    await lifecycle.bootstrap()

@app.on_event("shutdown")
async def shutdown():
    if lifecycle:
        await lifecycle.shutdown()

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    if not lifecycle:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    result = await lifecycle.review_source(request.source_id, request.source)
    return result

@app.post("/search")
async def search(request: SearchRequest):
    if not lifecycle:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    tensor = lifecycle.analyzer.analyze(request.source)
    sym_vec = lifecycle.ast_toolkit.extract_symbol_vector(request.source)
    combined = tensor.flatten()
    combined = list(combined) + list(sym_vec)
    import numpy as np
    results = await lifecycle.memory.similarity_search(np.array(combined), top_k=request.top_k)
    return {"results": results}

@app.get("/health")
async def health():
    if not lifecycle:
        return {"status": "offline"}
    count = await lifecycle.memory.get_record_count()
    return {"status": "online", "record_count": count}

async def start_server_async(host: str = "0.0.0.0", port: int = 8000):
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
