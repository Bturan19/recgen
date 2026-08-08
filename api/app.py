from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .serving import service

app = FastAPI(
    title="recgen-api",
    description="LLM-as-encoder embeddings and recommendation as a service. "
    "Prefill-only inference: one forward pass, no autoregressive decoding.",
    version="0.1.0",
)


class EncodeRequest(BaseModel):
    texts: list[str] = []
    rows: list[dict] = []
    fields: Optional[list[str]] = None
    instruction: str = ""
    cache_key: Optional[str] = None


class EncodeResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int


class RankRequest(BaseModel):
    context: str
    items: list[str]
    head: Optional[str] = None
    top_k: int = 10


class RankItem(BaseModel):
    item: str
    score: float


class RankResponse(BaseModel):
    ranking: list[RankItem]


@app.get("/health")
def health():
    return {"status": "ok", **service.status()}


@app.post("/v1/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    if not req.texts and not req.rows:
        raise HTTPException(status_code=400, detail="provide texts or rows")
    if len(req.texts) > 4096:
        raise HTTPException(status_code=400, detail="max 4096 texts per request")
    if req.texts:
        embs = service.encode(req.texts, cache_key=req.cache_key)
    else:
        embs = service.encode_rows(req.rows, req.fields, req.instruction)
    return {"embeddings": embs, "dim": len(embs[0]) if embs else 0, "count": len(embs)}


@app.post("/v1/rank", response_model=RankResponse)
def rank(req: RankRequest):
    if not req.items or len(req.items) > 1000:
        raise HTTPException(status_code=400, detail="items must be 1..1000 entries")
    ranking = service.rank(req.context, req.items, head_key=req.head, top_k=req.top_k)
    return {"ranking": [RankItem(**r) for r in ranking]}
