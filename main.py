"""
Contract Intelligence Agent — FastAPI entry point.
Run locally: uvicorn main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import contracts

app = FastAPI(title="Contract Intelligence Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your Lovable/frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(contracts.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
