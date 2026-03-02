"""
AutoHeal AI – FastAPI entry point for Render deployment.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AutoHeal AI",
    description="Intelligent System Failure Detection & Self-Recovery Engine",
    version="1.0.0",
)

_frontend_url = os.getenv("FRONTEND_URL", "https://your-frontend.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "AutoHeal AI"}


@app.get("/health")
def health():
    return {"status": "healthy"}
