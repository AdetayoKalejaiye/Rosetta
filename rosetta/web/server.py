"""FastAPI server for the Rosetta web interface.

Serves the frontend, manages model state, and routes WebSocket messages
to the agent for LLM-driven interpretability analysis.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Session:
    adapter: Any = None
    families: Optional[Dict] = None
    store: Any = None
    out_dir: str = "runs/web"
    candidates: List = field(default_factory=list)

    def load_toy_model(self, steps: int = 400, seed: int = 0):
        from ..model import ModelConfig, ToyAdapter, ToyGQATransformer, train_toy_on_induction
        from ..store import EvidenceStore
        from ..tasks import build_task_suite

        torch.manual_seed(seed)
        cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=3,
                          n_heads=6, n_kv_heads=2)
        model = ToyGQATransformer(cfg)
        train_toy_on_induction(model, steps=steps, seed=seed)
        self.adapter = ToyAdapter(model)
        self.families = build_task_suite(cfg.vocab_size, seed=seed)
        os.makedirs(self.out_dir, exist_ok=True)
        self.store = EvidenceStore(os.path.join(self.out_dir, "store"))

    def load_hf_model(self, name_or_path: str, seed: int = 0):
        from ..model import HFAdapter
        from ..store import EvidenceStore
        from ..tasks import build_task_suite

        self.adapter = HFAdapter.from_pretrained(name_or_path)
        self.families = build_task_suite(self.adapter.cfg.vocab_size, seed=seed)
        os.makedirs(self.out_dir, exist_ok=True)
        self.store = EvidenceStore(os.path.join(self.out_dir, "store"))

    @property
    def model_info(self) -> Dict:
        if not self.adapter:
            return {"loaded": False}
        cfg = self.adapter.cfg
        return {
            "loaded": True,
            "name": self.adapter.name,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "d_model": cfg.d_model,
            "vocab_size": cfg.vocab_size,
            "head_dim": cfg.head_dim,
            "group_size": cfg.group_size,
        }


def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def create_app(out_dir: str = "runs/web", auto_load_toy: bool = True) -> FastAPI:
    _load_dotenv()

    app = FastAPI(title="Rosetta")
    session = Session(out_dir=out_dir)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def startup():
        if auto_load_toy:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, session.load_toy_model)

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/model")
    async def model_info():
        return session.model_info

    @app.get("/api/results")
    async def results():
        if not session.store:
            return {"profiles": {}, "evidence": []}
        profiles = {n: p.to_dict() for n, p in session.store.profiles.items()}
        evidence = [r.to_dict() for r in session.store.all_records()]
        return {"profiles": profiles, "evidence": evidence}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        from .agent import RosettaAgent
        agent = RosettaAgent(session)

        async def send(msg: Dict):
            await ws.send_json(msg)

        await send({"type": "model_info", "data": session.model_info})
        if session.adapter:
            await send({"type": "chat", "sender": "system",
                        "message": f"Model loaded: **{session.adapter.name}** "
                                   f"({session.adapter.cfg.n_layers}L, "
                                   f"{session.adapter.cfg.n_heads} heads). "
                                   "Ask me anything about its internals."})
        else:
            await send({"type": "chat", "sender": "system",
                        "message": "No model loaded yet. Say 'load toy' to get started."})

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)

                if msg["type"] == "chat":
                    await agent.handle_message(msg["message"], send)
                elif msg["type"] == "load":
                    model = msg.get("model", "toy")
                    await send({"type": "status", "message": f"Loading {model}..."})
                    loop = asyncio.get_event_loop()
                    if model == "toy":
                        await loop.run_in_executor(None, session.load_toy_model)
                    else:
                        await loop.run_in_executor(
                            None, lambda: session.load_hf_model(model))
                    await send({"type": "model_info", "data": session.model_info})
                    await send({"type": "chat", "sender": "system",
                                "message": f"Model **{session.adapter.name}** loaded."})
        except WebSocketDisconnect:
            pass

    return app
