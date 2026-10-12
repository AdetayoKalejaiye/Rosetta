"""FastAPI server for the Rosetta web interface.

Run-first: finished runs are read straight off disk as JSON, so the UI is
populated the moment it opens. A torch model is loaded only when the user
asks for a new intervention.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .runs import component_detail, discover_runs, load_run

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class Session:
    runs_root: str = "runs"
    run_id: Optional[str] = None
    run_data: Optional[Dict] = None

    # only populated when a live intervention is requested
    adapter: Any = None
    families: Optional[Dict] = None
    store: Any = None
    candidates: List = field(default_factory=list)

    @property
    def run_path(self) -> Optional[str]:
        return self.run_data["path"] if self.run_data else None

    def select_run(self, run_id: str) -> Dict:
        path = os.path.join(self.runs_root, run_id)
        if not os.path.isdir(path):
            raise FileNotFoundError(run_id)
        self.run_id = run_id
        self.run_data = load_run(path)
        # a newly selected run invalidates any loaded model state
        self.adapter = None
        self.families = None
        self.store = None
        self.candidates = []
        return self.run_data

    def ensure_model(self):
        """Load a torch model for the selected run. Expensive; opt-in."""
        if self.adapter is not None:
            return
        import torch
        from ..model import (ModelConfig, ToyAdapter, ToyGQATransformer,
                             train_toy_on_induction)
        from ..store import EvidenceStore
        from ..tasks import build_task_suite

        if self.run_data and self.run_data["n_layers"] == 3 \
                and self.run_data["n_heads"] == 6:
            torch.manual_seed(0)
            cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=3,
                              n_heads=6, n_kv_heads=2)
            model = ToyGQATransformer(cfg)
            train_toy_on_induction(model, steps=400, seed=0)
            self.adapter = ToyAdapter(model)
        else:
            raise RuntimeError(
                "Live interventions on this run need its original checkpoint. "
                "Only the toy model can be rebuilt locally; for "
                f"'{self.run_id}' re-run the CLI pipeline instead.")

        self.families = build_task_suite(self.adapter.cfg.vocab_size, seed=0)
        self.store = EvidenceStore(os.path.join(self.run_path, "store"))


def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def create_app(runs_root: str = "runs") -> FastAPI:
    _load_dotenv()

    app = FastAPI(title="Rosetta")
    session = Session(runs_root=runs_root)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/runs")
    async def list_runs():
        return {"runs": discover_runs(session.runs_root),
                "selected": session.run_id}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        try:
            return session.select_run(run_id)
        except FileNotFoundError:
            raise HTTPException(404, f"No such run: {run_id}")

    @app.get("/api/runs/{run_id}/component/{comp}")
    async def get_component(run_id: str, comp: str):
        path = os.path.join(session.runs_root, run_id)
        if not os.path.isdir(path):
            raise HTTPException(404, f"No such run: {run_id}")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: component_detail(path, comp))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        from .agent import RosettaAgent
        agent = RosettaAgent(session)

        async def send(msg: Dict):
            await ws.send_json(msg)

        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if msg["type"] == "chat":
                    await agent.handle_message(msg["message"], send)
                elif msg["type"] == "select_run":
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: session.select_run(msg["run_id"]))
                    agent.reset()
        except WebSocketDisconnect:
            pass

    return app
