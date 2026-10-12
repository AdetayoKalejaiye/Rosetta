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

    @property
    def model_status(self) -> Dict:
        m = (self.run_data or {}).get("manifest", {}).get("model", {})
        return {"id": m.get("id"), "kind": m.get("kind", "unknown"),
                "inferred": m.get("inferred", False),
                "loaded": self.adapter is not None}

    def ensure_model(self, progress=None, checkpoint: Optional[str] = None):
        """Load the run's model so new interventions can be run. Expensive.

        The toy model is regenerated from code (deterministic given the seed),
        so it needs no checkpoint. Everything else is loaded from the id in
        the run manifest.
        """
        if self.adapter is not None:
            return
        say = progress or (lambda _m: None)

        from ..store import EvidenceStore
        from ..tasks import build_task_suite

        model_info = (self.run_data or {}).get("manifest", {}).get("model", {})
        kind = model_info.get("kind", "unknown")
        model_id = checkpoint or model_info.get("id")

        if kind == "toy" and not checkpoint:
            import torch
            from ..model import (ModelConfig, ToyAdapter, ToyGQATransformer,
                                 train_toy_on_induction)
            steps = model_info.get("train_steps", 400)
            say(f"Rebuilding toy model ({steps} training steps)")
            torch.manual_seed(0)
            cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=3,
                              n_heads=6, n_kv_heads=2)
            model = ToyGQATransformer(cfg)
            train_toy_on_induction(model, steps=steps, seed=0)
            self.adapter = ToyAdapter(model)
        elif model_id:
            from ..model import HFAdapter
            say(f"Loading {model_id} — first load takes a few seconds")
            self.adapter = HFAdapter.from_pretrained(model_id)
            got = self.adapter.cfg
            want = (self.run_data or {}).get("manifest", {}).get("arch", {})
            if want.get("n_layers") and (
                    got.n_layers != want["n_layers"]
                    or got.n_heads != want["n_heads"]):
                self.adapter = None
                raise RuntimeError(
                    f"{model_id} is {got.n_layers}L x {got.n_heads}H but run "
                    f"'{self.run_id}' was made with "
                    f"{want['n_layers']}L x {want['n_heads']}H — wrong "
                    "checkpoint.")
        else:
            raise RuntimeError(
                f"Run '{self.run_id}' does not record which checkpoint "
                "produced it, and its shape "
                f"({self.run_data['n_layers']}L x {self.run_data['n_heads']}H) "
                "matches no known model. Name the checkpoint explicitly to "
                "load it.")

        say("Building prompt families")
        self.families = build_task_suite(self.adapter.cfg.vocab_size, seed=0)
        self.store = EvidenceStore(os.path.join(self.run_path, "store"))

        # The checkpoint loaded and matched the run's shape, so the guess is
        # now a verified fact — record it and stop inferring.
        if model_info.get("inferred"):
            from .runs import write_manifest
            self.run_data["manifest"] = write_manifest(self.run_path,
                                                       self.adapter)
        say(f"Model ready: {self.adapter.name}")


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

    @app.get("/api/model")
    async def model_status():
        return session.model_status

    @app.post("/api/model/load")
    async def model_load(body: Dict = None):
        """Load the run's model up front. Optional {'checkpoint': '...'}."""
        if not session.run_data:
            raise HTTPException(400, "No run selected")
        ckpt = (body or {}).get("checkpoint")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: session.ensure_model(checkpoint=ckpt))
        except Exception as e:
            raise HTTPException(400, str(e))
        return session.model_status

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
