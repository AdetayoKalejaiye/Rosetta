"""Model access layer.

ModelAdapter exposes four hook points per layer, shared by every method:

  HEAD_OUT : per-(query)-head attention output, before o_proj mixes heads.
             Editing a slice here intervenes on ONE head's output.
  KV_K/KV_V: per-KV-head keys/values after k_proj/v_proj. With grouped-query
             attention (e.g. SmolLM2-360M: 15 query heads sharing 5 KV heads),
             editing one KV head touches a WHOLE GROUP of query heads. Rosetta
             keeps these two intervention types distinct on purpose.
  MLP_OUT  : the MLP block's output contribution to the residual stream.
  RESID    : residual stream after each block.

Backends:
  ToyGQATransformer - small Llama-style decoder with GQA, used to exercise the
                      whole pipeline without downloading a real checkpoint.
  HFAdapter         - wraps a HuggingFace Llama-family model (SmolLM2 etc.)
                      via the same module-path spec. Written but NOT exercised
                      here (per project constraint: no real-model runs).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hook point names
HEAD_OUT = "head_out"
KV_K = "kv_k"
KV_V = "kv_v"
MLP_OUT = "mlp_out"
RESID = "resid"

EditFn = Callable[[torch.Tensor], torch.Tensor]
HookKey = Tuple[str, int]  # (point, layer)


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_seq: int = 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads


# ============================================================================ toy model


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dh, H, K = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, H * dh, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, K * dh, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, K * dh, bias=False)
        self.o_proj = nn.Linear(H * dh, cfg.d_model, bias=False)
        self.last_attn: Optional[torch.Tensor] = None  # [B,H,S,S] when captured
        self.capture_attn = False

    def forward(self, x):
        B, S, _ = x.shape
        cfg = self.cfg
        dh, H, K, G = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.group_size
        q = self.q_proj(x).view(B, S, H, dh).transpose(1, 2)          # [B,H,S,dh]
        k = self.k_proj(x).view(B, S, K, dh).transpose(1, 2)          # [B,K,S,dh]
        v = self.v_proj(x).view(B, S, K, dh).transpose(1, 2)
        k = k.repeat_interleave(G, dim=1)                              # [B,H,S,dh]
        v = v.repeat_interleave(G, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = scores.softmax(-1)
        if self.capture_attn:
            self.last_attn = probs.detach()
        out = probs @ v                                                # [B,H,S,dh]
        out = out.transpose(1, 2).reshape(B, S, H * dh)                # pre-o_proj
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden_mult=4):
        super().__init__()
        h = cfg.d_model * hidden_mult
        self.gate_proj = nn.Linear(cfg.d_model, h, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, h, bias=False)
        self.down_proj = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model)
        self.self_attn = GQAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ToyGQATransformer(nn.Module):
    """Small decoder with learned absolute positions (enough for a toy)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens):
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device)
        x = self.embed_tokens(tokens) + self.pos_emb(pos)[None]
        for blk in self.layers:
            x = blk(x)
        return self.lm_head(self.norm(x))


# ============================================================================ adapter


class ModelAdapter:
    """Uniform hooked access. Subclasses only supply module getters."""

    def __init__(self, model: nn.Module, cfg: ModelConfig, name: str = "model"):
        self.model = model
        self.cfg = cfg
        self.name = name
        self.model.eval()

    # ---- backend-specific ------------------------------------------------
    def _attn(self, layer: int) -> nn.Module: ...
    def _o_proj(self, layer: int) -> nn.Module: ...
    def _k_proj(self, layer: int) -> nn.Module: ...
    def _v_proj(self, layer: int) -> nn.Module: ...
    def _mlp_down(self, layer: int) -> nn.Module: ...
    def _block(self, layer: int) -> nn.Module: ...
    def _final_norm_weight(self) -> torch.Tensor: ...
    def _unembed(self) -> torch.Tensor:
        """W_U with shape [vocab, d_model]."""
        ...
    def forward(self, tokens: torch.Tensor) -> torch.Tensor: ...
    def attention_patterns(self, tokens: torch.Tensor) -> torch.Tensor:
        """[n_layers, B, n_heads, S, S]"""
        ...

    # ---- shared hook machinery ------------------------------------------
    @contextlib.contextmanager
    def _hooks(self, capture: List[HookKey], edits: Dict[HookKey, EditFn],
               cache: Dict[HookKey, torch.Tensor], detach: bool = True):
        handles = []
        keys = set(capture) | set(edits.keys())

        def make_pre(key):  # pre-hook on a Linear: sees/edits its input
            def pre(mod, args):
                x = args[0]
                if key in edits:
                    x = edits[key](x)
                if key in capture:
                    cache[key] = x.detach().clone() if detach else x
                return (x,) + tuple(args[1:])
            return pre

        def make_post(key):  # post-hook: sees/edits a module's output
            def post(mod, args, out):
                if key in edits:
                    out = edits[key](out)
                if key in capture:
                    cache[key] = out.detach().clone() if detach else out
                return out
            return post

        for (point, layer) in keys:
            if point == HEAD_OUT:
                handles.append(self._o_proj(layer).register_forward_pre_hook(
                    make_pre((point, layer))))
            elif point == KV_K:
                handles.append(self._k_proj(layer).register_forward_hook(
                    make_post((point, layer))))
            elif point == KV_V:
                handles.append(self._v_proj(layer).register_forward_hook(
                    make_post((point, layer))))
            elif point == MLP_OUT:
                handles.append(self._mlp_down(layer).register_forward_hook(
                    make_post((point, layer))))
            elif point == RESID:
                handles.append(self._block(layer).register_forward_hook(
                    make_post((point, layer))))
            else:
                raise ValueError(point)
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def run(self, tokens, capture: Optional[List[HookKey]] = None,
            edits: Optional[Dict[HookKey, EditFn]] = None,
            grad: bool = False):
        """Forward pass with optional capture/edit hooks.

        Returns (logits, cache). With grad=True, captured tensors stay in the
        graph and have retain_grad() called (for attribution patching).
        """
        capture = capture or []
        edits = edits or {}
        cache: Dict[HookKey, torch.Tensor] = {}
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx, self._hooks(capture, edits, cache, detach=not grad):
            logits = self.forward(tokens)
        if grad:
            for t in cache.values():
                if t.requires_grad:
                    t.retain_grad()
        return logits, cache

    # ---- head-slice helpers ----------------------------------------------
    def head_slice(self, head: int) -> slice:
        dh = self.cfg.head_dim
        return slice(head * dh, (head + 1) * dh)

    def kv_slice(self, kv_head: int) -> slice:
        dh = self.cfg.head_dim
        return slice(kv_head * dh, (kv_head + 1) * dh)

    def query_heads_for_kv(self, kv_head: int) -> List[int]:
        g = self.cfg.group_size
        return list(range(kv_head * g, (kv_head + 1) * g))

    def edit_head(self, head: int, replace: Callable[[torch.Tensor], torch.Tensor]) -> EditFn:
        """EditFn over the full [B,S,H*dh] tensor that only touches one head."""
        sl = self.head_slice(head)

        def fn(x):
            x = x.clone()
            x[..., sl] = replace(x[..., sl])
            return x
        return fn

    def edit_kv_head(self, kv_head: int, replace) -> EditFn:
        sl = self.kv_slice(kv_head)

        def fn(x):
            x = x.clone()
            x[..., sl] = replace(x[..., sl])
            return x
        return fn


class ToyAdapter(ModelAdapter):
    def __init__(self, model: ToyGQATransformer, name="toy-gqa"):
        super().__init__(model, model.cfg, name)

    def _attn(self, l): return self.model.layers[l].self_attn
    def _o_proj(self, l): return self.model.layers[l].self_attn.o_proj
    def _k_proj(self, l): return self.model.layers[l].self_attn.k_proj
    def _v_proj(self, l): return self.model.layers[l].self_attn.v_proj
    def _mlp_down(self, l): return self.model.layers[l].mlp.down_proj
    def _block(self, l): return self.model.layers[l]

    def _final_norm_weight(self):
        return self.model.norm.weight

    def _unembed(self):
        return self.model.lm_head.weight  # [vocab, d_model]

    def forward(self, tokens):
        return self.model(tokens)

    def attention_patterns(self, tokens):
        for l in range(self.cfg.n_layers):
            self._attn(l).capture_attn = True
        with torch.no_grad():
            self.model(tokens)
        pats = torch.stack([self._attn(l).last_attn for l in range(self.cfg.n_layers)])
        for l in range(self.cfg.n_layers):
            self._attn(l).capture_attn = False
            self._attn(l).last_attn = None
        return pats  # [L,B,H,S,S]


class HFAdapter(ModelAdapter):
    """Adapter for HuggingFace Llama-family checkpoints (SmolLM2, Llama, ...).

    NOTE: written to the same interface but intentionally not exercised in
    this build (constraint: no runs against a real model). When you do point
    it at SmolLM2-360M, remember its GQA layout: 15 query heads sharing 5 KV
    heads per layer, so kv_head interventions hit 3 query heads at once.
    """

    @classmethod
    def from_pretrained(cls, name_or_path: str, device="cpu", dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype)
        model.to(device)
        hc = model.config
        cfg = ModelConfig(
            vocab_size=hc.vocab_size,
            d_model=hc.hidden_size,
            n_layers=hc.num_hidden_layers,
            n_heads=hc.num_attention_heads,
            n_kv_heads=getattr(hc, "num_key_value_heads", hc.num_attention_heads),
            max_seq=getattr(hc, "max_position_embeddings", 2048),
        )
        adapter = cls(model, cfg, name=name_or_path)
        adapter.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        return adapter

    def _layers(self):
        return self.model.model.layers

    def _attn(self, l): return self._layers()[l].self_attn
    def _o_proj(self, l): return self._layers()[l].self_attn.o_proj
    def _k_proj(self, l): return self._layers()[l].self_attn.k_proj
    def _v_proj(self, l): return self._layers()[l].self_attn.v_proj
    def _mlp_down(self, l): return self._layers()[l].mlp.down_proj
    def _block(self, l): return self._layers()[l]

    def _final_norm_weight(self):
        return self.model.model.norm.weight

    def _unembed(self):
        return self.model.lm_head.weight

    def forward(self, tokens):
        return self.model(tokens).logits

    def attention_patterns(self, tokens):
        with torch.no_grad():
            out = self.model(tokens, output_attentions=True)
        return torch.stack(out.attentions)  # [L,B,H,S,S]

    # HF blocks return tuples; RESID hooks need unwrapping.
    @contextlib.contextmanager
    def _hooks(self, capture, edits, cache, detach=True):
        # Wrap block outputs: intercept tuple element 0 for RESID.
        resid_keys = {k for k in (set(capture) | set(edits)) if k[0] == RESID}
        other_capture = [k for k in capture if k[0] != RESID]
        other_edits = {k: v for k, v in edits.items() if k[0] != RESID}
        handles = []

        def make_block_post(key):
            def post(mod, args, out):
                hidden = out[0] if isinstance(out, tuple) else out
                if key in edits:
                    hidden = edits[key](hidden)
                if key in capture:
                    cache[key] = hidden.detach().clone() if detach else hidden
                if isinstance(out, tuple):
                    return (hidden,) + tuple(out[1:])
                return hidden
            return post

        for key in resid_keys:
            handles.append(self._block(key[1]).register_forward_hook(make_block_post(key)))
        try:
            with super()._hooks(other_capture, other_edits, cache, detach):
                yield
        finally:
            for h in handles:
                h.remove()


# ============================================================================ training the toy


def train_toy_on_induction(model: ToyGQATransformer, steps=400, batch=32,
                           seq_len=24, lr=3e-3, seed=0, verbose=False):
    """Teach the toy model the repeated-sequence task so ablation/patching
    results are non-trivial. Sequences look like  [x1..xk, x1..xk]  and loss is
    taken on the repeated half, which a small model solves with induction-style
    attention. This trains the TOY only - no external checkpoint involved."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    half = seq_len // 2
    V = model.cfg.vocab_size
    for step in range(steps):
        first = torch.randint(2, V, (batch, half), generator=g)
        seq = torch.cat([first, first], dim=1)
        logits = model(seq[:, :-1])
        targets = seq[:, 1:].clone()
        targets[:, : half - 1] = -100  # only score the repeated half
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 100 == 0:
            print(f"  toy train step {step}: loss {loss.item():.3f}")
    model.eval()
    return model
