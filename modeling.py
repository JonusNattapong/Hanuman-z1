import math
from types import SimpleNamespace
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q, k, sin, cos):
    # q,k: (B, nh, T, hs)
    q_ = (q * cos) + (rotate_every_two(q) * sin)
    k_ = (k * cos) + (rotate_every_two(k) * sin)
    return q_, k_


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        sin = emb.sin()[None, None, :, :]
        cos = emb.cos()[None, None, :, :]
        return sin, cos


class RMSNorm(nn.Module):
    """Simple RMSNorm implementation compatible with HF's RMSNorm behavior."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, C)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, use_rotary=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim)

        # optional flash attention detection
        self.use_flash = False
        try:
            # try common flash attention package
            import flash_attn  # type: ignore
            self.use_flash = True
        except Exception:
            self.use_flash = False

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, nh, T, hs)

        if self.use_rotary:
            sin, cos = self.rotary(T, device=x.device)
            q, k = apply_rotary_pos_emb(q, k, sin, cos)

        if self.use_flash:
            # best-effort: if flash attention is available, try to use it (APIs vary by package)
            try:
                # flatten for flash attention calls
                qkv = torch.stack((q, k, v), dim=2)
                # fallback to manual matmul if API unknown
                raise RuntimeError('flash-attn integration placeholder; falling back')
            except Exception:
                att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        else:
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(B, 1, 1, T)
            att = att.masked_fill(attn_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # dim_out is the inner dim; we keep ability to set it equal to dim_in for smaller models
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.fc_gate = nn.Linear(dim_in, dim_out)
        self.fc2 = nn.Linear(dim_out, dim_in)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)) * self.fc_gate(x))


class FeedForward(nn.Module):
    def __init__(self, n_embd, mlp_ratio=1.0, pdrop=0.1, inner_dim=None):
        super().__init__()
        # Allow inner_dim override; default reduce to match embedding for compact model
        if inner_dim is None:
            inner = int(n_embd * mlp_ratio)
        else:
            inner = inner_dim
        self.fn = SwiGLU(n_embd, inner)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, tag_emb=None):
        # tag_emb is accepted for API compatibility with MoE variants that may use router bias
        return self.dropout(self.fn(x))


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feedforward: small top-k router routing per token.

    Notes: simplified router for resource-constrained mini models. Uses token-level routing.
    """
    def __init__(self, n_embd, num_experts=4, top_k=1, expert_ctor=None, router_temperature=1.0, aux_coef=0.0, tag_proj_dim=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.aux_coef = aux_coef
        assert 1 <= top_k <= num_experts
        if expert_ctor is None:
            expert_ctor = lambda: FeedForward(n_embd)
        self.experts = nn.ModuleList([expert_ctor() for _ in range(num_experts)])
        # lightweight router: linear to num_experts
        self.router = nn.Linear(n_embd, num_experts)
        # optional projection from a tag embedding (B, C) -> (B, num_experts) to bias router logits
        self.tag_proj = nn.Linear(tag_proj_dim, num_experts) if tag_proj_dim is not None else None

    def forward(self, x, tag_emb=None):
        # x: (B, T, C)
        B, T, C = x.size()
        logits = self.router(x)  # (B, T, num_experts)
        # if a tag embedding is provided (B, C) and we have a projection, add it as a bias
        if tag_emb is not None and self.tag_proj is not None:
            # project per-batch tag embedding to expert logits and broadcast to tokens
            # tag_emb: (B, C) -> (B, num_experts) -> (B, 1, num_experts)
            tag_bias = self.tag_proj(tag_emb).unsqueeze(1)
            logits = logits + tag_bias
        # apply temperature to router logits
        if self.router_temperature and self.router_temperature != 1.0:
            probs = F.softmax(logits / float(self.router_temperature), dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        # topk indices
        topk = probs.topk(self.top_k, dim=-1)
        indices = topk.indices  # (B, T, top_k)
        weights = topk.values  # (B, T, top_k)

        out = x.new_zeros(B, T, C)
        # naive per-expert dispatch (may be slower but simple)
        for e in range(self.num_experts):
            # mask tokens that route to expert e
            mask = (indices == e)  # (B, T, top_k)
            if not mask.any():
                continue
            # combine along top_k: compute contribution weight per (B,T)
            # for tokens where expert e selected, create input slice
            sel = mask.any(-1)  # (B, T)
            if not sel.any():
                continue
            inp = x[sel]
            expert_out = self.experts[e](inp)
            # add weighted contribution
            # weights for those selected tokens: take max across top_k positions where index==e
            w = torch.zeros(B, T, device=x.device)
            for k in range(self.top_k):
                w = w + (indices[..., k] == e).float() * weights[..., k]
            w_sel = w[sel].unsqueeze(-1)
            out[sel] = out[sel] + expert_out * w_sel

        # compute lightweight auxiliary load-balancing loss (optional)
        self.last_aux_loss = None
        if getattr(self, 'aux_coef', 0.0):
            # average probability mass per expert across tokens
            load = probs.sum(dim=(0, 1)) / (B * T)
            aux = (load * load).sum()
            self.last_aux_loss = aux * float(self.aux_coef)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, mlp_ratio=4, attn_pdrop=0.1, resid_pdrop=0.1, use_rotary=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop, use_rotary=use_rotary)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, mlp_ratio, resid_pdrop)

    def forward(self, x, attn_mask=None, tag_emb=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        # allow mlp variants (MoE) to accept tag_emb
        x = x + (self.mlp(self.ln2(x), tag_emb=tag_emb) if hasattr(self.mlp, '__call__') else self.mlp(self.ln2(x)))
        return x


class Hanuman(nn.Module):
    """Hanuman: advanced GPT-like mini model with rotary embeddings and SwiGLU MLP.

    Compatible forward signature with HF GPT2LMHeadModel: forward(input_ids, attention_mask, labels)
    Returns SimpleNamespace(loss=..., logits=...)
    """

    def __init__(self, *, vocab_size, n_positions=4096, n_embd=512, n_layer=8, n_head=8, mlp_ratio=1.0,
                 attn_pdrop=0.1, resid_pdrop=0.1, use_rotary=True, use_rmsnorm=True, use_moe=False,
                 moe_experts=4, moe_top_k=1, gradient_checkpointing=False, use_think_head=False, think_aux_coef=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd

        self.use_rmsnorm = use_rmsnorm
        self.gradient_checkpointing = gradient_checkpointing

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            blk = TransformerBlock(n_embd, n_head, mlp_ratio, attn_pdrop, resid_pdrop, use_rotary=use_rotary)
            self.blocks.append(blk)

        # final norm: RMSNorm or LayerNorm
        if use_rmsnorm:
            self.ln_f = RMSNorm(n_embd)
        else:
            self.ln_f = nn.LayerNorm(n_embd)

        # optional MoE on top of feedforwards inside blocks: swap block.mlp with MoE variant
        if use_moe:
            for blk in self.blocks:
                blk.mlp = MoEFeedForward(n_embd, num_experts=moe_experts, top_k=moe_top_k,
                                          expert_ctor=lambda: FeedForward(n_embd, mlp_ratio=mlp_ratio, inner_dim=n_embd))

        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # optional think head for intermediate reasoning outputs (same vocab by default)
        self.use_think_head = use_think_head
        self.think_aux_coef = float(think_aux_coef)
        if use_think_head:
            self.think_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None, thought_labels=None):
        B, T = input_ids.size()
        assert T <= self.n_positions, f"Sequence length {T} > model max {self.n_positions}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        # If user provided a special effort tag token (e.g., first token in input), compute tag_emb
        tag_emb = None
        try:
            # detect if first token corresponds to a special think token id set on the model
            if hasattr(self, 'think_token_ids') and isinstance(self.think_token_ids, dict):
                # look for a single-tag indicator in input_ids (assumed at position 0)
                first = input_ids[:, 0]
                # if a known tag id is present, make tag_emb from its token embedding
                for tag, tid in self.think_token_ids.items():
                    if (first == tid).any():
                        tag_emb = self.wte(tid).unsqueeze(0).expand(input_ids.size(0), -1)
                        break
        except Exception:
            tag_emb = None

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(blk, x, attention_mask, tag_emb)
            else:
                x = blk(x, attn_mask=attention_mask, tag_emb=tag_emb)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        thought_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = lm_loss

        # optional thinking head loss
        thought_logits = None
        if self.use_think_head and thought_labels is not None:
            thought_logits = self.think_head(x)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            thought_loss = loss_fct(thought_logits.view(-1, thought_logits.size(-1)), thought_labels.view(-1))
            if loss is None:
                loss = thought_loss * self.think_aux_coef
            else:
                loss = loss + thought_loss * self.think_aux_coef

        return SimpleNamespace(loss=loss, logits=logits, thought_logits=thought_logits, thought_loss=thought_loss)

    # runtime helpers
    def to_device(self, device):
        self.to(device)

    def enable_fp16(self):
        # cast model params to float16 where safe
        self.half()

    def set_gradient_checkpointing(self, enabled: bool):
        self.gradient_checkpointing = enabled

    # Simple autoregressive generator (CPU/GPU). Not optimized for speed.
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=0, top_p=0.0, eos_token_id=None):
        device = input_ids.device
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=out).logits
            next_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k > 0:
                vals, idx = torch.topk(next_logits, top_k)
                probs = torch.zeros_like(next_logits).scatter(1, idx, F.softmax(vals, dim=-1))
            elif top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative_probs > top_p
                cutoff_index = torch.argmax(cutoff.int(), dim=-1)
                mask = torch.zeros_like(sorted_logits).bool()
                for b in range(sorted_logits.size(0)):
                    mask[b, :cutoff_index[b]+1] = True
                probs = torch.zeros_like(next_logits)
                probs.scatter_(1, sorted_indices, F.softmax(sorted_logits, dim=-1) * mask.float())
            else:
                probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return out

    @torch.no_grad()
    def generate_effort(self, input_ids, effort='short', reason_budget=None, temperature=1.0, top_k=0, top_p=0.0, eos_token_id=None):
        """
        Two-phase decoding: generate reasoning tokens inside a <scratch> block up to reason_budget, then generate final answer after <final>.
        effort in {'none','short','medium','long'} maps to default budgets if reason_budget is None.
        This is a simple, synchronous implementation; production should use batched, streaming decodes.
        """
        budget_map = {'none': 0, 'short': 64, 'medium': 256, 'long': 1024}
        if reason_budget is None:
            reason_budget = budget_map.get(effort, 64)

        device = input_ids.device
        model = self
        # phase 1: generate reasoning tokens if budget > 0
        out = input_ids
        if reason_budget > 0:
            for _ in range(reason_budget):
                logits = model.forward(input_ids=out).logits
                next_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                out = torch.cat([out, next_token], dim=1)
        # phase 2: generate final answer until eos or short fixed length
        final_out = out
        for _ in range(128):
            logits = model.forward(input_ids=final_out).logits
            next_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            final_out = torch.cat([final_out, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return final_out

    # Utilities to play nice with train.py expectations
    def save_pretrained(self, out_dir: str, use_safetensors: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        # save state and a small config
        model_path = os.path.join(out_dir, 'pytorch_model.bin')
        cfg = {
            'vocab_size': self.vocab_size,
            'n_positions': self.n_positions,
            'n_embd': self.n_embd,
            'n_layer': len(self.blocks),
            'n_head': self.blocks[0].attn.n_head if len(self.blocks) else 0,
        }
        with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f)

        if use_safetensors:
            try:
                from safetensors.torch import save_file as safe_save
                state = {k: v.cpu() for k, v in self.state_dict().items()}
                safe_save(state, os.path.join(out_dir, 'pytorch_model.safetensors'))
                return
            except Exception:
                # fallback to torch.save if safetensors isn't available
                pass

        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, in_dir: str, map_location=None):
        with open(os.path.join(in_dir, 'config.json'), 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        model = cls(
            vocab_size=cfg.get('vocab_size', 32000),
            n_positions=cfg.get('n_positions', 1024),
            n_embd=cfg.get('n_embd', 768),
            n_layer=cfg.get('n_layer', 12),
            n_head=cfg.get('n_head', 12),
        )
        # Prefer safetensors if present
        safetensors_path = os.path.join(in_dir, 'pytorch_model.safetensors')
        bin_path = os.path.join(in_dir, 'pytorch_model.bin')
        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file as safe_load
                state = safe_load(safetensors_path, device=map_location or 'cpu')
            except Exception:
                state = torch.load(safetensors_path, map_location=map_location)
        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location=map_location)
        else:
            raise FileNotFoundError(f'No model file found in {in_dir}')

        # state is a mapping of tensors
        model.load_state_dict(state)
        return model

    def resize_token_embeddings(self, new_vocab_size: int):
        old_wte = self.wte
        old_vocab, emb_dim = old_wte.weight.shape
        if new_vocab_size == old_vocab:
            return
        new_wte = nn.Embedding(new_vocab_size, emb_dim)
        # copy existing weights
        with torch.no_grad():
            new_wte.weight[:old_vocab] = old_wte.weight
        self.wte = new_wte

        new_head = nn.Linear(emb_dim, new_vocab_size, bias=False)
        with torch.no_grad():
            new_head.weight[:,:old_vocab] = self.head.weight
        self.head = new_head


def build_from_config(config):
    # Build Hanuman from a GPT2Config-like object with mini-model defaults
    return Hanuman(
        vocab_size=getattr(config, 'vocab_size', 32000),
        n_positions=getattr(config, 'n_positions', getattr(config, 'n_ctx', 4096)),
        n_embd=getattr(config, 'n_embd', 512),
        n_layer=getattr(config, 'n_layer', 8),
        n_head=getattr(config, 'n_head', 8),
        mlp_ratio=getattr(config, 'mlp_ratio', 1.0),
        use_rmsnorm=getattr(config, 'use_rmsnorm', True),
        use_moe=getattr(config, 'use_moe', False),
        moe_experts=getattr(config, 'moe_experts', 4),
        moe_top_k=getattr(config, 'moe_top_k', 1),
    gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
    use_think_head=getattr(config, 'use_think_head', False),
    think_aux_coef=getattr(config, 'think_aux_coef', 1.0),
    )
