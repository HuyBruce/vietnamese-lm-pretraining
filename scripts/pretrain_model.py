"""Small dense causal LM: pre-RMSNorm, RoPE, MHA, SwiGLU, tied embeddings."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['hidden_size']
        self.heads = config['num_heads']
        self.head_dim = width // self.heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        frequencies = config['rope_theta'] ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        angles = torch.outer(torch.arange(config['context_length']).float(), frequencies)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def rotate(self, x):
        cos = self.cos[:x.shape[-2]].to(x.dtype)
        sin = self.sin[:x.shape[-2]].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        attention = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v,
                                                  dropout_p=0.0, is_causal=True)
        return self.output(attention.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, intermediate = config['hidden_size'], config['intermediate_size']
        self.attention_norm = RMSNorm(width, config['rms_norm_eps'])
        self.ffn_norm = RMSNorm(width, config['rms_norm_eps'])
        self.attention = Attention(config)
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        normalized = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        width, heads = config['hidden_size'], config['num_heads']
        if width % heads or (width // heads) % 2:
            raise ValueError('hidden_size must divide num_heads with even head dimension')
        self.embedding = nn.Embedding(config['vocab_size'], width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config['num_layers']))
        self.norm = RMSNorm(width, config['rms_norm_eps'])
        self.apply(self._initialize)
        for block in self.blocks:
            for layer in (block.attention.output, block.down):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * config['num_layers']))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def hidden(self, input_ids):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config['context_length']:
            raise ValueError('Expected [batch, sequence] within context_length')
        x = self.embedding(input_ids)
        for block in self.blocks:
            if self.training and self.config['gradient_checkpointing']:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, input_ids):
        return F.linear(self.hidden(input_ids), self.embedding.weight)

    def loss_sum(self, input_ids, targets, chunk_size=256):
        """Targets are ALREADY shifted. Recompute chunked head to bound logits memory."""
        if targets.shape != input_ids.shape or chunk_size < 1:
            raise ValueError('Invalid targets shape or loss chunk size')
        hidden = self.hidden(input_ids).reshape(-1, self.config['hidden_size'])
        targets = targets.reshape(-1)

        def chunk_loss(x, y):
            return F.cross_entropy(F.linear(x, self.embedding.weight).float(), y,
                                   ignore_index=-100, reduction='sum')

        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(targets), chunk_size):
            x, y = hidden[start:start + chunk_size], targets[start:start + chunk_size]
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, x, y, use_reentrant=False)
            else:
                total = total + chunk_loss(x, y)
        return total
