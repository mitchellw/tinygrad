from pathlib import Path
import math, json, argparse, random, time
import tiktoken
from tinygrad.nn.state import safe_load, load_state_dict, get_parameters
from tinygrad import Tensor, dtypes, nn, Device, GlobalCounters, TinyJit, Variable
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, tqdm
from extra.bench_log import BenchEvent, WallTimeEvent
from dataclasses import dataclass
from typing import Union, Optional

# hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
# DEFAULT_FLOAT=bfloat16 PYTHONPATH=. CUDA=1 python3 examples/gpt_oss.py --model=./gpt-oss-20b/original/model.safetensors --benchmark

# References
# https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b
# https://huggingface.co/openai/gpt-oss-20b/tree/main/original
# https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py

# TODO: Get from files
config = { "num_hidden_layers": 24, "num_experts": 32, "experts_per_token": 4, "vocab_size": 201088, "hidden_size": 2880, "intermediate_size": 2880, "swiglu_limit": 7.0, "head_dim": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "sliding_window": 128, "initial_context_length": 4096, "rope_theta": 150000, "rope_scaling_factor": 32.0, "rope_ntk_alpha": 1, "rope_ntk_beta": 32, }
# data_types = {"embedding.weight": "BF16", "block.0.attn.norm.scale": "BF16", "block.0.attn.qkv.weight": "BF16", "block.0.attn.qkv.bias": "BF16", "block.0.attn.sinks": "BF16", "block.0.attn.out.weight": "BF16", "block.0.attn.out.bias": "BF16", "block.0.mlp.norm.scale": "BF16", "block.0.mlp.gate.weight": "BF16", "block.0.mlp.gate.bias": "BF16", "block.0.mlp.mlp1_weight.blocks": "FP4", "block.0.mlp.mlp1_weight.scales": "UE8", "block.0.mlp.mlp1_bias": "BF16", "block.0.mlp.mlp2_weight.blocks": "FP4", "block.0.mlp.mlp2_weight.scales": "UE8", "block.0.mlp.mlp2_bias": "BF16", "block.1.attn.norm.scale": "BF16", "block.1.attn.qkv.weight": "BF16", "block.1.attn.qkv.bias": "BF16", "block.1.attn.sinks": "BF16", "block.1.attn.out.weight": "BF16", "block.1.attn.out.bias": "BF16", "block.1.mlp.norm.scale": "BF16", "block.1.mlp.gate.weight": "BF16", "block.1.mlp.gate.bias": "BF16", "block.1.mlp.mlp1_weight.blocks": "FP4", "block.1.mlp.mlp1_weight.scales": "UE8", "block.1.mlp.mlp1_bias": "BF16", "block.1.mlp.mlp2_weight.blocks": "FP4", "block.1.mlp.mlp2_weight.scales": "UE8", "block.1.mlp.mlp2_bias": "BF16", "block.2.attn.norm.scale": "BF16", "block.2.attn.qkv.weight": "BF16", "block.2.attn.qkv.bias": "BF16", "block.2.attn.sinks": "BF16", "block.2.attn.out.weight": "BF16", "block.2.attn.out.bias": "BF16", "block.2.mlp.norm.scale": "BF16", "block.2.mlp.gate.weight": "BF16", "block.2.mlp.gate.bias": "BF16", "block.2.mlp.mlp1_weight.blocks": "FP4", "block.2.mlp.mlp1_weight.scales": "UE8", "block.2.mlp.mlp1_bias": "BF16", "block.2.mlp.mlp2_weight.blocks": "FP4", "block.2.mlp.mlp2_weight.scales": "UE8", "block.2.mlp.mlp2_bias": "BF16", "block.3.attn.norm.scale": "BF16", "block.3.attn.qkv.weight": "BF16", "block.3.attn.qkv.bias": "BF16", "block.3.attn.sinks": "BF16", "block.3.attn.out.weight": "BF16", "block.3.attn.out.bias": "BF16", "block.3.mlp.norm.scale": "BF16", "block.3.mlp.gate.weight": "BF16", "block.3.mlp.gate.bias": "BF16", "block.3.mlp.mlp1_weight.blocks": "FP4", "block.3.mlp.mlp1_weight.scales": "UE8", "block.3.mlp.mlp1_bias": "BF16", "block.3.mlp.mlp2_weight.blocks": "FP4", "block.3.mlp.mlp2_weight.scales": "UE8", "block.3.mlp.mlp2_bias": "BF16", "block.4.attn.norm.scale": "BF16", "block.4.attn.qkv.weight": "BF16", "block.4.attn.qkv.bias": "BF16", "block.4.attn.sinks": "BF16", "block.4.attn.out.weight": "BF16", "block.4.attn.out.bias": "BF16", "block.4.mlp.norm.scale": "BF16", "block.4.mlp.gate.weight": "BF16", "block.4.mlp.gate.bias": "BF16", "block.4.mlp.mlp1_weight.blocks": "FP4", "block.4.mlp.mlp1_weight.scales": "UE8", "block.4.mlp.mlp1_bias": "BF16", "block.4.mlp.mlp2_weight.blocks": "FP4", "block.4.mlp.mlp2_weight.scales": "UE8", "block.4.mlp.mlp2_bias": "BF16", "block.5.attn.norm.scale": "BF16", "block.5.attn.qkv.weight": "BF16", "block.5.attn.qkv.bias": "BF16", "block.5.attn.sinks": "BF16", "block.5.attn.out.weight": "BF16", "block.5.attn.out.bias": "BF16", "block.5.mlp.norm.scale": "BF16", "block.5.mlp.gate.weight": "BF16", "block.5.mlp.gate.bias": "BF16", "block.5.mlp.mlp1_weight.blocks": "FP4", "block.5.mlp.mlp1_weight.scales": "UE8", "block.5.mlp.mlp1_bias": "BF16", "block.5.mlp.mlp2_weight.blocks": "FP4", "block.5.mlp.mlp2_weight.scales": "UE8", "block.5.mlp.mlp2_bias": "BF16", "block.6.attn.norm.scale": "BF16", "block.6.attn.qkv.weight": "BF16", "block.6.attn.qkv.bias": "BF16", "block.6.attn.sinks": "BF16", "block.6.attn.out.weight": "BF16", "block.6.attn.out.bias": "BF16", "block.6.mlp.norm.scale": "BF16", "block.6.mlp.gate.weight": "BF16", "block.6.mlp.gate.bias": "BF16", "block.6.mlp.mlp1_weight.blocks": "FP4", "block.6.mlp.mlp1_weight.scales": "UE8", "block.6.mlp.mlp1_bias": "BF16", "block.6.mlp.mlp2_weight.blocks": "FP4", "block.6.mlp.mlp2_weight.scales": "UE8", "block.6.mlp.mlp2_bias": "BF16", "block.7.attn.norm.scale": "BF16", "block.7.attn.qkv.weight": "BF16", "block.7.attn.qkv.bias": "BF16", "block.7.attn.sinks": "BF16", "block.7.attn.out.weight": "BF16", "block.7.attn.out.bias": "BF16", "block.7.mlp.norm.scale": "BF16", "block.7.mlp.gate.weight": "BF16", "block.7.mlp.gate.bias": "BF16", "block.7.mlp.mlp1_weight.blocks": "FP4", "block.7.mlp.mlp1_weight.scales": "UE8", "block.7.mlp.mlp1_bias": "BF16", "block.7.mlp.mlp2_weight.blocks": "FP4", "block.7.mlp.mlp2_weight.scales": "UE8", "block.7.mlp.mlp2_bias": "BF16", "block.8.attn.norm.scale": "BF16", "block.8.attn.qkv.weight": "BF16", "block.8.attn.qkv.bias": "BF16", "block.8.attn.sinks": "BF16", "block.8.attn.out.weight": "BF16", "block.8.attn.out.bias": "BF16", "block.8.mlp.norm.scale": "BF16", "block.8.mlp.gate.weight": "BF16", "block.8.mlp.gate.bias": "BF16", "block.8.mlp.mlp1_weight.blocks": "FP4", "block.8.mlp.mlp1_weight.scales": "UE8", "block.8.mlp.mlp1_bias": "BF16", "block.8.mlp.mlp2_weight.blocks": "FP4", "block.8.mlp.mlp2_weight.scales": "UE8", "block.8.mlp.mlp2_bias": "BF16", "block.9.attn.norm.scale": "BF16", "block.9.attn.qkv.weight": "BF16", "block.9.attn.qkv.bias": "BF16", "block.9.attn.sinks": "BF16", "block.9.attn.out.weight": "BF16", "block.9.attn.out.bias": "BF16", "block.9.mlp.norm.scale": "BF16", "block.9.mlp.gate.weight": "BF16", "block.9.mlp.gate.bias": "BF16", "block.9.mlp.mlp1_weight.blocks": "FP4", "block.9.mlp.mlp1_weight.scales": "UE8", "block.9.mlp.mlp1_bias": "BF16", "block.9.mlp.mlp2_weight.blocks": "FP4", "block.9.mlp.mlp2_weight.scales": "UE8", "block.9.mlp.mlp2_bias": "BF16", "block.10.attn.norm.scale": "BF16", "block.10.attn.qkv.weight": "BF16", "block.10.attn.qkv.bias": "BF16", "block.10.attn.sinks": "BF16", "block.10.attn.out.weight": "BF16", "block.10.attn.out.bias": "BF16", "block.10.mlp.norm.scale": "BF16", "block.10.mlp.gate.weight": "BF16", "block.10.mlp.gate.bias": "BF16", "block.10.mlp.mlp1_weight.blocks": "FP4", "block.10.mlp.mlp1_weight.scales": "UE8", "block.10.mlp.mlp1_bias": "BF16", "block.10.mlp.mlp2_weight.blocks": "FP4", "block.10.mlp.mlp2_weight.scales": "UE8", "block.10.mlp.mlp2_bias": "BF16", "block.11.attn.norm.scale": "BF16", "block.11.attn.qkv.weight": "BF16", "block.11.attn.qkv.bias": "BF16", "block.11.attn.sinks": "BF16", "block.11.attn.out.weight": "BF16", "block.11.attn.out.bias": "BF16", "block.11.mlp.norm.scale": "BF16", "block.11.mlp.gate.weight": "BF16", "block.11.mlp.gate.bias": "BF16", "block.11.mlp.mlp1_weight.blocks": "FP4", "block.11.mlp.mlp1_weight.scales": "UE8", "block.11.mlp.mlp1_bias": "BF16", "block.11.mlp.mlp2_weight.blocks": "FP4", "block.11.mlp.mlp2_weight.scales": "UE8", "block.11.mlp.mlp2_bias": "BF16", "block.12.attn.norm.scale": "BF16", "block.12.attn.qkv.weight": "BF16", "block.12.attn.qkv.bias": "BF16", "block.12.attn.sinks": "BF16", "block.12.attn.out.weight": "BF16", "block.12.attn.out.bias": "BF16", "block.12.mlp.norm.scale": "BF16", "block.12.mlp.gate.weight": "BF16", "block.12.mlp.gate.bias": "BF16", "block.12.mlp.mlp1_weight.blocks": "FP4", "block.12.mlp.mlp1_weight.scales": "UE8", "block.12.mlp.mlp1_bias": "BF16", "block.12.mlp.mlp2_weight.blocks": "FP4", "block.12.mlp.mlp2_weight.scales": "UE8", "block.12.mlp.mlp2_bias": "BF16", "block.13.attn.norm.scale": "BF16", "block.13.attn.qkv.weight": "BF16", "block.13.attn.qkv.bias": "BF16", "block.13.attn.sinks": "BF16", "block.13.attn.out.weight": "BF16", "block.13.attn.out.bias": "BF16", "block.13.mlp.norm.scale": "BF16", "block.13.mlp.gate.weight": "BF16", "block.13.mlp.gate.bias": "BF16", "block.13.mlp.mlp1_weight.blocks": "FP4", "block.13.mlp.mlp1_weight.scales": "UE8", "block.13.mlp.mlp1_bias": "BF16", "block.13.mlp.mlp2_weight.blocks": "FP4", "block.13.mlp.mlp2_weight.scales": "UE8", "block.13.mlp.mlp2_bias": "BF16", "block.14.attn.norm.scale": "BF16", "block.14.attn.qkv.weight": "BF16", "block.14.attn.qkv.bias": "BF16", "block.14.attn.sinks": "BF16", "block.14.attn.out.weight": "BF16", "block.14.attn.out.bias": "BF16", "block.14.mlp.norm.scale": "BF16", "block.14.mlp.gate.weight": "BF16", "block.14.mlp.gate.bias": "BF16", "block.14.mlp.mlp1_weight.blocks": "FP4", "block.14.mlp.mlp1_weight.scales": "UE8", "block.14.mlp.mlp1_bias": "BF16", "block.14.mlp.mlp2_weight.blocks": "FP4", "block.14.mlp.mlp2_weight.scales": "UE8", "block.14.mlp.mlp2_bias": "BF16", "block.15.attn.norm.scale": "BF16", "block.15.attn.qkv.weight": "BF16", "block.15.attn.qkv.bias": "BF16", "block.15.attn.sinks": "BF16", "block.15.attn.out.weight": "BF16", "block.15.attn.out.bias": "BF16", "block.15.mlp.norm.scale": "BF16", "block.15.mlp.gate.weight": "BF16", "block.15.mlp.gate.bias": "BF16", "block.15.mlp.mlp1_weight.blocks": "FP4", "block.15.mlp.mlp1_weight.scales": "UE8", "block.15.mlp.mlp1_bias": "BF16", "block.15.mlp.mlp2_weight.blocks": "FP4", "block.15.mlp.mlp2_weight.scales": "UE8", "block.15.mlp.mlp2_bias": "BF16", "block.16.attn.norm.scale": "BF16", "block.16.attn.qkv.weight": "BF16", "block.16.attn.qkv.bias": "BF16", "block.16.attn.sinks": "BF16", "block.16.attn.out.weight": "BF16", "block.16.attn.out.bias": "BF16", "block.16.mlp.norm.scale": "BF16", "block.16.mlp.gate.weight": "BF16", "block.16.mlp.gate.bias": "BF16", "block.16.mlp.mlp1_weight.blocks": "FP4", "block.16.mlp.mlp1_weight.scales": "UE8", "block.16.mlp.mlp1_bias": "BF16", "block.16.mlp.mlp2_weight.blocks": "FP4", "block.16.mlp.mlp2_weight.scales": "UE8", "block.16.mlp.mlp2_bias": "BF16", "block.17.attn.norm.scale": "BF16", "block.17.attn.qkv.weight": "BF16", "block.17.attn.qkv.bias": "BF16", "block.17.attn.sinks": "BF16", "block.17.attn.out.weight": "BF16", "block.17.attn.out.bias": "BF16", "block.17.mlp.norm.scale": "BF16", "block.17.mlp.gate.weight": "BF16", "block.17.mlp.gate.bias": "BF16", "block.17.mlp.mlp1_weight.blocks": "FP4", "block.17.mlp.mlp1_weight.scales": "UE8", "block.17.mlp.mlp1_bias": "BF16", "block.17.mlp.mlp2_weight.blocks": "FP4", "block.17.mlp.mlp2_weight.scales": "UE8", "block.17.mlp.mlp2_bias": "BF16", "block.18.attn.norm.scale": "BF16", "block.18.attn.qkv.weight": "BF16", "block.18.attn.qkv.bias": "BF16", "block.18.attn.sinks": "BF16", "block.18.attn.out.weight": "BF16", "block.18.attn.out.bias": "BF16", "block.18.mlp.norm.scale": "BF16", "block.18.mlp.gate.weight": "BF16", "block.18.mlp.gate.bias": "BF16", "block.18.mlp.mlp1_weight.blocks": "FP4", "block.18.mlp.mlp1_weight.scales": "UE8", "block.18.mlp.mlp1_bias": "BF16", "block.18.mlp.mlp2_weight.blocks": "FP4", "block.18.mlp.mlp2_weight.scales": "UE8", "block.18.mlp.mlp2_bias": "BF16", "block.19.attn.norm.scale": "BF16", "block.19.attn.qkv.weight": "BF16", "block.19.attn.qkv.bias": "BF16", "block.19.attn.sinks": "BF16", "block.19.attn.out.weight": "BF16", "block.19.attn.out.bias": "BF16", "block.19.mlp.norm.scale": "BF16", "block.19.mlp.gate.weight": "BF16", "block.19.mlp.gate.bias": "BF16", "block.19.mlp.mlp1_weight.blocks": "FP4", "block.19.mlp.mlp1_weight.scales": "UE8", "block.19.mlp.mlp1_bias": "BF16", "block.19.mlp.mlp2_weight.blocks": "FP4", "block.19.mlp.mlp2_weight.scales": "UE8", "block.19.mlp.mlp2_bias": "BF16", "block.20.attn.norm.scale": "BF16", "block.20.attn.qkv.weight": "BF16", "block.20.attn.qkv.bias": "BF16", "block.20.attn.sinks": "BF16", "block.20.attn.out.weight": "BF16", "block.20.attn.out.bias": "BF16", "block.20.mlp.norm.scale": "BF16", "block.20.mlp.gate.weight": "BF16", "block.20.mlp.gate.bias": "BF16", "block.20.mlp.mlp1_weight.blocks": "FP4", "block.20.mlp.mlp1_weight.scales": "UE8", "block.20.mlp.mlp1_bias": "BF16", "block.20.mlp.mlp2_weight.blocks": "FP4", "block.20.mlp.mlp2_weight.scales": "UE8", "block.20.mlp.mlp2_bias": "BF16", "block.21.attn.norm.scale": "BF16", "block.21.attn.qkv.weight": "BF16", "block.21.attn.qkv.bias": "BF16", "block.21.attn.sinks": "BF16", "block.21.attn.out.weight": "BF16", "block.21.attn.out.bias": "BF16", "block.21.mlp.norm.scale": "BF16", "block.21.mlp.gate.weight": "BF16", "block.21.mlp.gate.bias": "BF16", "block.21.mlp.mlp1_weight.blocks": "FP4", "block.21.mlp.mlp1_weight.scales": "UE8", "block.21.mlp.mlp1_bias": "BF16", "block.21.mlp.mlp2_weight.blocks": "FP4", "block.21.mlp.mlp2_weight.scales": "UE8", "block.21.mlp.mlp2_bias": "BF16", "block.22.attn.norm.scale": "BF16", "block.22.attn.qkv.weight": "BF16", "block.22.attn.qkv.bias": "BF16", "block.22.attn.sinks": "BF16", "block.22.attn.out.weight": "BF16", "block.22.attn.out.bias": "BF16", "block.22.mlp.norm.scale": "BF16", "block.22.mlp.gate.weight": "BF16", "block.22.mlp.gate.bias": "BF16", "block.22.mlp.mlp1_weight.blocks": "FP4", "block.22.mlp.mlp1_weight.scales": "UE8", "block.22.mlp.mlp1_bias": "BF16", "block.22.mlp.mlp2_weight.blocks": "FP4", "block.22.mlp.mlp2_weight.scales": "UE8", "block.22.mlp.mlp2_bias": "BF16", "block.23.attn.norm.scale": "BF16", "block.23.attn.qkv.weight": "BF16", "block.23.attn.qkv.bias": "BF16", "block.23.attn.sinks": "BF16", "block.23.attn.out.weight": "BF16", "block.23.attn.out.bias": "BF16", "block.23.mlp.norm.scale": "BF16", "block.23.mlp.gate.weight": "BF16", "block.23.mlp.gate.bias": "BF16", "block.23.mlp.mlp1_weight.blocks": "FP4", "block.23.mlp.mlp1_weight.scales": "UE8", "block.23.mlp.mlp1_bias": "BF16", "block.23.mlp.mlp2_weight.blocks": "FP4", "block.23.mlp.mlp2_weight.scales": "UE8", "block.23.mlp.mlp2_bias": "BF16", "norm.scale": "BF16", "unembedding.weight": "BF16"}

# Bytes per MXFP4 block: 32 FP4 numbers packed in 16 bytes, with a scale value associated with them
BYTES_PER_BLOCK = 16

FP4_VALUES = [ +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, ]


@dataclass
class ModelConfig:
  num_hidden_layers: int = 36
  num_experts: int = 128
  experts_per_token: int = 4
  vocab_size: int = 201088
  hidden_size: int = 2880
  intermediate_size: int = 2880
  swiglu_limit: float = 7.0
  head_dim: int = 64
  num_attention_heads: int = 64
  num_key_value_heads: int = 8
  sliding_window: int = 128
  initial_context_length: int = 4096
  rope_theta: float = 150000.0
  rope_scaling_factor: float = 32.0
  rope_ntk_alpha: float = 1.0
  rope_ntk_beta: float = 32.0


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
  cos = cos.unsqueeze(-2).cast(x.dtype)
  sin = sin.unsqueeze(-2).cast(x.dtype)
  [x1, x2] = x.chunk(2, dim=-1)
  o1 = x1 * cos - x2 * sin
  o2 = x2 * cos + x1 * sin
  return o1.cat(o2, dim=-1)


class RotaryEmbedding:
  def __init__(
    self,
    head_dim: int,
    base: int,
    initial_context_length: int = 4096,
    scaling_factor: float = 1.0,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
    device=Device.DEFAULT,
  ) -> None:
    self.head_dim = head_dim
    self.base = base
    self.initial_context_length = initial_context_length
    self.scaling_factor = scaling_factor
    self.ntk_alpha = ntk_alpha
    self.ntk_beta = ntk_beta
    self.device = device

  def _compute_concentration_and_inv_freq(self) -> Tensor:
    """See YaRN paper: https://arxiv.org/abs/2309.00071"""
    freq = self.base ** (Tensor.arange(0, self.head_dim, 2, dtype=dtypes.float32, device=self.device) / self.head_dim)
    if self.scaling_factor > 1.0:
      concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

      d_half = self.head_dim / 2
      # NTK by parts
      low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
      high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
      assert 0 < low < high < d_half - 1

      interpolation = 1.0 / (self.scaling_factor * freq)
      extrapolation = 1.0 / freq

      ramp = (Tensor.arange(d_half, dtype=dtypes.float32, device=freq.device) - low) / (high - low)
      mask = 1 - ramp.clamp(min_=0, max_=1)

      inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
      concentration = 1.0
      inv_freq = 1.0 / freq

    return concentration, inv_freq

  def _compute_cos_sin(self, num_tokens: int) -> tuple[Tensor, Tensor]:
    concentration, inv_freq = self._compute_concentration_and_inv_freq()
    t = Tensor.arange(num_tokens, dtype=dtypes.float32, device=self.device)
    freqs = Tensor.einsum("i,j->ij", t, inv_freq, math_dtype=dtypes.float32)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin

  def __call__(
    self,
    query: Tensor,
    key: Tensor,
    pre_computed_cos: Tensor,
    pre_computed_sin: Tensor,
    start_pos: int,
  ) -> tuple[Tensor, Tensor]:
    num_tokens = query.shape[0]
    cos, sin = pre_computed_cos[start_pos : start_pos + num_tokens, :], pre_computed_sin[start_pos : start_pos + num_tokens, :]

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_dim)
    query = _apply_rotary_emb(query, cos, sin)
    query = query.reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_dim)
    key = _apply_rotary_emb(key, cos, sin)
    key = key.reshape(key_shape)
    return query, key


# TODO: Replace with scaled_dot_product_attention from Tensor?
def sdpa(Q: Tensor, K: Tensor, V: Tensor, S: Tensor, sm_scale: float, mask: Tensor):
  n_tokens, n_heads, q_mult, _d_head = Q.shape
  K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
  V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
  S = S.reshape((n_heads, q_mult, 1, 1)).expand(-1, -1, n_tokens, -1)
  QK = Tensor.einsum("qhmd,khmd->hmqk", Q, K, math_dtype=dtypes.float32)
  QK *= sm_scale
  QK += mask[None, None, :, :]
  QK = QK.cat(S, dim=-1)
  W = QK.softmax(axis=-1, dtype=dtypes.float32).cast(QK.dtype)
  # Can't just use a slice because of the variable
  W = W.shrink(tuple((0, x - y) for x, y in zip(W.shape, (0, 0, 0, 1))))
  attn = Tensor.einsum("hmqk,khmd->qhmd", W, V, math_dtype=dtypes.float32)
  return attn.reshape(n_tokens, -1)


class AttentionBlock:
  def __init__(
    self,
    config: ModelConfig,
    rope: RotaryEmbedding,
    pre_computed_cos: Tensor,
    pre_computed_sin: Tensor,
    device=Device.DEFAULT,
    max_context=16384,
  ):
    self.max_context = max_context
    self.head_dim = config.head_dim
    self.num_attention_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.sinks = Tensor.empty(config.num_attention_heads, device=device, dtype=dtypes.bfloat16)
    self.norm = nn.RMSNorm(config.hidden_size, eps=1e-05, dtype=dtypes.float32)
    qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
    self.qkv = nn.Linear(config.hidden_size, qkv_dim, dtype=dtypes.bfloat16, math_dtype=dtypes.float32)
    self.out = nn.Linear(config.head_dim * config.num_attention_heads, config.hidden_size, dtype=dtypes.bfloat16, math_dtype=dtypes.float32)
    self.sm_scale = 1 / math.sqrt(config.head_dim)
    self.rope = rope
    self.pre_computed_cos, self.pre_computed_sin = pre_computed_cos, pre_computed_sin

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], mask: Optional[Tensor] = None) -> Tensor:
    t = self.norm(x)
    qkv = self.qkv(t)
    q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
    k = qkv[
      :,
      self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
    ].contiguous()
    v = qkv[
      :,
      (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
      * self.head_dim,
    ].contiguous()

    q = q.view(
      -1,
      self.num_key_value_heads,
      self.num_attention_heads // self.num_key_value_heads,
      self.head_dim,
    )
    k = k.view(-1, self.num_key_value_heads, self.head_dim)
    v = v.view(-1, self.num_key_value_heads, self.head_dim)

    q, k = self.rope(q, k, self.pre_computed_cos, self.pre_computed_sin, start_pos)
    seqlen = x.shape[0]

    if self.max_context:
      if not hasattr(self, "cache_kv"):
        self.cache_kv = Tensor.zeros(2, self.max_context, self.num_key_value_heads, self.head_dim, dtype=k.dtype).contiguous().realize()

      # update the cache
      assert k.dtype == v.dtype == self.cache_kv.dtype, f"{k.dtype=}, {v.dtype=}, {self.cache_kv.dtype=}"
      self.cache_kv[:, start_pos : start_pos + seqlen, :, :].assign(Tensor.stack(k, v)).realize()

      k = self.cache_kv[0, : start_pos + seqlen, :, :]
      v = self.cache_kv[1, : start_pos + seqlen, :, :]
    else:
      assert start_pos == 0

    # TODO: Replace with scaled_dot_product_attention from Tensor?
    # t = q.scaled_dot_product_attention(k, v, self.sinks, self.sm_scale, self.sliding_window)
    t = sdpa(q, k, v, self.sinks, self.sm_scale, mask=mask)
    t = self.out(t)
    t = x + t
    return t


# TODO: Move this to Tensor?
def swiglu(x: Tensor, alpha: float = 1.702, limit: float = 7.0):
  x_glu, x_linear = x[..., ::2], x[..., 1::2]
  # Clamp the input values
  x_glu = x_glu.clamp(min_=None, max_=limit)
  x_linear = x_linear.clamp(min_=-limit, max_=limit)
  out_glu = x_glu * x_glu.mul(alpha).sigmoid(math_dtype=dtypes.float32)
  # Note we add an extra bias of 1 to the linear layer
  return out_glu * (x_linear + 1)


class MLPBlock:
  def __init__(
    self,
    config: ModelConfig,
    device=Device.DEFAULT,
  ):
    self.num_experts = config.num_experts
    self.experts_per_token = config.experts_per_token
    self.swiglu_limit = config.swiglu_limit
    self.world_size = 1  # TODO: for multigpu
    self.norm = nn.RMSNorm(config.hidden_size, eps=1e-05, dtype=dtypes.float32)
    self.gate = nn.Linear(config.hidden_size, config.num_experts, dtype=dtypes.bfloat16, math_dtype=dtypes.float32)
    assert config.intermediate_size % self.world_size == 0
    # TODO: Split up experts so we can load each per token when vram constrained
    # TODO: Use fp4 properly, for now we'll just upcast to bfloat16 from fp4 as reference impl does
    self.mlp1_weight = Tensor.empty(
      (
        config.num_experts,
        config.intermediate_size * 2 // self.world_size,
        config.hidden_size,
      ),
      device=device,
      dtype=dtypes.bfloat16,
    )
    self.mlp1_bias = Tensor.empty(
      (config.num_experts, config.intermediate_size * 2 // self.world_size),
      device=device,
      dtype=dtypes.bfloat16,
    )
    # TODO: Use fp4 properly, for now we'll just upcast to bfloat16 from fp4 as reference impl does
    self.mlp2_weight = Tensor.empty(
      (
        config.num_experts,
        config.hidden_size,
        config.intermediate_size // self.world_size,
      ),
      device=device,
      dtype=dtypes.bfloat16,
    )
    self.mlp2_bias = Tensor.empty(
      (config.num_experts, config.hidden_size),
      device=device,
      dtype=dtypes.bfloat16,
    )

  def __call__(self, x: Tensor) -> Tensor:
    t = self.norm(x)
    g = self.gate(t)
    experts, expert_indices = g.topk(self.experts_per_token, dim=-1)
    expert_weights = experts.softmax(axis=1, dtype=dtypes.float32).cast(dtypes.float16)

    # MLP #1
    mlp1_weight = self.mlp1_weight[expert_indices, ...]
    mlp1_bias = self.mlp1_bias[expert_indices, ...]
    t = Tensor.einsum("beck,bk->bec", mlp1_weight, t, math_dtype=dtypes.float32) + mlp1_bias
    t = swiglu(t, limit=self.swiglu_limit)

    # MLP #2
    mlp2_weight = self.mlp2_weight[expert_indices, ...]
    mlp2_bias = self.mlp2_bias[expert_indices, ...]
    t = Tensor.einsum("beck,bek->bec", mlp2_weight, t, math_dtype=dtypes.float32)
    # TODO: For multi gpu, need to properly use shard() and reduce here
    # if self.world_size > 1:
    #    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t += mlp2_bias

    # Weighted sum of experts
    # TODO: einsum in general is returning differently than torch even using float32
    # This seems to be the biggest or only source of error compared to reference impl
    t = Tensor.einsum("bec,be->bc", t, expert_weights, math_dtype=dtypes.float32)

    return x + t


class TransformerBlock:
  def __init__(
    self,
    config: ModelConfig,
    rope: RotaryEmbedding,
    pre_computed_cos: Tensor,
    pre_computed_sin: Tensor,
    device=Device.DEFAULT,
    max_context=16384,
  ):
    self.attn = AttentionBlock(config, rope, pre_computed_cos=pre_computed_cos, pre_computed_sin=pre_computed_sin, max_context=max_context, device=device)
    self.mlp = MLPBlock(config, device=device)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], mask: Optional[Tensor] = None) -> Tensor:
    x = self.attn(x, start_pos, mask)
    x = self.mlp(x)
    return x


class Transformer:
  def __init__(
    self,
    config: ModelConfig,
    device=Device.DEFAULT,
    max_context=16384,
  ):
    rope = RotaryEmbedding(config.head_dim, config.rope_theta, initial_context_length=config.initial_context_length, scaling_factor=config.rope_scaling_factor, ntk_alpha=config.rope_ntk_alpha, ntk_beta=config.rope_ntk_beta, device=device)
    pre_computed_cos, pre_computed_sin = rope._compute_cos_sin(max_context)
    self.max_context = max_context
    self.sliding_window = config.sliding_window
    # TODO: This seems to always go to DEFAULT_FLOAT despite passing in dtype
    self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtypes.bfloat16)
    self.block = [TransformerBlock(config, rope, pre_computed_cos, pre_computed_sin, device=device, max_context=max_context) for _ in range(config.num_hidden_layers)]
    self.norm = nn.RMSNorm(config.hidden_size, eps=1e-05, dtype=dtypes.float32)
    self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtypes.bfloat16, math_dtype=dtypes.float32)
    self.full_mask = Tensor.full((self.max_context, self.max_context), float("-inf"), dtype=dtypes.bfloat16).triu(1)
    self.full_sliding_mask = self.full_mask + Tensor.full((self.max_context, self.max_context), float("-inf"), dtype=self.full_mask.dtype).tril(-self.sliding_window)

  @TinyJit
  def forward(self, x: Tensor, start_pos: Union[Variable, int], TEMPERATURE=0.0) -> Tensor:
    seqlen = x.shape[0]
    x = self.embedding(x)
    mask = self.full_mask[start_pos : start_pos + seqlen, : start_pos + seqlen]
    sliding_mask = self.full_sliding_mask[start_pos : start_pos + seqlen, : start_pos + seqlen]
    for layer_idx, block in enumerate(self.block):
      # Only apply sliding window to every other layer
      x = block(x, start_pos, sliding_mask if layer_idx % 2 == 0 else mask)
    x = self.norm(x)
    logits = self.unembedding(x)[-1]

    return logits.argmax(axis=-1) if TEMPERATURE == 0.0 else (logits * (1.0 / TEMPERATURE)).softmax(axis=-1).multinomial(num_samples=1)

  def prefill(self, x: Tensor, start_pos: int, TEMPERATURE=0.0) -> Tensor:
    return self.forward(x, Variable("start_pos", 0, self.max_context - x.shape[0]).bind(start_pos), TEMPERATURE)

  def __call__(self, x: Tensor, start_pos: int, TEMPERATURE=0.0) -> Tensor:
    # TODO: Support when start_pos is 0 and also x is empty tensor
    assert x.shape[0] == 1, f"You can only call this model with 1 token, but you passed {x.shape[0]}. If you'd like to prefill many tokens at once, call prefill."
    return self.forward(x, Variable("start_pos", 0, self.max_context - 1).bind(start_pos), TEMPERATURE)

  @staticmethod
  def from_checkpoint(model_path: str, model_size: str, device=Device.DEFAULT, max_context=16384) -> "Transformer":
    model_config = ModelConfig() if model_size == "120B" else ModelConfig(**config)

    with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
      # TODO: For loading from folders, index.json, etc.
      """
      if model_path.is_dir():
        if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
        elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
        else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
      else:
        weights = load(str(model_path))
        if "model.embed_tokens.weight" in weights:
          weights = convert_from_huggingface(weights, MODEL_PARAMS[model_size]["args"]["n_layers"], MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
        elif "token_embd.weight" in weights:
          weights = convert_from_gguf(weights, MODEL_PARAMS[model_size]["args"]["n_layers"])
      """
      weights = safe_load(str(model_path))

      # Change norms to use 'weights' instead of 'scale'
      norms_to_rename = []
      for k, v in weights.items():
        if k.endswith("norm.scale"):
          norms_to_rename += [k]
      for k in norms_to_rename:
        weights[k[:-5] + "weight"] = weights.pop(k)

      # TODO: Move this into a more generic function
      lut = Tensor(FP4_VALUES, dtype=dtypes.bfloat16, device=device)
      to_add = []
      to_delete = []
      for k, v in weights.items():
        if v.dtype == dtypes.uchar and k.endswith(".blocks") and k[:-7] + ".scales" in weights:
          blocks = weights[k]
          # TODO: Casting seems like it doesn't work the same on disk tensors as it does on device
          scales = (weights[k[:-7] + ".scales"]).to(device).cast(dtypes.int16) - 127

          *prefix_shape, G, B = blocks.shape
          rows_total = math.prod(prefix_shape) * G

          blocks = blocks.reshape(rows_total, B)
          scales = scales.reshape(rows_total, 1)

          rows_per_chunk = 16384 * 512
          out = Tensor.empty((rows_total, B * 2), dtype=dtypes.bfloat16, device=device)

          for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1].to(device)
            exp = scales[r0:r1].to(device)

            # use nibbles as indices into fp4 lut
            idx_lo = (blk & 0x0F).cast(dtypes.uint8).realize()
            idx_hi = (blk >> 4).cast(dtypes.uint8).realize()

            out[r0:r1, 0::2] = lut[idx_lo]
            out[r0:r1, 1::2] = lut[idx_hi]

            out[r0:r1] = out[r0:r1] * exp.exp2()

          to_add += [(k[:-7], out.reshape(tuple(prefix_shape) + (G * B * 2,)).cast(dtypes.bfloat16))]
          to_delete += [k[:-7] + ".scales"]
          to_delete += [k]
      for x in to_delete:
        del weights[x]
      weights.update(to_add)

    # replace weights in model
    model = Transformer(
      config=model_config,
      device=device,
      max_context=max_context,
    )
    load_state_dict(model, weights, strict=False, consume=True, keep_model_dtypes=True)
    return model


def get_tokenizer():
  o200k_base = tiktoken.get_encoding("o200k_base")
  tokenizer = tiktoken.Encoding(
    name="o200k_harmony",
    pat_str=o200k_base._pat_str,
    mergeable_ranks=o200k_base._mergeable_ranks,
    special_tokens={
      **o200k_base._special_tokens,
      "<|startoftext|>": 199998,
      "<|endoftext|>": 199999,
      "<|reserved_200000|>": 200000,
      "<|reserved_200001|>": 200001,
      "<|return|>": 200002,
      "<|constrain|>": 200003,
      "<|reserved_200004|>": 200004,
      "<|channel|>": 200005,
      "<|start|>": 200006,
      "<|end|>": 200007,
      "<|message|>": 200008,
      "<|reserved_200009|>": 200009,
      "<|reserved_200010|>": 200010,
      "<|reserved_200011|>": 200011,
      "<|call|>": 200012,
    }
    | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
  )
  return tokenizer

last_seen_toks = []

def prefill(model: Transformer, toks: list[int], start_pos=0) -> int:
  global last_seen_toks
  if len(toks) > model.max_context:
    raise ValueError(f"Prompt too long. Must be less than {str(model.max_context)}, but is {str(len(toks))}")

  # we can skip part of the prompt if it is the same as last and start_pos=0
  if start_pos == 0:
    for i, (a, b) in enumerate(zip(toks, last_seen_toks)):
      if a != b:
        break
    else:
      i = min(len(toks), len(last_seen_toks))
    start_pos += i
    last_seen_toks = toks
    toks = toks[i:]

  # prefill the model
  for tok in tqdm(toks):
    GlobalCounters.reset()
    model(Tensor([tok], device=device), start_pos).realize()
    start_pos += 1

  # TODO: Implement an efficient prefill. Maybe at least worth it at startup time to initialize
  """
  model.prefill(Tensor(toks, device=device), start_pos).realize()
  start_pos += len(toks)
  """

  return start_pos


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # TODO: Implement downloading model instead of needing safetensors file locally already
  #parser.add_argument("--download_model", action="store_true", help="Download a model")
  parser.add_argument("--model", type=Path, help="Model path")
  parser.add_argument("--size", choices=["20B", "120B"], default="20B", help="Model size")
  # TODO: Implement sharding
  #parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
  parser.add_argument("--no_api", action="store_true", help="Disable the api and run a cli test interface")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
  parser.add_argument("--port", type=int, default=7776, help="Web server port")
  parser.add_argument("--debug", action="store_true", help="Enable debug mode")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
  # TODO: Implement top_p, top_k, etc.
  #parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
  parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data")
  args = parser.parse_args()

  assert args.model is not None, "please provide --model option"

  if args.seed is not None:
    Tensor.manual_seed(args.seed)
  if args.benchmark:
    Tensor.manual_seed(42)
  print(f"seed = {Tensor._seed}")
  # TODO: Add back in top_p, possibly others, but for now they recommend top_p as 1.0 anyway
  TEMPERATURE: float = args.temperature

  tokenizer = get_tokenizer()
  # The reference implementation only uses <|endoftext|> / tokenizer.eot_token, but that's clearly not always right based on my usage
  # So basing it on this https://github.com/openai/harmony/blob/508cbaa7f6b0277bd37c9bdf6d4dc8a4d51aada5/demo/harmony-demo/src/components/HarmonyDemo.tsx#L79
  stop_tokens = [tokenizer._special_tokens["<|endoftext|>"], tokenizer._special_tokens["<|return|>"], tokenizer._special_tokens["<|call|>"]]


  # TODO: Use openai_harmony?
  def encode_role(role: str, channel: str | None = None):
    return (
      [tokenizer._special_tokens["<|start|>"]]
      + tokenizer.encode(role)
      + (([tokenizer._special_tokens["<|channel|>"]] + tokenizer.encode(channel)) if channel else [])
    )

  # TODO: Use openai_harmony?
  def encode_message(role: str, content: str, channel: str | None = None, allow_special: bool = False):
    return (
      encode_role(role, channel)
      + [tokenizer._special_tokens["<|message|>"]]
      + tokenizer.encode(content.strip(), allowed_special=("all" if allow_special else set()))
      + [tokenizer._special_tokens["<|end|>"]]
    )

  ENCODED_DEFAULT_SYSTEM_PROMPT = tokenizer.encode(
    """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: low

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
""",
    allowed_special="all",
  )

  device = Device.DEFAULT # TODO for multigpu: tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT

  model = Transformer.from_checkpoint(args.model, args.size, device)
  param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(model))

  # TODO: Support tool calls, chat interface, etc.?
  if not args.no_api and not args.benchmark:
    from bottle import Bottle, request, response, HTTPResponse, abort, static_file

    app = Bottle()

    cors_headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
      "Access-Control-Allow-Credentials": "true",
    }

    @app.hook("before_request")
    def handle_options():
      if request.method == "OPTIONS":
        raise HTTPResponse(headers=cors_headers)

    @app.hook("after_request")
    def enable_cors():
      for key, value in cors_headers.items():
        response.set_header(key, value)

    @app.route("/<filename>")
    def server_static(filename):
      return static_file(filename, root=(Path(__file__).parent / "tinychat").as_posix())

    @app.route("/assets/<filename:path>")
    def server_assets(filename):
      return static_file(filename, root=(Path(__file__).parent / "tinychat" / "assets").as_posix())

    @app.route("/")
    def index():
      return static_file("index.html", root=(Path(__file__).parent / "tinychat").as_posix())

    @app.get("/v1/models")
    def models():
      return json.dumps([str(args.model)])

    @app.post("/v1/internal/token-count")
    def token_count():
      rjson = json.loads(request.body.read())
      return json.dumps(len(tokenizer.encode(rjson.get("text", ""), allowed_special="all")))

    @app.post("/v1/token/encode")
    def token_encode():
      rjson = json.loads(request.body.read())
      return json.dumps(tokenizer.encode(rjson.get("text", ""), allowed_special="all"))

    @app.post("/v1/completions")
    def completions():
      global last_seen_toks
      rjson = json.loads(request.body.read())

      # check if we are streaming
      if rjson.get("stream", False):
        response.content_type = "text/event-stream"
        response.set_header("Cache-Control", "no-cache")
      else:
        abort(400, "streaming required")

      # This is just a local tool, no big deal if they're trying to send special tokens.
      toks = tokenizer.encode(rjson.get("prompt", ""), allowed_special="all")

      start_pos = prefill(model, toks[:-1])
      last_tok = toks[-1]
      last_seen_toks.append(last_tok)
      while True:
        GlobalCounters.reset()
        tok = model(Tensor([last_tok], device=device), start_pos, TEMPERATURE).item()
        start_pos += 1
        last_tok = tok
        last_seen_toks.append(tok)

        res = {
          "choices": [
            {
              "text": tokenizer.decode([tok]),
            }
          ]
        }
        yield f"data: {json.dumps(res)}\n\n"

        # Be more raw on this endpoint and return the stop token to the client, but then end generation
        if tok in stop_tokens:
          break

    @app.post("/v1/chat/token/encode")
    def chat_token_encode():
      rjson = json.loads(request.body.read())
      if "messages" not in rjson:
        abort(400, "messages required")
      toks = []
      for message in rjson["messages"]:
        role = message["role"]
        if role == "assistant":
          toks += tokenizer.encode(message["content"], allowed_special="all")
        else: # This is just a local tool, no big deal if they're trying to send special tokens.
          toks += encode_message(role, message["content"], allow_special=True)
      return json.dumps(toks)

    @app.post("/v1/chat/completions")
    def chat_completions():
      global last_seen_toks
      rjson = json.loads(request.body.read())
      if "messages" not in rjson:
        abort(400, "messages required")

      # check if we are streaming
      if rjson.get("stream", False):
        response.content_type = "text/event-stream"
        response.set_header("Cache-Control", "no-cache")
      else:
        abort(400, "streaming required")

      toks = [] if rjson["messages"][0]["role"] == "system" else ENCODED_DEFAULT_SYSTEM_PROMPT.copy()
      for message in rjson["messages"]:
        role = message["role"]
        # TODO: Since we are using the harmony format and the way the frontend works,
        # the format isn't fully implemented. But we'd need to do some more work to allow for
        # different channels, display them, etc. So just take assistant messages as-is for now
        if role == "assistant":
          toks += tokenizer.encode(message["content"], allowed_special="all")
        else: # This is just a local tool, no big deal if they're trying to send special tokens.
          toks += encode_message(role, message["content"], allow_special=True)
      # ensure that the last message was a user message
      if role != "user":
        abort(400, "last message must be a user message")

      random_id = random.randbytes(16).hex()
      # This helps prompt the model to follow its format well. We'll send it in the first message back to the client to keep things simple
      assistant_start = encode_role("assistant")
      toks += assistant_start

      start_pos = prefill(model, toks[:-1])
      last_tok = toks[-1]
      last_seen_toks.append(last_tok)

      has_sent_assistant_start = False
      should_stop = False
      while not should_stop:
        GlobalCounters.reset()
        tok = model(Tensor([last_tok], device=device), start_pos, TEMPERATURE).item()
        if tok in stop_tokens:
          should_stop = True
          # Replace the stop token with an end token to make things easier while not using harmony
          tok = tokenizer._special_tokens["<|end|>"]
        else:
          start_pos += 1
          last_tok = tok
          last_seen_toks.append(tok)

        # TODO: Some unknown characters (emojis from context?), could also be breaking kv cache
        res = {
          "id": random_id,
          "object": "chat.completion.chunk",
          "created": int(time.time()),
          "model": str(args.model),
          "choices": [
            {
              "index": 0,
              "delta": {
                "role": "assistant",
                "content": tokenizer.decode(([] if has_sent_assistant_start else assistant_start) + [tok]),
              },
              "finish_reason": None,
            }
          ],
        }
        yield f"data: {json.dumps(res)}\n\n"
        has_sent_assistant_start = True

      res = {
        "id": random_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": str(args.model),
        "choices": [
          {
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
          }
        ],
      }
      yield f"data: {json.dumps(res)}\n\n"

    app.run(host=args.host, port=args.port, debug=args.debug)
  elif args.benchmark:
    toks = ENCODED_DEFAULT_SYSTEM_PROMPT.copy() + encode_message("user", "what's the weather like where you are?") + encode_role("assistant") # tokenizer.encode('why?')

    start_pos = prefill(model, toks[:-1])
    last_tok = toks[-1]
    generated = ""
    for _ in range(4096):
      GlobalCounters.reset()
      st = GlobalCounters.time_sum_s
      with Profiling(enabled=args.profile):
        with Timing("total ", on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, param {param_bytes / x:.2f} GB/s"):
          with WallTimeEvent(BenchEvent.STEP):
            with Timing(
              "enqueue in ",
              on_exit=(
                lambda et: (f", {(GlobalCounters.time_sum_s - st) * 1e3:.2f} ms on GPU" if DEBUG >= 2 else "")
                + f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, {GlobalCounters.global_mem * 1e-9:.2f} GB"
                + (f", {GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s, param {param_bytes * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s" if DEBUG >= 2 else "")
              ) if DEBUG else None,
            ):
              # Use 0 temperature for benchmarking and comparing to reference implementation
              predicted_token = model(Tensor([last_tok], device=device), start_pos, 0.0)  
            predicted_token = predicted_token.item()

      start_pos += 1
      last_tok = predicted_token
      generated += tokenizer.decode([last_tok])
      print(generated)
      # Show stop token for benchmarking to confirm expected one is there
      if predicted_token in stop_tokens:
        break

    if "20B" in args.size:
      EXPECTED_TEXT = "<|channel|>analysis<|message|>Need to answer that I have no location.<|end|><|start|>assistant<|channel|>final<|message|>I don’t have a physical location, so I can’t check the weather for a specific place. If you tell me the city or region you’re interested in, I can look up the current forecast for you!<|return|>"
      assert generated == EXPECTED_TEXT, f"{generated=} {EXPECTED_TEXT}"
      print("\n" + colored("output validated", "green"))  # NOTE: "\n" inside colored does not render the color in github action
  else:
    prompt = ENCODED_DEFAULT_SYSTEM_PROMPT.copy()

    start_pos = prefill(model, prompt)
    first_message = True
    while True:
      # This relies on the fact that the kv cache is filled properly during inference, but the issue is that the harmony format expects
      # you to replace stop tokens with <|end|>, so we may need to add that here at the start and not add to start_pos when hitting stop tokens
      # to avoid weird behavior.
      # This is just a local tool, no big deal if they're trying to send special tokens.
      toks = ([] if first_message else [tokenizer._special_tokens["<|end|>"]]) + encode_message("user", input("Q: "), allow_special=True)
      first_message = False
      # This helps prompt the model to follow its format well. We'll send it in the first message back to the client to keep things simple
      assistant_start = encode_role("assistant")
      toks += assistant_start

      start_pos = prefill(model, toks[:-1], start_pos=start_pos)
      last_tok = toks[-1]

      has_sent_assistant_start = False
      should_stop = False
      while not should_stop:
        GlobalCounters.reset()
        if args.timing or args.profile:
          print("")
        st = GlobalCounters.time_sum_s
        with Profiling(enabled=args.profile):
          with Timing(
            "total ",
            enabled=args.timing,
            on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, param {param_bytes / x:.2f} GB/s",
          ):
            with Timing(
              "enqueue in ",
              on_exit=(
                lambda et: (f", {(GlobalCounters.time_sum_s - st) * 1e3:.2f} ms on GPU" if DEBUG >= 2 else "")
                + f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, {GlobalCounters.global_mem * 1e-9:.2f} GB"
                + (
                  f", {GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s, param {param_bytes * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s"
                  if DEBUG >= 2
                  else ""
                )
              )
              if DEBUG
              else None,
              enabled=args.timing,
            ):
              tok = model(Tensor([last_tok], device=device), start_pos, TEMPERATURE)
            tok = tok.item()
        if tok in stop_tokens:
          should_stop = True
          # Replace the stop token with an end token to make things easier while not using harmony, but don't increment start_pos
          tok = tokenizer._special_tokens["<|end|>"]
        else:
          start_pos += 1
          last_tok = tok
        print(tokenizer.decode(([] if has_sent_assistant_start else assistant_start) + [tok]), end="", flush=True)
        has_sent_assistant_start = True
      print(flush=True)
