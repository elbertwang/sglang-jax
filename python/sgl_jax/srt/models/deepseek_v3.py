"""DeepSeek V3 / K2-Thinking model implementation for sglang-jax.

Uses Multi-head Latent Attention (MLA) with non-absorbed KV projection
(following ant-pretrain/MaxText pattern):
- Q: LoRA compression -> RMSNorm -> expansion -> split(nope, rope) -> RoPE
- KV: LoRA compression -> split(latent, rope) -> RMSNorm(latent) -> wkv_b expansion
      -> split(k_nope, v) + broadcast(k_rope) -> k = concat(k_nope, k_rope)
- Standard attention with fully expanded K/V (N heads each)

Architecture:
- Layers 0..first_k_dense_replace-1: Dense FFN
- Layers first_k_dense_replace..num_layers-1: MoE (routed + shared experts)
- All layers use MLA attention
"""

import logging
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import PretrainedConfig

_MOE_DEBUG = os.environ.get("SGLANG_MOE_DEBUG", "0") == "1"

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import (
    Embed, ParallelLMHead, RotaryEmbedding,
    _yarn_find_correction_range,
)
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class DeepseekV3MLP(nnx.Module):
    """Dense FFN with gate/up/down projections (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        return self.down_proj(jax.nn.silu(gate) * up)[0]


class DeepseekYaRNRotaryEmbedding(RotaryEmbedding):
    """YaRN-scaled Rotary Embedding for DeepSeek V3 / K2."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        scaling_factor: float = 40.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        original_max_position_embeddings: int = 4096,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
    ):
        # Don't call super().__init__ yet — we need to override _inv_freq_np
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.scaling_factor = scaling_factor

        # Standard inv_freq
        inv_freq = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))

        # YaRN correction: find the frequency range to interpolate
        # PyTorch reference passes (beta_fast, beta_slow) as (low_rot, high_rot)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, original_max_position_embeddings
        )

        # Create interpolation ramp
        inv_freq_indices = np.arange(rotary_dim // 2, dtype=np.float32)
        if high == low:
            ramp = np.ones_like(inv_freq_indices)
        else:
            ramp = np.clip((inv_freq_indices - low) / (high - low), 0.0, 1.0)

        # Interpolate between standard and scaled frequencies
        inv_freq_scaled = inv_freq / scaling_factor
        inv_freq_yarn = inv_freq * (1.0 - ramp) + inv_freq_scaled * ramp

        self._inv_freq_np = inv_freq_yarn


class DeepseekV3MLAAttention(nnx.Module):
    """Multi-head Latent Attention (MLA) for DeepSeek V3.

    Non-absorbed approach (following ant-pretrain/MaxText pattern):
    - Q path: x -> q_a_proj -> RMSNorm -> q_b_proj -> split(q_nope, q_rope)
              -> apply RoPE to q_rope -> concat -> scale
    - KV path: x -> kv_a_proj -> split(latent, rope_key)
               -> RMSNorm(latent) -> kv_b_proj(latent) -> split(k_nope, v)
               -> apply RoPE to rope_key -> broadcast to N heads
               -> k = concat(k_nope, k_rope)
    - Standard attention with expanded K/V (N heads each)
    - Output: o_proj

    KV cache stores full expanded K [T, N, qk_head_dim] and V [T, N, v_head_dim].
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Scaling factor with YaRN mscale
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        scaling_factor = rope_scaling.get("factor", 1.0)
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)
        if scaling_factor <= 1.0:
            yarn_mscale = 1.0
        else:
            yarn_mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
        self.scaling = self.qk_head_dim ** -0.5 * yarn_mscale ** 2

        # Q path: x -> q_a_proj [D, q_lora_rank] -> RMSNorm -> q_b_proj [q_lora_rank, N*qk_head_dim]
        self.q_a_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.q_lora_rank,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
        )
        self.q_b_proj = LinearBase(
            input_size=self.q_lora_rank,
            output_size=self.num_heads * self.qk_head_dim,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        # KV path: x -> kv_a_proj [D, kv_lora_rank + qk_rope_head_dim]
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_lora_rank + self.qk_rope_head_dim,
            kernel_axes=(None, None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
        )
        # kv_b_proj: [kv_lora_rank, N * (qk_nope_head_dim + v_head_dim)]
        # Expands latent to full K_nope and V per head
        self.kv_b_proj = LinearBase(
            input_size=self.kv_lora_rank,
            output_size=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        # Output projection: [N * v_head_dim, D]
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.v_head_dim,
            output_size=self.hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )

        # RoPE for the rope part of Q and K
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 163840)
        rope_scaling_cfg = getattr(config, "rope_scaling", None) or {}
        rope_type = rope_scaling_cfg.get("type", rope_scaling_cfg.get("rope_type", ""))

        if rope_type == "yarn":
            self.rotary_emb = DeepseekYaRNRotaryEmbedding(
                head_size=self.qk_rope_head_dim,
                rotary_dim=self.qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                is_neox_style=True,
                dtype=dtype,
                scaling_factor=rope_scaling_cfg.get("factor", 40.0),
                beta_fast=rope_scaling_cfg.get("beta_fast", 32.0),
                beta_slow=rope_scaling_cfg.get("beta_slow", 1.0),
                original_max_position_embeddings=rope_scaling_cfg.get(
                    "original_max_position_embeddings", 4096),
                mscale=rope_scaling_cfg.get("mscale", 1.0),
                mscale_all_dim=rope_scaling_cfg.get("mscale_all_dim", 1.0),
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                head_size=self.qk_rope_head_dim,
                rotary_dim=self.qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                is_neox_style=True,
                dtype=dtype,
            )

        # RadixAttention with head_dim aligned to 128 for TPU Pallas kernels
        # Q/K have qk_head_dim=192, V has v_head_dim=128
        # All padded to 256 (next 128-aligned) in __call__
        self.padded_head_dim = ((self.qk_head_dim + 127) // 128) * 128  # 256
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.padded_head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_heads,
            v_head_dim=self.padded_head_dim,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        # === Q path ===
        q_compressed, _ = self.q_a_proj(hidden_states)  # [T, q_lora_rank]
        q_compressed = self.q_a_layernorm(q_compressed)
        q_full, _ = self.q_b_proj(q_compressed)  # [T, N * qk_head_dim]
        q = q_full.reshape(-1, self.num_heads, self.qk_head_dim)  # [T, N, qk_head_dim]

        # Split Q into non-positional and rotary parts
        q_nope = q[..., :self.qk_nope_head_dim]  # [T, N, nope_dim]
        q_rope = q[..., self.qk_nope_head_dim:]  # [T, N, rope_dim]

        # === KV path ===
        kv_combined, _ = self.kv_a_proj_with_mqa(hidden_states)  # [T, kv_lora_rank + rope_dim]
        kv_latent = kv_combined[..., :self.kv_lora_rank]  # [T, kv_lora_rank]
        k_rope_raw = kv_combined[..., self.kv_lora_rank:]  # [T, rope_dim]

        kv_latent = self.kv_a_layernorm(kv_latent)

        # Apply RoPE to q_rope and k_rope
        k_rope_1h = k_rope_raw[:, None, :]  # [T, 1, rope_dim]
        q_rope, k_rope_1h = self.rotary_emb(positions, q_rope, k_rope_1h)

        # Expand latent to full K_nope and V via kv_b_proj
        kv_expanded, _ = self.kv_b_proj(kv_latent)  # [T, N * (nope + v_head)]
        kv_expanded = kv_expanded.reshape(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )  # [T, N, nope + v_head]

        k_nope = kv_expanded[..., :self.qk_nope_head_dim]  # [T, N, nope_dim]
        v = kv_expanded[..., self.qk_nope_head_dim:]  # [T, N, v_head_dim]

        # Broadcast k_rope from 1 head to N heads, then concat with k_nope
        k_rope = jnp.broadcast_to(
            k_rope_1h, (k_rope_1h.shape[0], self.num_heads, self.qk_rope_head_dim)
        )  # [T, N, rope_dim]
        # Reshard k_rope/q_rope to match k_nope/q_nope sharding P(None, 'tensor', None)
        from jax.sharding import NamedSharding, PartitionSpec as P
        head_sharding = NamedSharding(
            self.q_a_proj.mesh, P(None, "tensor", None)
        )
        k_rope = jax.sharding.reshard(k_rope, head_sharding)
        k = jnp.concatenate([k_nope, k_rope], axis=-1)  # [T, N, qk_head_dim]

        # Reconstruct full Q
        q_rope = jax.sharding.reshard(q_rope, head_sharding)
        q = jnp.concatenate([q_nope, q_rope], axis=-1)  # [T, N, qk_head_dim]

        # === Attention ===
        # Pad Q/K/V to padded_head_dim (256, 128-aligned) for TPU Pallas kernels
        pad_qk = self.padded_head_dim - self.qk_head_dim  # 256 - 192 = 64
        pad_v = self.padded_head_dim - self.v_head_dim     # 256 - 128 = 128
        if pad_qk > 0:
            q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_qk)))
            k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_qk)))
        if pad_v > 0:
            v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_v)))

        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        # FlashAttention returns 2D: [T, N * padded_head_dim] (already flattened)
        # We need [T, N * v_head_dim] for o_proj.
        # Reshape to 3D, slice padded head_dim back to v_head_dim, then flatten.
        attn_3d = attn_output.reshape(-1, self.num_heads, self.padded_head_dim)
        attn_3d = attn_3d[..., :self.v_head_dim]  # [T, N, v_head_dim]

        # === Output projection ===
        from jax.sharding import NamedSharding, PartitionSpec as P
        out_reshape_sharding = NamedSharding(self.q_a_proj.mesh, P(None, "tensor"))
        attn_flat = jax.lax.reshape(
            attn_3d,
            (attn_3d.shape[0], self.num_heads * self.v_head_dim),
            out_sharding=out_reshape_sharding,
        )
        output, _ = self.o_proj(attn_flat)

        return output, kv_fused


class DeepseekV3DecoderLayer(nnx.Module):
    """Decoder layer with MLA attention and MoE/Dense FFN."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3MLAAttention(
            config=config,
            mesh=mesh,
            layer_id=layer_id,
            dtype=dtype,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 3)
        self.is_moe = layer_id >= first_k_dense_replace

        if self.is_moe:
            num_experts = getattr(config, "n_routed_experts", 256)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
            moe_intermediate_size = getattr(config, "moe_intermediate_size", 2048)
            n_group = getattr(config, "n_group", 8)
            topk_group = getattr(config, "topk_group", 4)
            routed_scaling_factor = getattr(config, "routed_scaling_factor", 2.5)
            norm_topk_prob = getattr(config, "norm_topk_prob", True)

            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=num_experts,
                weight_dtype=dtype,
                score_func="sigmoid",
                enable_expert_bias=True,
            )
            self.topk = TopK(
                topk=num_experts_per_tok,
                renormalize=norm_topk_prob,
                num_expert_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                layer_id=layer_id,
            )

            moe_backend = getattr(config, "moe_backend", "epmoe")
            self.use_fused = moe_backend == "fused"

            if self.use_fused:
                from sgl_jax.srt.layers.fused_moe import FusedEPMoE
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    intermediate_dim=moe_intermediate_size,
                    mesh=mesh,
                    ep_size=config.ep_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=norm_topk_prob,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    intermediate_dim=moe_intermediate_size,
                    mesh=mesh,
                    ep_size=config.ep_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )

            # Shared experts
            num_shared_experts = getattr(config, "n_shared_experts", 1)
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=num_shared_experts * moe_intermediate_size,
                mesh=mesh,
                layer_id=layer_id,
                dtype=dtype,
            )
        else:
            # Dense FFN for first few layers
            intermediate_size = getattr(config, "intermediate_size", 18432)
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                mesh=mesh,
                layer_id=layer_id,
                dtype=dtype,
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            # Shared experts
            shared_output = self.shared_experts(hidden_states)

            # Routed experts
            router_logits = self.moe_gate(hidden_states)
            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits, correction_bias=correction_bias, dispatch_info=dispatch_info
            )

            if self.use_fused:
                token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)

            routed_output = self.mlp(hidden_states, topk_weights, topk_ids)

            if _MOE_DEBUG and self.layer_id < 3:
                # Can't use jax.debug.callback on multi-host TPU
                # Instead, save debug tensors as model attributes for post-JIT inspection
                pass

            hidden_states = routed_output + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = jnp.zeros((hidden_states.shape[0], 1), dtype=jnp.int32)

        return hidden_states, residual, kv_fused, topk_ids


class DeepseekV3Model(nnx.Module):
    """DeepSeek V3 model backbone."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                DeepseekV3DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        inputs_embeds: jax.Array | None = None,
    ):
        residual = None
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(forward_batch.input_ids)
        layers_kv_fused = []
        layers_topk_ids = []

        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

            if _MOE_DEBUG and (i < 5 or i % 20 == 0):
                _layer_i = i
                _is_moe = layer.is_moe
                def _log_fwd(hn, rn):
                    logger.info("FWD_DEBUG layer=%d hs_norm=%.6f res_norm=%.6f is_moe=%s",
                                _layer_i, float(hn), float(rn), _is_moe)
                jax.debug.callback(_log_fwd,
                    jnp.mean(jnp.abs(hidden_states)),
                    jnp.mean(jnp.abs(residual)) if residual is not None else jnp.float32(0.0))

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_topk_ids


class DeepseekV3ForCausalLM(nnx.Module):
    """DeepSeek V3 for causal language modeling."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype

        # Note: quantization_config is NOT disabled.
        # If --quantization-config-path is provided, loader will apply quantization
        # after weight loading (dynamic per-channel FP8 re-quant).

        # Override head_dim for MLA: FlashAttention and KV cache use this.
        # MLA qk_head_dim=192 padded to 256 (128-aligned for TPU Pallas kernels).
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 192
        padded = ((qk_head_dim + 127) // 128) * 128  # 256
        config.head_dim = padded

        self.model = DeepseekV3Model(config, dtype=self.dtype, mesh=mesh)

        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("DeepseekV3 weights loaded successfully!")

        if _MOE_DEBUG:
            self._debug_weight_stats()

    def _debug_weight_stats(self):
        """Print weight statistics for debugging MoE issues."""
        first_k = getattr(self.config, "first_k_dense_replace", 1)
        # Check a few MoE layers
        for li in [first_k, first_k + 1, 30, 60]:
            if li >= len(self.model.layers):
                continue
            layer = self.model.layers[li]
            if not layer.is_moe:
                continue
            # Gate weights
            gate_k = layer.moe_gate.kernel.value
            gate_stats = f"gate: mean={float(jnp.mean(gate_k)):.4f} std={float(jnp.std(gate_k)):.4f} max={float(jnp.max(jnp.abs(gate_k))):.4f}"
            # Bias
            if layer.moe_gate.bias is not None:
                bias_v = layer.moe_gate.bias.value
                bias_stats = f"bias: mean={float(jnp.mean(bias_v)):.4f} max={float(jnp.max(jnp.abs(bias_v))):.4f}"
            else:
                bias_stats = "bias: None"
            # Expert weights (check first shard)
            mlp = layer.mlp
            wi0 = mlp.wi_0.value
            wi1 = mlp.wi_1.value
            wo = mlp.wo.value
            wi0_stats = f"wi0: dtype={wi0.dtype} mean={float(jnp.mean(jnp.abs(wi0))):.6f} max={float(jnp.max(jnp.abs(wi0))):.4f}"
            wi1_stats = f"wi1: dtype={wi1.dtype} mean={float(jnp.mean(jnp.abs(wi1))):.6f} max={float(jnp.max(jnp.abs(wi1))):.4f}"
            wo_stats = f"wo: dtype={wo.dtype} mean={float(jnp.mean(jnp.abs(wo))):.6f} max={float(jnp.max(jnp.abs(wo))):.4f}"
            # Shared experts
            sh = layer.shared_experts
            sh_gate = sh.gate_proj.weight.value
            sh_stats = f"shared_gate: mean={float(jnp.mean(jnp.abs(sh_gate))):.6f} max={float(jnp.max(jnp.abs(sh_gate))):.4f}"
            logger.info(
                "MOE_DEBUG layer=%d %s %s %s %s %s %s",
                li, gate_stats, bias_stats, wi0_stats, wi1_stats, wo_stats, sh_stats,
            )

    def _create_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        for layer_idx in range(self.config.num_hidden_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _is_quantized(self):
        """Check if model has been quantized (LinearBase -> QuantizedLinear)."""
        return hasattr(self, '_quantized') and self._quantized

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        # We disabled FP8 quantization in __init__, so model uses LinearBase with .weight
        # FP8 weights from safetensors are auto-cast to BF16 by WeightLoader
        quantized = False

        def _proj_mapping(src_prefix, tgt_prefix, sharding, transpose=True):
            """Create weight + scale mappings for a projection layer.

            sharding is specified in LinearBase convention: (input_axis, output_axis).
            For QuantizedLinear.weight_q [output, input], we reverse it.
            HF stores [output, input], same as weight_q, so no transpose needed.
            """
            if quantized:
                # weight_q is [output, input], HF is [output, input] → no transpose
                # Reverse sharding: LinearBase (input, output) → QuantizedLinear (output, input)
                q_sharding = (sharding[1], sharding[0]) if len(sharding) == 2 else sharding
                m = {
                    f"{src_prefix}.weight": WeightMapping(
                        target_path=f"{tgt_prefix}.weight_q",
                        sharding=q_sharding,
                        transpose=False,
                    ),
                    f"{src_prefix}.weight_scale_inv": WeightMapping(
                        target_path=f"{tgt_prefix}.weight_scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                }
            else:
                # LinearBase.weight is [input, output], HF is [output, input] → transpose
                m = {
                    f"{src_prefix}.weight": WeightMapping(
                        target_path=f"{tgt_prefix}.weight",
                        sharding=sharding,
                        transpose=transpose,
                    ),
                }
            return m

        mappings = {
            # Layer norms (not quantized)
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            # MLA layernorms (not quantized)
            f"{prefix}.self_attn.q_a_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.kv_a_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        # MLA attention projections (quantized)
        for proj_name, sharding in [
            ("q_a_proj", (None, None)),
            ("q_b_proj", (None, "tensor")),
            ("kv_a_proj_with_mqa", (None, None)),
            ("kv_b_proj", (None, "tensor")),
            ("o_proj", ("tensor", None)),
        ]:
            mappings.update(_proj_mapping(
                f"{prefix}.self_attn.{proj_name}",
                f"{target_prefix}.self_attn.{proj_name}",
                sharding,
            ))

        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 3)

        if layer_idx < first_k_dense_replace:
            # Dense FFN layers (quantized)
            for proj_name, sharding in [("gate_proj", (None, "tensor")), ("up_proj", (None, "tensor"))]:
                mappings.update(_proj_mapping(
                    f"{prefix}.mlp.{proj_name}",
                    f"{target_prefix}.mlp.{proj_name}",
                    sharding,
                ))
            mappings.update(_proj_mapping(
                f"{prefix}.mlp.down_proj",
                f"{target_prefix}.mlp.down_proj",
                ("tensor", None),
            ))
        else:
            # MoE layers
            # Router gate (not quantized - it's a GateLogit, not LinearBase)
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            # Expert score correction bias
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias",
                sharding=(None,),
                transpose=False,
            )

            # Shared experts (quantized)
            for proj_name, sharding in [("gate_proj", (None, "tensor")), ("up_proj", (None, "tensor"))]:
                mappings.update(_proj_mapping(
                    f"{prefix}.mlp.shared_experts.{proj_name}",
                    f"{target_prefix}.shared_experts.{proj_name}",
                    sharding,
                ))
            mappings.update(_proj_mapping(
                f"{prefix}.mlp.shared_experts.down_proj",
                f"{target_prefix}.shared_experts.down_proj",
                ("tensor", None),
            ))

            # Routed experts (stacked via create_moe_weights_mapping)
            num_experts = getattr(self.config, "n_routed_experts", 256)
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
            metadata = get_global_expert_location_metadata()
            phy_to_log = None
            if metadata is not None:
                physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
                phy_to_log = physical_to_logical_map[layer_idx]

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_experts,
                moe_backend=moe_backend,
                moe_path="mlp",
                source_expert_pattern="experts.{i}",
                physical_to_logical_map=phy_to_log,
            )
            mappings.update(moe_mappings)

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True, layers_topk_ids


EntryClass = DeepseekV3ForCausalLM
