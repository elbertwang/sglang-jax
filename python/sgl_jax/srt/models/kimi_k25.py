"""Kimi K2.5 multimodal model for sglang-jax.

Architecture: MoonViT-3D vision encoder + DeepSeek V3 text backbone.
Vision encoder processes images into embeddings that replace placeholder tokens
in the text input before feeding into the language model.

Phase A: Uses K2-Thinking text weights + K2.5 vision weights for validation.
Phase B: Full K2.5 weights.
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Model
from sgl_jax.srt.models.moonvit3d import MoonViT3dConfig, MoonViT3dModel, PatchMergerMLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def merge_multimodal_embeddings(
    input_ids: jax.Array,
    inputs_embeds: jax.Array,
    multimodal_embeddings: list[jax.Array],
    placeholder_token_id: int,
) -> jax.Array:
    """Replace placeholder tokens with vision embeddings.

    Args:
        input_ids: [T] token ids
        inputs_embeds: [T, D] text embeddings
        multimodal_embeddings: list of [Ni, D] vision embeddings per image
        placeholder_token_id: token id to replace (163605)

    Returns:
        [T, D] merged embeddings
    """
    if not multimodal_embeddings:
        return inputs_embeds

    # Concatenate all image embeddings
    all_vision = jnp.concatenate(multimodal_embeddings, axis=0)  # [total_vision_tokens, D]

    # Find placeholder positions
    mask = (input_ids == placeholder_token_id)  # [T]

    # Build index: for each placeholder token, which vision token to use
    vision_idx = jnp.cumsum(mask) - 1  # [T], -1 for non-placeholder

    # Replace: where mask is True, use vision embedding; otherwise keep text
    merged = jnp.where(
        mask[:, None],
        all_vision[vision_idx],
        inputs_embeds,
    )
    return merged


class KimiK25ForConditionalGeneration(nnx.Module):
    """Kimi K2.5 = MoonViT-3D + DeepSeek V3."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype

        # Disable quantization (FP8 weights auto-cast to BF16 by WeightLoader)
        if hasattr(config, "quantization_config"):
            config.quantization_config = None

        # Get text config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            # Also clear text_config quantization (K2.5 has INT4 config from original)
            if hasattr(text_config, "quantization_config"):
                text_config.quantization_config = None
        else:
            text_config = config

        # Set head_dim for MLA (128-aligned for TPU Pallas)
        qk_head_dim = getattr(text_config, "qk_nope_head_dim", 128) + getattr(text_config, "qk_rope_head_dim", 64)
        padded_head_dim = ((qk_head_dim + 127) // 128) * 128
        text_config.head_dim = padded_head_dim
        config.head_dim = padded_head_dim

        # Vision encoder — use underscore prefix to exclude from nnx.state()
        # This prevents vision tower weights from bloating the JIT-ed forward pass
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            vit_config = MoonViT3dConfig.from_dict(
                vars(vision_config) if hasattr(vision_config, "__dict__") else vision_config
            )
            self._vision_tower = MoonViT3dModel(vit_config, dtype=dtype)
            self._mm_projector = PatchMergerMLP(vit_config, dtype=dtype)
        else:
            self._vision_tower = None
            self._mm_projector = None

        # Text model (DeepSeek V3 backbone)
        self.model = DeepseekV3Model(text_config, dtype=dtype, mesh=mesh)

        # LM head
        vocab_size = getattr(text_config, "vocab_size", config.vocab_size)
        hidden_size = text_config.hidden_size
        if not getattr(text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                vocab_size, hidden_size,
                dtype=dtype, param_dtype=dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )

        self.logits_processor = LogitsProcessor(vocab_size, mesh=mesh)

        # Placeholder token for image embeddings
        self.media_placeholder_token_id = getattr(config, "media_placeholder_token_id", 163605)

    def embed_multimodal(
        self,
        pixel_values: jax.Array,
        grid_thws: list[tuple[int, int, int]],
    ) -> list[jax.Array]:
        """Process images through vision encoder + projector.

        Args:
            pixel_values: [total_patches, 3, patch_size, patch_size]
            grid_thws: [(t, h, w), ...] per image

        Returns:
            List of [Ni, text_hidden_size] embeddings per image
        """
        if self._vision_tower is None:
            return []

        image_embeds = []
        current_idx = 0
        for t, h, w in grid_thws:
            image_size = t * h * w
            img_pixels = pixel_values[current_idx:current_idx + image_size]

            # MoonViT: returns list of (N, kh*kw, hidden) per image
            vision_out = self._vision_tower(img_pixels, [(t, h, w)])
            # Projector: returns list of (N, text_hidden) per image
            proj_out = self._mm_projector(vision_out)
            image_embeds.append(proj_out[0])  # (N, 7168)
            current_idx += image_size

        return image_embeds

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        # Text-only path: identical to DeepseekV3ForCausalLM.__call__
        # No embed_tokens outside, no getattr, no conditional branches inside JIT
        # Multimodal merging should happen BEFORE this call (in scheduler/tokenizer layer)
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )

        # Logits
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, True, layers_topk_ids

    def _dequantize_fp8_experts(self, model_config: ModelConfig):
        """Dequantize FP8 block-wise expert weights using weight_scale_inv.

        FP8 weights need: w_bf16 = w_fp8.astype(bf16) * scale_inv (per 128x128 block)
        Without this, weights are ~5000x too small → garbled output.

        Only applies to MoE expert weights (attention weights are already BF16).
        Expert weights are stacked in model as (num_experts_per_device, out, in).
        Scale_inv is per-expert (out_blocks, in_blocks).
        """
        import os, json
        from safetensors import safe_open
        import ml_dtypes

        model_path = model_config.model_path
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})

        scale_keys = [k for k in weight_map if k.endswith(".weight_scale_inv") and "expert" in k]
        if not scale_keys:
            logger.info("No expert scale_inv found, skipping FP8 dequant")
            return

        logger.info(f"Found {len(scale_keys)} FP8 expert scale_inv entries. Dequantizing stacked MoE weights...")

        # The MoE weights in model are stacked: (num_experts_per_device, out, in)
        # We need to read each expert's scale_inv from safetensors and apply to
        # the corresponding slice of the stacked weight.

        # Group scales by layer and weight type
        from collections import defaultdict
        # key: (layer_idx, weight_type) → list of (expert_idx, scale_inv_key, shard_file)
        layer_experts = defaultdict(list)
        for sk in scale_keys:
            # Parse: language_model.model.layers.X.mlp.experts.E.gate_proj.weight_scale_inv
            parts = sk.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
                expert_idx = int(parts[parts.index("experts") + 1])
                # weight type: gate_proj, up_proj, or down_proj
                wtype_idx = parts.index("experts") + 2
                wtype = parts[wtype_idx]  # gate_proj, up_proj, down_proj
                layer_experts[(layer_idx, wtype)].append((expert_idx, sk, weight_map[sk]))
            except (ValueError, IndexError):
                continue

        # Get model state
        from flax import nnx
        state = nnx.state(self)

        # Map weight type to EPMoE param name
        wtype_to_param = {"gate_proj": "wi_0", "up_proj": "wi_1", "down_proj": "wo"}

        dequant_count = 0
        block_size = 128

        for (layer_idx, wtype), experts in layer_experts.items():
            param_name = wtype_to_param.get(wtype)
            if not param_name:
                continue

            # Find the stacked weight in model state
            # Path: model.layers.{layer_idx}.mlp.{param_name}
            try:
                layers_state = state["model"]["layers"]
                layer_state = layers_state[layer_idx]
                mlp_state = layer_state["mlp"]
                param = mlp_state[param_name]
                if not hasattr(param, 'value'):
                    continue
            except (KeyError, IndexError, TypeError):
                continue

            w = np.array(jax.device_get(param.value), dtype=np.float32)
            # w shape: (num_experts_per_device, out_dim, in_dim) for stacked MoE

            if w.ndim != 3:
                continue

            num_local_experts, out_dim, in_dim = w.shape

            # Read scale_inv for each expert and apply
            for expert_idx, scale_key, shard_file in sorted(experts):
                fpath = os.path.join(model_path, shard_file)
                if not os.path.exists(fpath):
                    continue

                with safe_open(fpath, framework="numpy") as f:
                    if scale_key not in f.keys():
                        continue
                    scale_inv = f.get_tensor(scale_key)  # (blocks_out, blocks_in)

                # Find which local index this expert maps to
                # With ep_size=8, device gets experts [start..start+num_local-1]
                # expert_idx is global (0..383)
                # local_idx = expert_idx % num_local_experts (simple mapping)
                local_idx = expert_idx % num_local_experts
                if local_idx >= w.shape[0]:
                    continue

                # Apply block-wise dequant
                for bo in range(scale_inv.shape[0]):
                    for bi in range(scale_inv.shape[1]):
                        r_s = bo * block_size
                        r_e = min(r_s + block_size, out_dim)
                        c_s = bi * block_size
                        c_e = min(c_s + block_size, in_dim)
                        w[local_idx, r_s:r_e, c_s:c_e] *= scale_inv[bo, bi]

            param.value = jnp.array(w, dtype=param.value.dtype)
            dequant_count += 1

        nnx.update(self, state)
        logger.info(f"Dequantized {dequant_count} stacked MoE weight groups")

    def load_weights(self, model_config: ModelConfig):
        """Load weights with FP8 dequantization."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)

        # TODO: FP8 expert weights need dequant with weight_scale_inv
        # Currently loading raw FP8→BF16 without scale → garbled output
        # Fix: either use BF16 weights or implement block-wise dequant in loader

        logger.info("KimiK25 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        """Create weight mappings for both text and vision weights."""
        text_config = getattr(self.config, "text_config", self.config)
        mappings = {}

        # --- Text weight mappings (reuse DeepseekV3 logic) ---
        # For K2.5, HF keys have "language_model." prefix that needs stripping.
        # For K2-Thinking (Phase A), keys match directly.
        # WeightLoader handles missing keys gracefully.

        # Embeddings
        for prefix_pair in [
            ("model.embed_tokens.weight", "model.embed_tokens.embedding"),
            ("language_model.model.embed_tokens.weight", "model.embed_tokens.embedding"),
        ]:
            mappings[prefix_pair[0]] = WeightMapping(
                target_path=prefix_pair[1],
                sharding=("tensor", None),
                transpose=False,
            )

        # Norm
        for prefix_pair in [
            ("model.norm.weight", "model.norm.scale"),
            ("language_model.model.norm.weight", "model.norm.scale"),
        ]:
            mappings[prefix_pair[0]] = WeightMapping(
                target_path=prefix_pair[1],
                sharding=(None,),
                transpose=False,
            )

        # LM head
        for prefix_pair in [
            ("lm_head.weight", "lm_head.embedding"),
            ("language_model.lm_head.weight", "lm_head.embedding"),
        ]:
            mappings[prefix_pair[0]] = WeightMapping(
                target_path=prefix_pair[1],
                sharding=("tensor", None),
                transpose=False,
            )

        # Per-layer mappings (both with and without language_model. prefix)
        num_layers = text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            from sgl_jax.srt.models.deepseek_v3 import DeepseekV3ForCausalLM
            # Create a temporary instance to reuse _create_layer_mappings
            # Actually, just inline the logic:
            layer_mappings = self._create_text_layer_mappings(layer_idx, text_config)
            mappings.update(layer_mappings)

        # Vision weights excluded from JIT state (_vision_tower uses underscore prefix).
        # Vision weight loading happens separately, not through WeightLoader.
        # TODO: implement separate vision weight loading for multimodal inference.

        return mappings

    def _create_text_layer_mappings(self, layer_idx: int, text_config) -> dict:
        """Create text layer mappings, supporting both K2-Thinking and K2.5 key formats."""
        mappings = {}
        quantized = False  # We disable quantization

        # Generate mappings for both prefix formats
        for src_base in [f"model.layers.{layer_idx}", f"language_model.model.layers.{layer_idx}"]:
            target_prefix = f"model.layers.{layer_idx}"

            # Layer norms
            mappings[f"{src_base}.input_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,), transpose=False,
            )
            mappings[f"{src_base}.post_attention_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,), transpose=False,
            )

            # MLA layernorms
            mappings[f"{src_base}.self_attn.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_layernorm.scale",
                sharding=(None,), transpose=False,
            )
            mappings[f"{src_base}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
                sharding=(None,), transpose=False,
            )

            # MLA projections (LinearBase: [input, output], HF: [output, input] → transpose)
            for proj_name, sharding in [
                ("q_a_proj", (None, None)),
                ("q_b_proj", (None, "tensor")),
                ("kv_a_proj_with_mqa", (None, None)),
                ("kv_b_proj", (None, "tensor")),
                ("o_proj", ("tensor", None)),
            ]:
                mappings[f"{src_base}.self_attn.{proj_name}.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.self_attn.{proj_name}.weight",
                    sharding=sharding, transpose=True,
                )

            # Dense FFN or MoE
            first_k = getattr(text_config, "first_k_dense_replace", 3)
            if layer_idx < first_k:
                for proj_name, sharding in [("gate_proj", (None, "tensor")), ("up_proj", (None, "tensor"))]:
                    mappings[f"{src_base}.mlp.{proj_name}.weight"] = WeightMapping(
                        target_path=f"{target_prefix}.mlp.{proj_name}.weight",
                        sharding=sharding, transpose=True,
                    )
                mappings[f"{src_base}.mlp.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight",
                    sharding=("tensor", None), transpose=True,
                )
            else:
                # MoE router
                mappings[f"{src_base}.mlp.gate.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.moe_gate.kernel",
                    sharding=(None, None), transpose=True,
                )
                # Shared experts
                for proj_name, sharding in [("gate_proj", (None, "tensor")), ("up_proj", (None, "tensor"))]:
                    mappings[f"{src_base}.mlp.shared_experts.{proj_name}.weight"] = WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.{proj_name}.weight",
                        sharding=sharding, transpose=True,
                    )
                mappings[f"{src_base}.mlp.shared_experts.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.weight",
                    sharding=("tensor", None), transpose=True,
                )

                # Routed experts (MoE stacking via create_moe_weights_mapping)
                from sgl_jax.srt.layers.moe import create_moe_weights_mapping
                from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

                num_experts = getattr(text_config, "n_routed_experts", 256)
                moe_backend = getattr(text_config, "moe_backend", "epmoe")
                metadata = get_global_expert_location_metadata()
                phy_to_log = None
                if metadata is not None:
                    physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
                    phy_to_log = physical_to_logical_map[layer_idx]

                # Add MoE mappings for both prefix formats
                if True:  # Generate for each prefix
                    moe_mappings = create_moe_weights_mapping(
                        prefix=src_base,
                        target_prefix=target_prefix,
                        num_experts=num_experts,
                        moe_backend=moe_backend,
                        moe_path="mlp",
                        source_expert_pattern="experts.{i}",
                        physical_to_logical_map=phy_to_log,
                    )
                    mappings.update(moe_mappings)

        return mappings

    def _create_vision_weight_mappings(self) -> dict:
        """Create weight mappings for MoonViT-3D vision encoder."""
        mappings = {}

        # Patch embedding
        mappings["vision_tower.patch_embed.proj.weight"] = WeightMapping(
            target_path="vision_tower.patch_embed.proj.kernel",
            sharding=(None, None, None, None),
            transpose=False,
            transpose_axes=(2, 3, 1, 0),  # Conv2d: (out,in,kH,kW) → (kH,kW,in,out)
        )
        mappings["vision_tower.patch_embed.proj.bias"] = WeightMapping(
            target_path="vision_tower.patch_embed.proj.bias",
            sharding=(None,), transpose=False,
        )
        mappings["vision_tower.patch_embed.pos_emb.weight"] = WeightMapping(
            target_path="vision_tower.patch_embed.pos_emb.weight",
            sharding=(None, None, None), transpose=False,  # (H, W, D) no transpose
        )

        # Encoder blocks (27 layers)
        for i in range(27):
            block_prefix = f"vision_tower.encoder.blocks.{i}"
            for name_pair in [
                ("norm0.weight", "norm0.scale"),
                ("norm0.bias", "norm0.bias"),
                ("norm1.weight", "norm1.scale"),
                ("norm1.bias", "norm1.bias"),
                ("wqkv.weight", "wqkv.kernel"),
                ("wqkv.bias", "wqkv.bias"),
                ("wo.weight", "wo.kernel"),
                ("wo.bias", "wo.bias"),
                ("mlp.fc0.weight", "mlp.fc0.kernel"),
                ("mlp.fc0.bias", "mlp.fc0.bias"),
                ("mlp.fc1.weight", "mlp.fc1.kernel"),
                ("mlp.fc1.bias", "mlp.fc1.bias"),
            ]:
                src_name, tgt_name = name_pair
                is_weight = src_name.endswith(".weight") and "norm" not in src_name
                mappings[f"{block_prefix}.{src_name}"] = WeightMapping(
                    target_path=f"{block_prefix}.{tgt_name}",
                    sharding=(None,) * (2 if is_weight else 1),
                    transpose=is_weight,  # Linear weights need transpose
                )

        # Final layernorm
        mappings["vision_tower.encoder.final_layernorm.weight"] = WeightMapping(
            target_path="vision_tower.encoder.final_layernorm.scale",
            sharding=(None,), transpose=False,
        )
        mappings["vision_tower.encoder.final_layernorm.bias"] = WeightMapping(
            target_path="vision_tower.encoder.final_layernorm.bias",
            sharding=(None,), transpose=False,
        )

        # MM Projector
        for name_pair in [
            ("mm_projector.pre_norm.weight", "mm_projector.pre_norm.scale"),
            ("mm_projector.pre_norm.bias", "mm_projector.pre_norm.bias"),
            ("mm_projector.proj.0.weight", "mm_projector.proj_0.kernel"),
            ("mm_projector.proj.0.bias", "mm_projector.proj_0.bias"),
            ("mm_projector.proj.2.weight", "mm_projector.proj_2.kernel"),
            ("mm_projector.proj.2.bias", "mm_projector.proj_2.bias"),
        ]:
            src_name, tgt_name = name_pair
            is_weight = src_name.endswith(".weight") and "norm" not in src_name
            mappings[src_name] = WeightMapping(
                target_path=tgt_name,
                sharding=(None,) * (2 if is_weight else 1),
                transpose=is_weight,
            )

        return mappings


EntryClass = KimiK25ForConditionalGeneration
