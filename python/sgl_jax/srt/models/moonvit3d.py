# Copyright 2025 MoonViT JAX Port
# Ported from moonshotai/Kimi-K2.5 modeling_kimi_k25.py (Apache 2.0 + MIT)
#
# MoonViT-3D vision encoder for Kimi K2.5, implemented in JAX/Flax NNX.

import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# =============================================================================
# Config
# =============================================================================

@dataclass
class MoonViT3dConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    merge_kernel_size: tuple = (2, 2)
    mm_hidden_size: int = 1152
    text_hidden_size: int = 7168
    projector_ln_eps: float = 1e-5

    @classmethod
    def from_dict(cls, d: dict) -> "MoonViT3dConfig":
        return cls(
            hidden_size=d.get("vt_hidden_size", 1152),
            intermediate_size=d.get("vt_intermediate_size", 4304),
            num_attention_heads=d.get("vt_num_attention_heads", 16),
            num_hidden_layers=d.get("vt_num_hidden_layers", 27),
            patch_size=d.get("patch_size", 14),
            init_pos_emb_height=d.get("init_pos_emb_height", 64),
            init_pos_emb_width=d.get("init_pos_emb_width", 64),
            init_pos_emb_time=d.get("init_pos_emb_time", 4),
            merge_kernel_size=tuple(d.get("merge_kernel_size", [2, 2])),
            mm_hidden_size=d.get("mm_hidden_size", 1152),
            text_hidden_size=d.get("text_hidden_size", 7168),
            projector_ln_eps=d.get("projector_ln_eps", 1e-5),
        )


# =============================================================================
# Positional Embeddings
# =============================================================================

def get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """Sinusoidal 1D positional embedding. Returns (t_size, embed_dim)."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(t_size, dtype=np.float64)
    out = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def _cubic_kernel(x, a=-0.75):
    """Keys cubic interpolation kernel (same as PyTorch F.interpolate bicubic)."""
    abs_x = jnp.abs(x)
    abs_x2 = abs_x * abs_x
    abs_x3 = abs_x2 * abs_x
    f1 = (a + 2) * abs_x3 - (a + 3) * abs_x2 + 1
    f2 = a * abs_x3 - 5 * a * abs_x2 + 8 * a * abs_x - 4 * a
    return jnp.where(abs_x <= 1, f1, jnp.where(abs_x <= 2, f2, 0.0))


def _bicubic_resize_2d(img: jax.Array, target_h: int, target_w: int) -> jax.Array:
    """PyTorch-compatible bicubic resize for a 2D array (align_corners=False)."""
    in_h, in_w = img.shape
    out_y = jnp.arange(target_h, dtype=jnp.float32)
    out_x = jnp.arange(target_w, dtype=jnp.float32)
    in_y = (out_y + 0.5) * (in_h / target_h) - 0.5
    in_x = (out_x + 0.5) * (in_w / target_w) - 0.5
    iy_floor = jnp.floor(in_y).astype(jnp.int32)
    ix_floor = jnp.floor(in_x).astype(jnp.int32)

    result = jnp.zeros((target_h, target_w), dtype=jnp.float32)
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            sy = jnp.clip(iy_floor + dy, 0, in_h - 1)
            sx = jnp.clip(ix_floor + dx, 0, in_w - 1)
            wy = _cubic_kernel(in_y - (iy_floor + dy).astype(jnp.float32))
            wx = _cubic_kernel(in_x - (ix_floor + dx).astype(jnp.float32))
            result = result + img[sy[:, None], sx[None, :]] * (wy[:, None] * wx[None, :])
    return result


def _bicubic_resize_2d_batched(imgs: jax.Array, target_h: int, target_w: int) -> jax.Array:
    """Bicubic resize for batched 2D arrays: (D, H, W) -> (D, target_h, target_w)."""
    in_h, in_w = imgs.shape[1], imgs.shape[2]
    out_y = jnp.arange(target_h, dtype=jnp.float32)
    out_x = jnp.arange(target_w, dtype=jnp.float32)
    in_y = (out_y + 0.5) * (in_h / target_h) - 0.5
    in_x = (out_x + 0.5) * (in_w / target_w) - 0.5
    iy_floor = jnp.floor(in_y).astype(jnp.int32)
    ix_floor = jnp.floor(in_x).astype(jnp.int32)

    result = jnp.zeros((imgs.shape[0], target_h, target_w), dtype=jnp.float32)
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            sy = jnp.clip(iy_floor + dy, 0, in_h - 1)
            sx = jnp.clip(ix_floor + dx, 0, in_w - 1)
            wy = _cubic_kernel(in_y - (iy_floor + dy).astype(jnp.float32))
            wx = _cubic_kernel(in_x - (ix_floor + dx).astype(jnp.float32))
            # (D, target_h, target_w) via broadcasting
            result = result + imgs[:, sy[:, None], sx[None, :]] * (wy[None, :, None] * wx[None, None, :])
    return result


def _bicubic_resize_3d(weight: jax.Array, target_h: int, target_w: int) -> jax.Array:
    """Bicubic resize (H, W, D) -> (target_h, target_w, D)."""
    # Transpose to (D, H, W), resize, transpose back
    w_dhw = jnp.transpose(weight, (2, 0, 1))
    resized = _bicubic_resize_2d_batched(w_dhw, target_h, target_w)
    return jnp.transpose(resized, (1, 2, 0))


class Learnable2DInterpPosEmb(nnx.Module):
    """Learnable spatial positional embedding with bicubic interpolation + sincos temporal."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        dtype: jnp.dtype = jnp.float32,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.height = height
        self.width = width
        self.dim = dim
        _rngs = rngs or nnx.Rngs(0)
        self.weight = nnx.Param(
            jax.random.normal(_rngs.params(), (height, width, dim), dtype=jnp.float32)
        )
        # Fixed sincos temporal embedding: (num_frames, 1, dim)
        time_emb = get_1d_sincos_pos_embed(dim, num_frames)
        self.time_weight = jnp.array(time_emb[:, np.newaxis, :], dtype=jnp.float32)

    def __call__(self, x: jax.Array, grid_thws: list) -> jax.Array:
        pos_embs = []
        for t, h, w in grid_thws:
            if (h, w) == (self.height, self.width):
                pos_emb_2d = self.weight[...].reshape(-1, self.dim)
            else:
                pos_emb_2d = _bicubic_resize_3d(
                    self.weight[...], h, w
                ).reshape(-1, self.dim)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                # (t, h*w, D) = (1, h*w, D) + (t, 1, D)
                pos_emb_3d = pos_emb_2d[None, :, :] + self.time_weight[:t]
                pos_emb_3d = pos_emb_3d.reshape(-1, self.dim)

            pos_embs.append(pos_emb_3d)

        return x + jnp.concatenate(pos_embs, axis=0)


# =============================================================================
# 2D Rotary Position Embedding
# =============================================================================

class Rope2DPosEmbRepeated:
    """2D rotary position embedding. Stores cos/sin instead of complex tensors."""

    def __init__(self, dim: int, max_height: int = 512, max_width: int = 512,
                 theta_base: float = 10000.0):
        assert dim % 4 == 0, "dim must be divisible by 4"
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self._cos = None
        self._sin = None

    def _precompute(self):
        N = self.max_height * self.max_width
        flat_pos = np.arange(N, dtype=np.float32)
        x_pos = flat_pos % self.max_width    # column = x
        y_pos = flat_pos // self.max_width   # row = y

        dim_range = np.arange(0, self.dim, 4, dtype=np.float32)[:self.dim // 4]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))

        x_angles = np.outer(x_pos, freqs)  # (N, dim//4)
        y_angles = np.outer(y_pos, freqs)  # (N, dim//4)

        # Interleave: [x0, y0, x1, y1, ...] -> (N, dim//2)
        angles = np.stack([x_angles, y_angles], axis=-1).reshape(N, -1)

        cos_freqs = np.cos(angles).reshape(self.max_height, self.max_width, -1)
        sin_freqs = np.sin(angles).reshape(self.max_height, self.max_width, -1)
        self._cos = jnp.array(cos_freqs, dtype=jnp.float32)
        self._sin = jnp.array(sin_freqs, dtype=jnp.float32)

    def get_freqs(self, grid_thws: list):
        """Returns (cos, sin) each of shape (total_tokens, dim//2)."""
        if self._cos is None:
            self._precompute()

        cos_parts, sin_parts = [], []
        for t, h, w in grid_thws:
            c = self._cos[:h, :w].reshape(-1, self.dim // 2)
            s = self._sin[:h, :w].reshape(-1, self.dim // 2)
            if t > 1:
                c = jnp.tile(c, (t, 1))
                s = jnp.tile(s, (t, 1))
            cos_parts.append(c)
            sin_parts.append(s)

        return jnp.concatenate(cos_parts, axis=0), jnp.concatenate(sin_parts, axis=0)


def apply_rope(
    xq: jax.Array,
    xk: jax.Array,
    cos_freqs: jax.Array,
    sin_freqs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply 2D rotary position embedding.

    Args:
        xq, xk: (..., num_heads, head_dim)
        cos_freqs, sin_freqs: (..., head_dim//2)
    Returns:
        rotated xq, xk with same shape
    """
    cos_f = cos_freqs[..., None, :]  # (..., 1, dim//2)
    sin_f = sin_freqs[..., None, :]

    def _rotate(x):
        x = x.astype(jnp.float32)
        x_even = x[..., ::2]   # (..., heads, dim//2)
        x_odd = x[..., 1::2]
        out_even = x_even * cos_f - x_odd * sin_f
        out_odd = x_even * sin_f + x_odd * cos_f
        return jnp.stack([out_even, out_odd], axis=-1).reshape(x.shape)

    return _rotate(xq).astype(xq.dtype), _rotate(xk).astype(xk.dtype)


# =============================================================================
# Attention
# =============================================================================

def moonvit_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens: jax.Array,
    scale: float,
) -> jax.Array:
    """Packed sequence attention using segment masks.

    Args:
        q, k, v: (total_tokens, num_heads, head_dim)
        cu_seqlens: (num_segments + 1,) int32
        scale: attention scale factor
    Returns:
        output: (total_tokens, hidden_dim)
    """
    T, N, H = q.shape

    # Build segment mask from cu_seqlens
    indices = jnp.arange(T)
    segment_ids = jnp.searchsorted(cu_seqlens[1:], indices, side="right")
    segment_mask = segment_ids[:, None] == segment_ids[None, :]  # (T, T)

    # Transpose to (N, T, H) for matmul
    q = jnp.transpose(q, (1, 0, 2))
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))

    attn_weights = jnp.einsum("nth,nsh->nts", q, k) * scale
    attn_weights = jnp.where(segment_mask[None, :, :], attn_weights,
                             jnp.finfo(attn_weights.dtype).min)
    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
    output = jnp.einsum("nts,nsh->nth", attn_weights, v)
    output = jnp.transpose(output, (1, 0, 2))  # (T, N, H)
    return output.reshape(T, -1)  # (T, N*H)


# =============================================================================
# MLP
# =============================================================================

class MoonViTMLP(nnx.Module):

    def __init__(self, hidden_dim: int, mlp_dim: int,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: Optional[nnx.Rngs] = None):
        _rngs = rngs or nnx.Rngs(0)
        self.fc0 = nnx.Linear(hidden_dim, mlp_dim, use_bias=True,
                               param_dtype=dtype, rngs=_rngs)
        self.fc1 = nnx.Linear(mlp_dim, hidden_dim, use_bias=True,
                               param_dtype=dtype, rngs=_rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc0(x)
        x = jax.nn.gelu(x, approximate=True)
        return self.fc1(x)


# =============================================================================
# Encoder Layer
# =============================================================================

class MoonViTEncoderLayer(nnx.Module):

    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: Optional[nnx.Rngs] = None):
        _rngs = rngs or nnx.Rngs(0)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm0 = nnx.LayerNorm(hidden_dim, param_dtype=dtype, rngs=_rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, param_dtype=dtype, rngs=_rngs)
        self.wqkv = nnx.Linear(hidden_dim, hidden_dim * 3, use_bias=True,
                                param_dtype=dtype, rngs=_rngs)
        self.wo = nnx.Linear(hidden_dim, hidden_dim, use_bias=True,
                              param_dtype=dtype, rngs=_rngs)
        self.mlp = MoonViTMLP(hidden_dim, mlp_dim, dtype=dtype, rngs=_rngs)

    def __call__(
        self,
        x: jax.Array,
        cu_seqlens: jax.Array,
        max_seqlen: int,
        cos_freqs: jax.Array,
        sin_freqs: jax.Array,
    ) -> jax.Array:
        residual = x
        x = self.norm0(x)

        # QKV projection
        qkv = self.wqkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rope(q, k, cos_freqs, sin_freqs)

        # Attention
        attn_out = moonvit_attention(q, k, v, cu_seqlens, self.scale)
        x = self.wo(attn_out)
        x = residual + x

        # MLP
        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = residual + x

        return x


# =============================================================================
# Patch Embedding
# =============================================================================

class MoonVision3dPatchEmbed(nnx.Module):

    def __init__(self, config: MoonViT3dConfig,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: Optional[nnx.Rngs] = None):
        _rngs = rngs or nnx.Rngs(0)
        ps = config.patch_size
        self.patch_size = ps
        self.hidden_size = config.hidden_size

        self.proj = nnx.Conv(
            in_features=3,
            out_features=config.hidden_size,
            kernel_size=(ps, ps),
            strides=(ps, ps),
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.pos_emb = Learnable2DInterpPosEmb(
            height=config.init_pos_emb_height,
            width=config.init_pos_emb_width,
            num_frames=config.init_pos_emb_time,
            dim=config.hidden_size,
            dtype=jnp.float32,
            rngs=_rngs,
        )

    def __call__(self, x: jax.Array, grid_thws: list) -> jax.Array:
        """
        Args:
            x: (L, 3, patch_size, patch_size) — PyTorch NCHW format pixel patches
            grid_thws: list of (t, h, w) tuples
        Returns:
            (L, hidden_size)
        """
        L = x.shape[0]
        ps = self.patch_size
        # Convert from NCHW to NHWC for JAX conv
        x = jnp.transpose(x, (0, 2, 3, 1))  # (L, ps, ps, 3)
        x = self.proj(x)  # (L, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        x = self.pos_emb(x, grid_thws)
        return x


# =============================================================================
# Full Encoder
# =============================================================================

class MoonViT3dEncoder(nnx.Module):

    def __init__(self, config: MoonViT3dConfig,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: Optional[nnx.Rngs] = None):
        head_dim = config.hidden_size // config.num_attention_heads
        self.rope_2d = Rope2DPosEmbRepeated(dim=head_dim, max_height=512, max_width=512)

        _rngs = rngs or nnx.Rngs(0)
        blocks = []
        for _ in range(config.num_hidden_layers):
            blocks.append(MoonViTEncoderLayer(
                num_heads=config.num_attention_heads,
                hidden_dim=config.hidden_size,
                mlp_dim=config.intermediate_size,
                dtype=dtype,
                rngs=_rngs,
            ))
        self.blocks = nnx.List(blocks)
        self.final_layernorm = nnx.LayerNorm(config.hidden_size,
                                              param_dtype=dtype, rngs=_rngs)

    def __call__(self, hidden_states: jax.Array, grid_thws: list) -> jax.Array:
        cos_freqs, sin_freqs = self.rope_2d.get_freqs(grid_thws)

        # Compute cu_seqlens
        lengths = [t * h * w for t, h, w in grid_thws]
        cu_seqlens = jnp.array([0] + list(np.cumsum(lengths)), dtype=jnp.int32)
        max_seqlen = max(lengths)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen,
                                  cos_freqs, sin_freqs)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


# =============================================================================
# Patch Merger (temporal pooling + spatial downsampling)
# =============================================================================

def tpool_patch_merger(
    x: jax.Array,
    grid_thws: list,
    merge_kernel_size: tuple = (2, 2),
) -> list[jax.Array]:
    """2x2 spatial reshape + temporal mean pooling.

    Args:
        x: (total_tokens, hidden_dim)
        grid_thws: list of (t, h, w) tuples
        merge_kernel_size: spatial merge kernel
    Returns:
        list of (new_h*new_w, kh*kw, hidden_dim) arrays
    """
    d_model = x.shape[-1]
    outputs = []
    pre_sum = 0

    for t, h, w in grid_thws:
        seq = x[pre_sum:pre_sum + t * h * w]
        kh, kw = merge_kernel_size
        new_h, new_w = h // kh, w // kw
        # (t, new_h, kh, new_w, kw, D)
        reshaped = seq.reshape(t, new_h, kh, new_w, kw, d_model)
        # (t, new_h, new_w, kh, kw, D)
        reshaped = jnp.transpose(reshaped, (0, 1, 3, 2, 4, 5))
        # temporal mean: (new_h, new_w, kh, kw, D)
        merged = jnp.mean(reshaped, axis=0)
        # (new_h*new_w, kh*kw, D)
        padded = merged.reshape(new_h * new_w, kh * kw, d_model)
        outputs.append(padded)
        pre_sum += t * h * w

    return outputs


# =============================================================================
# MoonViT-3D Full Model (vision_tower)
# =============================================================================

class MoonViT3dModel(nnx.Module):

    def __init__(self, config: MoonViT3dConfig,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.patch_embed = MoonVision3dPatchEmbed(config, dtype=dtype, rngs=rngs)
        self.encoder = MoonViT3dEncoder(config, dtype=dtype, rngs=rngs)

    def __call__(self, pixel_values: jax.Array, grid_thws: list) -> list[jax.Array]:
        """
        Args:
            pixel_values: (total_patches, 3, patch_size, patch_size)
            grid_thws: list of (t, h, w)
        Returns:
            list of (num_merged_tokens, kh*kw, hidden_dim) per image/video
        """
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        hidden_states = tpool_patch_merger(hidden_states, grid_thws,
                                           self.config.merge_kernel_size)
        return hidden_states


# =============================================================================
# PatchMergerMLP (mm_projector)
# =============================================================================

class PatchMergerMLP(nnx.Module):

    def __init__(self, config: MoonViT3dConfig,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: Optional[nnx.Rngs] = None):
        _rngs = rngs or nnx.Rngs(0)
        kh, kw = config.merge_kernel_size
        hidden = config.mm_hidden_size * kh * kw
        self.mm_hidden_size = config.mm_hidden_size

        self.pre_norm = nnx.LayerNorm(config.mm_hidden_size,
                                       epsilon=config.projector_ln_eps,
                                       param_dtype=dtype, rngs=_rngs)
        self.proj_0 = nnx.Linear(hidden, hidden, use_bias=True,
                                  param_dtype=dtype, rngs=_rngs)
        self.proj_2 = nnx.Linear(hidden, config.text_hidden_size, use_bias=True,
                                  param_dtype=dtype, rngs=_rngs)

    def __call__(self, x_list: list[jax.Array]) -> list[jax.Array]:
        """
        Args:
            x_list: list of (N, kh*kw, mm_hidden_size) arrays
        Returns:
            list of (N, text_hidden_size) arrays
        """
        outputs = []
        for item in x_list:
            normed = self.pre_norm(item)  # (N, kh*kw, mm_hidden_size)
            flat = normed.reshape(item.shape[0], -1)  # (N, kh*kw * mm_hidden_size)
            out = self.proj_0(flat)
            out = jax.nn.gelu(out)
            out = self.proj_2(out)
            outputs.append(out)
        return outputs


# =============================================================================
# Weight Loading
# =============================================================================

def load_weights_from_safetensors(
    model: MoonViT3dModel,
    projector: PatchMergerMLP,
    safetensors_files: list[str],
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Load vision_tower and mm_projector weights from K2.5 safetensors.

    Args:
        model: MoonViT3dModel instance
        projector: PatchMergerMLP instance
        safetensors_files: list of safetensors file paths
        dtype: target dtype for weights
    """
    from safetensors import safe_open

    # Collect all vision/projector weights
    state = {}
    for path in safetensors_files:
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                if key.startswith("vision_tower.") or key.startswith("mm_projector."):
                    state[key] = f.get_tensor(key)

    def _to_jax(arr, target_dtype=dtype):
        return jnp.array(arr, dtype=target_dtype)

    # --- Patch Embedding ---
    # Conv2d: PyTorch (out, in, kH, kW) -> JAX (kH, kW, in, out)
    w = state["vision_tower.patch_embed.proj.weight"]
    model.patch_embed.proj.kernel[...] = _to_jax(np.transpose(w, (2, 3, 1, 0)))
    model.patch_embed.proj.bias[...] = _to_jax(state["vision_tower.patch_embed.proj.bias"])

    # Learnable pos emb: (64, 64, 1152) - direct
    model.patch_embed.pos_emb.weight[...] = jnp.array(
        state["vision_tower.patch_embed.pos_emb.weight"], dtype=jnp.float32
    )

    # --- Encoder Blocks ---
    for i in range(len(model.encoder.blocks)):
        prefix = f"vision_tower.encoder.blocks.{i}"
        block = model.encoder.blocks[i]

        # LayerNorm: weight -> scale, bias -> bias
        block.norm0.scale[...] = _to_jax(state[f"{prefix}.norm0.weight"])
        block.norm0.bias[...] = _to_jax(state[f"{prefix}.norm0.bias"])
        block.norm1.scale[...] = _to_jax(state[f"{prefix}.norm1.weight"])
        block.norm1.bias[...] = _to_jax(state[f"{prefix}.norm1.bias"])

        # Linear: weight -> kernel (transposed)
        block.wqkv.kernel[...] = _to_jax(state[f"{prefix}.wqkv.weight"].T)
        block.wqkv.bias[...] = _to_jax(state[f"{prefix}.wqkv.bias"])
        block.wo.kernel[...] = _to_jax(state[f"{prefix}.wo.weight"].T)
        block.wo.bias[...] = _to_jax(state[f"{prefix}.wo.bias"])

        # MLP
        block.mlp.fc0.kernel[...] = _to_jax(state[f"{prefix}.mlp.fc0.weight"].T)
        block.mlp.fc0.bias[...] = _to_jax(state[f"{prefix}.mlp.fc0.bias"])
        block.mlp.fc1.kernel[...] = _to_jax(state[f"{prefix}.mlp.fc1.weight"].T)
        block.mlp.fc1.bias[...] = _to_jax(state[f"{prefix}.mlp.fc1.bias"])

    # Final layernorm
    model.encoder.final_layernorm.scale[...] = _to_jax(
        state["vision_tower.encoder.final_layernorm.weight"]
    )
    model.encoder.final_layernorm.bias[...] = _to_jax(
        state["vision_tower.encoder.final_layernorm.bias"]
    )

    # --- MM Projector ---
    projector.pre_norm.scale[...] = _to_jax(state["mm_projector.pre_norm.weight"])
    projector.pre_norm.bias[...] = _to_jax(state["mm_projector.pre_norm.bias"])
    projector.proj_0.kernel[...] = _to_jax(state["mm_projector.proj.0.weight"].T)
    projector.proj_0.bias[...] = _to_jax(state["mm_projector.proj.0.bias"])
    projector.proj_2.kernel[...] = _to_jax(state["mm_projector.proj.2.weight"].T)
    projector.proj_2.bias[...] = _to_jax(state["mm_projector.proj.2.bias"])

    print(f"Loaded {len(state)} weights (vision_tower + mm_projector)")


def load_weights_from_pytorch(
    model: MoonViT3dModel,
    projector: PatchMergerMLP,
    pt_state_dict: dict,
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Load from a PyTorch state_dict (for testing against PyTorch reference)."""

    def _to_jax(tensor, target_dtype=dtype):
        arr = tensor.detach().cpu().numpy()
        return jnp.array(arr, dtype=target_dtype)

    # Patch Embedding
    w = pt_state_dict["vision_tower.patch_embed.proj.weight"]
    model.patch_embed.proj.kernel[...] = _to_jax(w.permute(2, 3, 1, 0))
    model.patch_embed.proj.bias[...] = _to_jax(pt_state_dict["vision_tower.patch_embed.proj.bias"])
    model.patch_embed.pos_emb.weight[...] = jnp.array(
        pt_state_dict["vision_tower.patch_embed.pos_emb.weight"].detach().cpu().numpy(),
        dtype=jnp.float32,
    )

    # Encoder blocks
    for i in range(len(model.encoder.blocks)):
        prefix = f"vision_tower.encoder.blocks.{i}"
        block = model.encoder.blocks[i]

        block.norm0.scale[...] = _to_jax(pt_state_dict[f"{prefix}.norm0.weight"])
        block.norm0.bias[...] = _to_jax(pt_state_dict[f"{prefix}.norm0.bias"])
        block.norm1.scale[...] = _to_jax(pt_state_dict[f"{prefix}.norm1.weight"])
        block.norm1.bias[...] = _to_jax(pt_state_dict[f"{prefix}.norm1.bias"])

        block.wqkv.kernel[...] = _to_jax(pt_state_dict[f"{prefix}.wqkv.weight"].T)
        block.wqkv.bias[...] = _to_jax(pt_state_dict[f"{prefix}.wqkv.bias"])
        block.wo.kernel[...] = _to_jax(pt_state_dict[f"{prefix}.wo.weight"].T)
        block.wo.bias[...] = _to_jax(pt_state_dict[f"{prefix}.wo.bias"])

        block.mlp.fc0.kernel[...] = _to_jax(pt_state_dict[f"{prefix}.mlp.fc0.weight"].T)
        block.mlp.fc0.bias[...] = _to_jax(pt_state_dict[f"{prefix}.mlp.fc0.bias"])
        block.mlp.fc1.kernel[...] = _to_jax(pt_state_dict[f"{prefix}.mlp.fc1.weight"].T)
        block.mlp.fc1.bias[...] = _to_jax(pt_state_dict[f"{prefix}.mlp.fc1.bias"])

    model.encoder.final_layernorm.scale[...] = _to_jax(
        pt_state_dict["vision_tower.encoder.final_layernorm.weight"]
    )
    model.encoder.final_layernorm.bias[...] = _to_jax(
        pt_state_dict["vision_tower.encoder.final_layernorm.bias"]
    )

    # MM Projector
    projector.pre_norm.scale[...] = _to_jax(pt_state_dict["mm_projector.pre_norm.weight"])
    projector.pre_norm.bias[...] = _to_jax(pt_state_dict["mm_projector.pre_norm.bias"])
    projector.proj_0.kernel[...] = _to_jax(pt_state_dict["mm_projector.proj.0.weight"].T)
    projector.proj_0.bias[...] = _to_jax(pt_state_dict["mm_projector.proj.0.bias"])
    projector.proj_2.kernel[...] = _to_jax(pt_state_dict["mm_projector.proj.2.weight"].T)
    projector.proj_2.bias[...] = _to_jax(pt_state_dict["mm_projector.proj.2.bias"])
