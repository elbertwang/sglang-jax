import numpy as np

def blockwise_dequant_fp8_np(weight_fp8: np.ndarray, scale_inv: np.ndarray, block_size: int = 128) -> np.ndarray:
    out_dim, in_dim = weight_fp8.shape
    w = weight_fp8.astype(np.float32)

    out_blocks = (out_dim + block_size - 1) // block_size
    in_blocks = (in_dim + block_size - 1) // block_size

    pad_out = out_blocks * block_size - out_dim
    pad_in = in_blocks * block_size - in_dim
    if pad_out > 0 or pad_in > 0:
        w = np.pad(w, ((0, pad_out), (0, pad_in)))

    w = w.reshape(out_blocks, block_size, in_blocks, block_size)
    s = scale_inv[:out_blocks, :in_blocks].astype(np.float32)
    w = w / s[:, None, :, None]

    w = w.reshape(out_blocks * block_size, in_blocks * block_size)
    return w[:out_dim, :in_dim]

np.random.seed(0)
# Mock weight shape: out=2048, in=7168
full_w = np.random.randint(0, 255, size=(2048, 7168)).astype(np.uint8)
full_s = np.random.rand(2048//128, 7168//128).astype(np.float32)

# JAX inner_slice for transposed weight (JAX weight is [7168, 2048])
# JAX needs slice e.g. [0:896, 0:256] -> inner_slice = (slice(0, 896), slice(0, 256))
# This is inner_slice on the JAX tensor (already transposed).
jax_slice = (slice(0, 896), slice(0, 256))

# Path 1: Load full, dequant full, transpose, slice (current slow path)
full_dequant = blockwise_dequant_fp8_np(full_w, full_s)
jax_weight_full = np.transpose(full_dequant)
result_slow = jax_weight_full[jax_slice]

# Path 2: Slice first, then dequant (fast path)
# Since do_transpose is True, JAX's inner_slice = (s_in, s_out) corresponds to HF's (s_out, s_in).
hf_slice = (jax_slice[1], jax_slice[0]) # (slice(0, 256), slice(0, 896))

# Check if hf_slice is block-aligned
def get_block_slice(s, block_size=128):
    start = s.start or 0
    stop = s.stop
    assert start % block_size == 0
    assert stop % block_size == 0
    return slice(start // block_size, stop // block_size)

scale_slice = (get_block_slice(hf_slice[0]), get_block_slice(hf_slice[1]))

w_sliced = full_w[hf_slice]
s_sliced = full_s[scale_slice]

sliced_dequant = blockwise_dequant_fp8_np(w_sliced, s_sliced)
result_fast = np.transpose(sliced_dequant)

print(f"Shape match: {result_slow.shape} == {result_fast.shape}")
print(f"Value match: {np.allclose(result_slow, result_fast)}")
