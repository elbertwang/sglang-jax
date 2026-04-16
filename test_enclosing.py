import numpy as np
import ml_dtypes

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

def test():
    # Mock data
    np.random.seed(42)
    hf_shape = (2048, 7168)
    full_data = np.random.randint(0, 255, size=hf_shape).astype(np.uint8)
    scale_inv_full = np.random.rand(hf_shape[0]//128, hf_shape[1]//128).astype(np.float32)
    
    # We want to request JAX slice [576:640, 0:896]
    hf_slice = (slice(576, 640), slice(0, 896))
    
    # 1. Slow path fallback result
    data_full = blockwise_dequant_fp8_np(full_data, scale_inv_full, block_size=128)
    result_slow = data_full[hf_slice].astype(ml_dtypes.bfloat16)
    
    # 2. Fast path logic
    block_size = 128
    def get_enclosing_slice(s, max_dim):
        start = s.start or 0
        stop = s.stop if s.stop is not None else max_dim
        aligned_start = (start // block_size) * block_size
        aligned_stop = min(((stop + block_size - 1) // block_size) * block_size, max_dim)
        return slice(aligned_start, aligned_stop)
    
    s0 = get_enclosing_slice(hf_slice[0], hf_shape[0])
    s1 = get_enclosing_slice(hf_slice[1], hf_shape[1])
    enclosing_slice = (s0, s1)
    
    data = full_data[enclosing_slice]
    
    scale_s0 = slice(s0.start // block_size, (s0.stop + block_size - 1) // block_size)
    scale_s1 = slice(s1.start // block_size, (s1.stop + block_size - 1) // block_size)
    scale_inv_sliced = scale_inv_full[scale_s0, scale_s1]
    
    data = blockwise_dequant_fp8_np(data, scale_inv_sliced, block_size=block_size)
    
    start0 = (hf_slice[0].start or 0) - s0.start
    stop0 = start0 + ((hf_slice[0].stop if hf_slice[0].stop is not None else hf_shape[0]) - (hf_slice[0].start or 0))
    start1 = (hf_slice[1].start or 0) - s1.start
    stop1 = start1 + ((hf_slice[1].stop if hf_slice[1].stop is not None else hf_shape[1]) - (hf_slice[1].start or 0))
    
    data = data[start0:stop0, start1:stop1]
    result_fast = data.astype(ml_dtypes.bfloat16)
    
    print(f"Shape match: {result_slow.shape} == {result_fast.shape}")
    print(f"Value match: {np.allclose(result_slow, result_fast)}")
    print(f"Read bounds: s0={s0}, s1={s1}, local bounds: {start0}:{stop0}, {start1}:{stop1}")

test()
