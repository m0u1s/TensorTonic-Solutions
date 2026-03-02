import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    B, N, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projection
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split heads: (B,N,h,d_k) → (B,h,N,d_k)
    def split_heads(x):
        return x.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)

    Qh = split_heads(Q_proj)
    Kh = split_heads(K_proj)
    Vh = split_heads(V_proj)

    # Scaled dot-product attention (vectorized)
    scores = Qh @ Kh.swapaxes(-2, -1) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    context = weights @ Vh  # (B,h,N,d_k)

    # Concat heads: (B,h,N,d_k) → (B,N,d_model)
    context = context.transpose(0, 2, 1, 3).reshape(B, N, d_model)

    # Output projection
    output = context @ W_o

    return output