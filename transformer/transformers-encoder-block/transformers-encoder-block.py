import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta

    return out 

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here

    # linear
    B, N, d_model = Q.shape
    d_k = d_model // num_heads

    # prj
    Q_prj = Q @ W_q
    K_prj = K @ W_k
    V_prj = V @ W_v

    def split_head(x):
        x = x.reshape(B, N, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)
    
    # split head
    Qh = split_head(Q_prj)
    Kh = split_head(K_prj)
    Vh = split_head(V_prj)

    # scale
    scores = Qh @ Kh.swapaxes(-2, -1) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    context = weights @ Vh

    # concat 
    context = context.transpose(0, 2, 1, 3).reshape(B, N, d_model)

    # out prj
    output = context @ W_o

    return output
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    h = x @ W1 + b1
    h = np.maximum(0, h)  # ReLU
    out = h @ W2 + b2
    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)
    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_out, gamma2, beta2)

    return x