import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Tạo vector vị trí (pos)
    pos = np.arange(seq_length).reshape(-1, 1)  # (seq_length, 1)
    
    # Tạo vector dimension (i)
    i = np.arange(d_model).reshape(1, -1)       # (1, d_model)
    
    # Tính phần chia (denominator)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    
    # Tính góc
    angle_rads = pos * angle_rates
    
    # Áp dụng sin cho index chẵn, cos cho index lẻ
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])  # index chẵn
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])  # index lẻ
    
    return pe