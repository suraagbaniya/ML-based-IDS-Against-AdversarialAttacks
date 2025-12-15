import numpy as np

def bit_depth_reduction(x, bits=8):
    """
    Feature squeezing using bit depth reduction
    """
    max_val = np.max(x)
    x_squeezed = np.round(x / max_val * (2**bits)) / (2**bits)
    return x_squeezed
