import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_denoising(x, sigma=0.5):
    """
    Gaussian noise smoothing defense
    """
    return gaussian_filter(x, sigma=sigma)
