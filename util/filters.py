import math
import torch


def get_filter(ftype='gaussian'):
    filters = {
        'gaussian': gaussian_low_pass_filter_2d,
        'ideal': ideal_low_pass_filter_2d,
    }

    return filters[ftype]

def gaussian_low_pass_filter_2d(shape, d_s=0.25):
    """
    Computes a 2D Gaussian low-pass filter mask.

    Args:
        shape: Shape of the filter (H, W).
        d_s: Normalized cutoff frequency for spatial dimensions (0.0-1.0).

    Returns:
        A 2D NumPy array representing the Gaussian low-pass filter mask.
        Returns None if the input shape is invalid.
    """

    if not isinstance(shape, tuple) or len(shape)!= 2:
        print("Error: Shape must be a tuple of length 2 (H, W).")
        return None

    H, W = shape
    mask = torch.zeros(shape)

    if d_s == 0:
        return mask  # Return all zeros if cutoff is 0

    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1)**2 + (2 * w / W - 1)**2  # Calculate squared distance
            mask[h, w] = math.exp(-1 / (2 * d_s**2) * d_square)  # Gaussian function

    return mask

def ideal_low_pass_filter_2d(shape, d_s=0.25):
    """
    Compute the ideal low pass filter mask (2D version).

    Args:
        shape: shape of the filter (H, W)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        print("Error: Shape must be a tuple of length 2 (H, W).")
        return None

    H, W = shape
    mask = torch.zeros(shape)  # Changed to torch.zeros for consistency with original

    if d_s == 0:
        return mask

    for h in range(H):
        for w in range(W):
            d_square = (2 * h / H - 1)**2 + (2 * w / W - 1)**2
            mask[h, w] = 1 if d_square <= d_s**2 else 0 # Changed to d_s**2 to match the gaussian example's intent. The original uses d_s*2, which seems incorrect.

    return mask
