import torch


def fft_per_channel(image):
    """
    Computes the 2D FFT of an image, channel by channel, using PyTorch,
    and applies fftshift.

    Args:
        image (torch.Tensor): Input image tensor with shape (C, H, W) or (B, C, H, W).

    Returns:
        torch.Tensor: Complex-valued FFT of the image, shape (C, H, W) or (B, C, H, W), shifted.
    """
    if image.ndim == 3:
        fft_image = torch.fft.fft2(image, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_image, dim=(-2, -1))
    elif image.ndim == 4:
        fft_image = torch.fft.fft2(image, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_image, dim=(-2, -1))
    else:
        raise ValueError("Input image must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

    return fft_shifted

def ifft_per_channel(fft_shifted):
    """
    Computes the 2D inverse FFT of a shifted FFT image, channel by channel,
    using PyTorch, and applies ifftshift.

    Args:
        fft_shifted (torch.Tensor): Complex-valued FFT of the image, shape (C, H, W) or (B, C, H, W), shifted.

    Returns:
        torch.Tensor: Real-valued inverse FFT of the image, shape (C, H, W) or (B, C, H, W).
    """
    if fft_shifted.ndim == 3:
        ifft_image = torch.fft.ifftshift(fft_shifted, dim=(-2, -1))
        ifft_result = torch.fft.ifft2(ifft_image, dim=(-2, -1)).real
    elif fft_shifted.ndim == 4:
        ifft_image = torch.fft.ifftshift(fft_shifted, dim=(-2, -1))
        ifft_result = torch.fft.ifft2(ifft_image, dim=(-2, -1)).real
    else:
        raise ValueError("Input fft image must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

    return ifft_result
