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
    if image.ndim == 4 or image.ndim == 3:
        fft_image = torch.fft.fftn(image, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_image, dim=(-2, -1))
    else:
        raise ValueError("Input image must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

    return fft_shifted

def ifft_per_channel(fft_shifted, get_img=False):
    """
    Computes the 2D inverse FFT of a shifted FFT image, channel by channel,
    using PyTorch, and applies ifftshift.

    Args:
        fft_shifted (torch.Tensor): Complex-valued FFT of the image, shape (C, H, W) or (B, C, H, W), shifted.
        get_img (bool): If set to True, returns a ready-to-plot batch of images (uint8, 0-255).

    Returns:
        torch.Tensor: Real-valued inverse FFT of the image, shape (C, H, W) or (B, C, H, W).
                    If get_img is True, returns a uint8 tensor representing the image(s).
    """
    if fft_shifted.ndim == 3 or fft_shifted.ndim == 4:
        ifft_image = torch.fft.ifftshift(fft_shifted, dim=(-2, -1))
        ifft_result = torch.fft.ifftn(ifft_image, dim=(-2, -1))
    else:
        raise ValueError("Input fft image must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

    if get_img:
        if fft_shifted.ndim == 3
            ifft_result = ifft_result.permute(1, 2, 0)
        elif fft_shifted.ndim == 4:
            ifft_result = ifft_result.permute(0, 2, 3, 1)
        return (ifft_result * 255.0).to(torch.uint8)

    return ifft_result
