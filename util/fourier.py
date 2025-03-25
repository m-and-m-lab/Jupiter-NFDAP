import torch
from util.filters import get_filter


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
        fft_image = torch.fft.fftn(image, dim=(-2, -1), norm='ortho')
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
        ifft_result = torch.fft.ifftn(ifft_image, dim=(-2, -1), norm='ortho')
    else:
        raise ValueError("Input fft image must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

    if get_img:
        if fft_shifted.ndim == 3:
            ifft_result = ifft_result.permute(1, 2, 0)
        elif fft_shifted.ndim == 4:
            ifft_result = ifft_result.permute(0, 2, 3, 1)
        return (ifft_result * 255.0).to(torch.uint8)

    return ifft_result



class ImageFiltering:
    """
    A class for applying various frequency domain filters to images.
    """

    def __init__(self, filter_type, freq_r):
        """
        Initializes the ImageFiltering class.

        Args:
            filter_type ('str'): Type of gilter to apply ("gaussian", "ideal")
            freq_r (float): The radius of the low-frequency region to be suppressed.
        """
        self.filter_type = filter_type
        self.freq_r = freq_r

    def highpass_img(self, imgs, get_real=True, get_img=False):
        """
        Applies a high-pass filter to an image in the frequency domain.

        Args:
            imgs (torch.Tensor): The input image as a PyTorch tensor, shape (B, C, H, W)
            get_img (bool): If set to True, returns a ready-to-plot batch of images (uint8, 0-255).
            get_real (bool): If set to True, returns only real part of inverse transform

        Returns:
            torch.Tensor: The high-pass filtered image.
        """
        _, C, H, W = imgs.shape
        fft = fft_per_channel(imgs)

        # TODO seperate high and low pass
        filter_mask = get_filter(self.filter_type)(shape=(H, W), d_s=self.freq_r)
        filter_mask = filter_mask.to(fft.device)
        high_mask = (1 - filter_mask)

        freq_high = fft * high_mask.reshape(-1, 1, H, W)

        img_high = ifft_per_channel(freq_high, get_img=get_img)

        if get_real:
            return img_high.real
        return img_high
