"""Image workloads: Gaussian blur, resize, Sobel edges.

Each takes a float32 / float64 tensor shaped (..., H, W), e.g. (N, C, H, W), and runs on whatever device it
is on. (float16 is out: antialiased resize has no half-precision CPU kernel, so CPU and GPU would diverge.)
Every channel of every image is processed independently, so all three split along any leading dim.
"""
import math

import torch
import torch.nn.functional as F


def blur(images, sigma=2.0):
    """Gaussian blur of each channel, same shape out. Separable: one row pass, one column pass."""
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    k = torch.exp(-((x / sigma) ** 2) / 2)
    k = k / k.sum()
    # Planes as channels of one image (depthwise conv): as a batch of planes, CPU conv is ~17x slower.
    planes = images.reshape(1, -1, *images.shape[-2:])
    n = planes.shape[1]
    planes = F.pad(planes, (radius,) * 4, mode="replicate")  # replicate works at any image size
    planes = F.conv2d(planes, k.view(1, 1, 1, -1).expand(n, 1, 1, -1), groups=n)
    planes = F.conv2d(planes, k.view(1, 1, -1, 1).expand(n, 1, -1, 1), groups=n)
    return planes.reshape(images.shape)


def resize(images, scale=0.5):
    """Bilinear, antialiased resize of each channel by scale (H, W -> floor(H*scale), floor(W*scale))."""
    out = F.interpolate(_planes(images), scale_factor=scale, mode="bilinear", antialias=True)
    return out.reshape(*images.shape[:-2], *out.shape[-2:])


def sobel(images):
    """Sobel gradient magnitude of each channel, same shape out."""
    gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=images.dtype, device=images.device)
    g = F.conv2d(F.pad(_planes(images), (1, 1, 1, 1), mode="replicate"), torch.stack([gx, gx.T]).unsqueeze(1))
    return torch.hypot(g[:, 0], g[:, 1]).reshape(images.shape)


def _planes(images):
    """(..., H, W) -> (N, 1, H, W): one plane per channel per image."""
    return images.reshape(-1, 1, *images.shape[-2:])
