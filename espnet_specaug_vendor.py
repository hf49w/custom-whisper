from __future__ import annotations

from distutils.version import LooseVersion
from typing import Sequence, Union

import torch


# Vendored from the local paper codebase:
# D:\研究生\智能体\espnet-multimodal-asr\espnet2\asr\specaug\specaug.py
# D:\研究生\智能体\espnet-multimodal-asr\espnet2\layers\mask_along_axis.py
# D:\研究生\智能体\espnet-multimodal-asr\espnet2\layers\time_warp.py
# The only adaptations are removal of ESPnet/typeguard dependencies and an
# inline pad_list helper.

if LooseVersion(torch.__version__) >= LooseVersion("1.1"):
    DEFAULT_TIME_WARP_MODE = "bicubic"
else:
    # PyTorch 1.0 doesn't implement bicubic interpolation here.
    DEFAULT_TIME_WARP_MODE = "bilinear"


def pad_list(xs: Sequence[torch.Tensor], pad_value: float) -> torch.Tensor:
    if not xs:
        raise ValueError("pad_list requires at least one tensor.")
    max_len = max(x.size(0) for x in xs)
    pad_shape = (len(xs), max_len) + tuple(xs[0].shape[1:])
    padded = xs[0].new_full(pad_shape, pad_value)
    for index, tensor in enumerate(xs):
        padded[index, : tensor.size(0)] = tensor
    return padded


def time_warp(
    x: torch.Tensor,
    window: int = 80,
    mode: str = DEFAULT_TIME_WARP_MODE,
) -> torch.Tensor:
    """Time warping using torch.interpolate.

    Args:
        x: (Batch, Time, Freq)
        window: time warp parameter
        mode: Interpolate mode
    """

    org_size = x.size()
    if x.dim() == 3:
        x = x[:, None]

    t = x.shape[2]
    if t - window <= window:
        return x.view(*org_size)

    center = torch.randint(window, t - window, (1,))[0]
    warped = torch.randint(center - window, center + window, (1,))[0] + 1

    left = torch.nn.functional.interpolate(
        x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
    )
    right = torch.nn.functional.interpolate(
        x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False
    )

    if x.requires_grad:
        x = torch.cat([left, right], dim=-2)
    else:
        x[:, :, :warped] = left
        x[:, :, warped:] = right

    return x.view(*org_size)


class TimeWarp(torch.nn.Module):
    """Time warping using torch.interpolate."""

    def __init__(self, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
        super().__init__()
        self.window = window
        self.mode = mode

    def extra_repr(self) -> str:
        return f"window={self.window}, mode={self.mode}"

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        if x_lengths is None or all(length == x_lengths[0] for length in x_lengths):
            y = time_warp(x, window=self.window, mode=self.mode)
        else:
            ys = []
            for index in range(x.size(0)):
                warped = time_warp(
                    x[index][None, : x_lengths[index]],
                    window=self.window,
                    mode=self.mode,
                )[0]
                ys.append(warped)
            y = pad_list(ys, 0.0)

        return y, x_lengths


def mask_along_axis(
    spec: torch.Tensor,
    spec_lengths: torch.Tensor,
    mask_width_range: Sequence[int] = (0, 30),
    dim: int = 1,
    num_mask: int = 2,
    replace_with_zero: bool = True,
):
    """Apply mask along the specified direction."""

    org_size = spec.size()
    if spec.dim() == 4:
        spec = spec.view(-1, spec.size(2), spec.size(3))

    batch_size = spec.shape[0]
    axis_size = spec.shape[dim]
    mask_length = torch.randint(
        mask_width_range[0],
        mask_width_range[1],
        (batch_size, num_mask),
        device=spec.device,
    ).unsqueeze(2)

    mask_pos = torch.randint(
        0,
        max(1, axis_size - mask_length.max()),
        (batch_size, num_mask),
        device=spec.device,
    ).unsqueeze(2)

    axis_range = torch.arange(axis_size, device=spec.device)[None, None, :]
    mask = (mask_pos <= axis_range) * (axis_range < (mask_pos + mask_length))
    mask = mask.any(dim=1)
    if dim == 1:
        mask = mask.unsqueeze(2)
    elif dim == 2:
        mask = mask.unsqueeze(1)

    value = 0.0 if replace_with_zero else spec.mean()
    if spec.requires_grad:
        spec = spec.masked_fill(mask, value)
    else:
        spec = spec.masked_fill_(mask, value)
    spec = spec.view(*org_size)
    return spec, spec_lengths


class MaskAlongAxis(torch.nn.Module):
    def __init__(
        self,
        mask_width_range: Union[int, Sequence[int]] = (0, 30),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: {mask_width_range}"
            )
        if mask_width_range[1] <= mask_width_range[0]:
            raise ValueError("mask_width_range max must be greater than min.")

        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self) -> str:
        return (
            f"mask_width_range={self.mask_width_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        return mask_along_axis(
            spec,
            spec_lengths,
            mask_width_range=self.mask_width_range,
            dim=self.dim,
            num_mask=self.num_mask,
            replace_with_zero=self.replace_with_zero,
        )


class SpecAug(torch.nn.Module):
    """Implementation of SpecAug from the ESPnet multimodal-asr codebase."""

    def __init__(
        self,
        apply_time_warp: bool = True,
        time_warp_window: int = 5,
        time_warp_mode: str = DEFAULT_TIME_WARP_MODE,
        apply_freq_mask: bool = True,
        freq_mask_width_range: Union[int, Sequence[int]] = (0, 20),
        num_freq_mask: int = 2,
        apply_time_mask: bool = True,
        time_mask_width_range: Union[int, Sequence[int]] = (0, 100),
        num_time_mask: int = 2,
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError("Either one of time_warp, time_mask, or freq_mask should be applied")
        super().__init__()
        self.apply_time_warp = apply_time_warp
        self.apply_freq_mask = apply_freq_mask
        self.apply_time_mask = apply_time_mask

        if apply_time_warp:
            self.time_warp = TimeWarp(window=time_warp_window, mode=time_warp_mode)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskAlongAxis(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            self.time_mask = MaskAlongAxis(
                dim="time",
                mask_width_range=time_mask_width_range,
                num_mask=num_time_mask,
            )
        else:
            self.time_mask = None

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        if self.time_warp is not None:
            x, x_lengths = self.time_warp(x, x_lengths)
        if self.freq_mask is not None:
            x, x_lengths = self.freq_mask(x, x_lengths)
        if self.time_mask is not None:
            x, x_lengths = self.time_mask(x, x_lengths)
        return x, x_lengths
