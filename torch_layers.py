import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch import Tensor
from typing import Union
from math import ceil


class Add(nn.Module):
    @staticmethod
    def forward(X: list[Tensor]) -> Tensor:
        r"""Returns sum of first two tensors from the list.

        Args:
            X: List of at least two tensors.
        """
        assert len(X) >= 2
        return X[0] + X[1]


class Multiply(nn.Module):
    @staticmethod
    def forward(X: list[Tensor]) -> Tensor:
        r"""Returns element-wise multiplication of first two tensors from the list.

        Args:
            X: List of at least two tensors.
        """
        assert len(X) >= 2
        return X[0] * X[1]


class Concatenate(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, X: list[Tensor]) -> Tensor:
        r"""Returns the concatenation of all tensors in a list.

        Args:
            X: List of tensors.
        """
        return torch.concatenate(X, dim=self.axis)


class Reshape(nn.Module):
    def __init__(self, shape: list[int]):
        super().__init__()
        self.shape = shape.copy()
        self.shape.insert(0, -1)

    def forward(self, X: Tensor) -> Tensor:
        r"""Reshapes a tensor

        Args:
            X: Input tensor.
        """
        return torch.reshape(X, shape=self.shape)


class GlobalAveragePool2d(nn.AvgPool2d):
    def __init__(self, kernel_size: _size_2_t):
        super().__init__(kernel_size=kernel_size)

    def forward(self, input: Tensor) -> Tensor:
        r"""Applies global average pooling on a given tensor

        Args:
            input: Input tensor.
        """
        pooled_input = super().forward(input)
        return pooled_input.flatten(start_dim=1)


class Conv2dSame(nn.Conv2d):
    r"""Convolutional layer supporting 'same' padding with stride greater than 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        expected_ih: int,
        expected_iw: int,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        nn.Module.__init__(self)

        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        transposed = False
        output_padding = _pair(0)

        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        # Computes the pad based on expected size of input images.
        self.pad_h = self.calc_same_pad(i=expected_ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        self.pad_w = self.calc_same_pad(i=expected_iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)  # type: ignore

        if transposed:
            self.weight = nn.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @staticmethod
    def calc_same_pad(i: int, k: int, s: int, d: int) -> int:
        return max((ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, input: Tensor) -> Tensor:
        pad_h = self.pad_h
        pad_w = self.pad_w
        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            "valid",
            self.dilation,
            self.groups,
        )


class ChannelWiseSoftmax(nn.Module):
    r"""Channel-wise Softmax. When given a tensor of shape ``N x C x H x W``, apply Softmax to each channel."""

    @staticmethod
    def forward(input: Tensor) -> Tensor:
        assert input.dim() == 4, "ChannelWiseSoftmax requires a 4D tensor as input"
        # Reshape tensor to N x C x (H x W)
        reshaped = input.view(*input.size()[:2], -1)
        # Apply softmax along 2nd dimension than reshape to original
        return nn.Softmax(2)(reshaped).view_as(input)
