import torch.nn as nn
import torch.nn.functional as functional
import torch
import numpy as np

class deform_conv_v1(nn.Module):
    r"""Applies Deformable Convolution version 1 over a 4D input(a mini-batch of 2D inputs with additional channel
    dimension) as described in the paper: Deformable Convolutional Networks.

    Math:
        $y(p_0) = \sum_{p_n\in{R}} w(p_n) * x(p_0+p_n+\Delta{p_n})$

    Note:
        1, Dilation is just a special case of deformable convolution and is meaningless in deformable convolution,
           which is forced to be 1 in this implementation.
        2, Data format is restricted to be [batch_size, channels, height, width] in this implementation.
        3, Only square feature map (height == width and symmetric padding) is supported in this implementation.
        4, Convolution kernel must be odd and convolution kernel center starts exactly at the 0th row, 0th column
           element in un-padded feature map, which is also the padding-th row, padding-th column in padded feature map
           as we force padding to be symmetric, e.g. if kernel size is 3, then padding must be 1.

    Args:
        in_channels(C_in): input channels for deformable convolution.
        out_channels(C_out): output channels for deformable convolution
        kernel_size(k): kernel size of both normal convolution and offset convolution
        stride(s): stride of both normal convolution and offset convolution
        padding(p): padding of both normal convolution and offset convolution
        bias: whether to add bias in both normal convolution and offset convolution

    Shape:
        input: (N, $C_in$, $H_in$, $W_in$)
        output: (N, $C_out$, $H_out$, $W_out$)
            $H_out = W_out = \lfloor{\frac{H_{in} + 2*p - k}{s} + 1}\rfloor$

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Convolution operator to generate feature map index offset, set padding=0 for manually dealing.
        self.p_conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size * kernel_size, padding=0,
                                kernel_size=kernel_size, stride=stride, bias=bias)
        # Initialize weight and bias in p_conv to be zero.
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv.bias, 0)

        # Convolution operator for normal feature extraction
        # Note that feature map will be re-arranged using index offset from [n, c, h, w] -> [n, c, kh, kw] where
        # k is the kernel size. Stride of this convolution operator is just k and this convolution performs
        # non-overlapping scan across re-arranged feature map to implement deformable convolution.
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=kernel_size, padding=0, bias=bias)

        # Manual padding of input feature map to generate offset feature map
        self.zero_padding = nn.ZeroPad2d(padding=padding)

    def forward(self, x):
        x = self.zero_padding(x)
        padded_h, padded_w = x.size()[2], x.size()[3]
        assert padded_h == padded_w

        # [n, 2k*k, h_out, w_out]
        offset = self.p_conv(x)

        n, _, h_out, w_out = list(offset.size())

        # [n, 2k*k, h_out, w_out]
        p = self.get_p(offset, h_out, w_out)

        # [n, h_out, w_out, 2k*k]
        p = p.permute([0, 2, 3, 1])

        # [n, h_out, w_out, 2, k*k]
        p = torch.reshape(p, [n, h_out, w_out, 2, -1])

        # [n, h_out, w_out, k, k, 2]
        p = torch.reshape(p.permute([0, 1, 2, 4, 3]), [n, h_out, w_out, self.kernel_size, -1, 2])

        # [n, h_out, k, w_out, k, 2]
        grid = p.permute([0, 1, 3, 2, 4, 5])

        # [n, h_out*k, w_out*k, 2]
        grid = torch.reshape(grid, [n, h_out * self.kernel_size, -1, 2])

        # normalize grid to be in range [-1, 1]
        grid = torch.clamp(grid * 2 / (padded_h - 1) - 1, -1, 1)

        # [n, c_in, h_out*k, h_out*w]
        x = functional.grid_sample(x, grid, align_corners=True)

        # [n, c_out, h_out, w_out]
        x = self.conv(x)

        return x

    def get_p(self, offset, h_out, w_out):

        # 1, 2k*k, h_out, w_out. p_0[:, :k*k, :, :] is x coordinates, while p_0[:, k*k:, :, :] is y coordinates
        p_0 = self.get_p_0(h_out, w_out)

        # 1, 2k*k, h_out, w_out
        p_0 = torch.reshape(p_0, [1, 2, self.kernel_size * self.kernel_size, h_out, w_out]).view(1, -1, h_out, w_out)

        # 1, 2k*k, 1, 1. p_n[:, :k*k, :, :] is x coordinates, while p_n[:, k*k:, :, :] is y coordinates
        p_n = self.get_p_n()

        # 1, 2k*k, 1, 1
        p_n = torch.reshape(p_n, [1, 2, -1, 1, 1]).view(1, -1, 1, 1)

        # n, 2k*k, h_out, w_out
        p = offset + p_0 + p_n

        return p

    def get_p_0(self, h_out, w_out):

        p_0_row, p_0_col = torch.meshgrid(
            torch.arange(start=self.padding, end=self.padding + h_out * self.stride, step=self.stride),
            torch.arange(start=self.padding, end=self.padding + w_out * self.stride, step=self.stride)
        )
        p_0_row = torch.flatten(p_0_row).view(1, 1, h_out, w_out).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        p_0_col = torch.flatten(p_0_col).view(1, 1, h_out, w_out).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        p_0 = torch.cat([p_0_col, p_0_row], 1)
        return p_0

    def get_p_n(self):
        p_n_row, p_n_col = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_col), torch.flatten(p_n_row)], 0)
        p_n = p_n.view(1, 2 * self.kernel_size * self.kernel_size, 1, 1)

        return p_n

if __name__ == '__main__':
    x = np.random.rand(8, 3, 32, 32).astype(np.float32)
    x = torch.from_numpy(x)

    model = deform_conv_v1(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)

    res = model(x)
    print (res)
