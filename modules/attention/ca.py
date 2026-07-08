import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CA)
    Paper: Coordinate Attention for Efficient Mobile Network Design
    Efficient channel-spatial attention capturing long-range dependencies
    along height and width separately.
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # Pool along width
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # Pool along height

        mip = max(8, channels // reduction)

        # Shared 1x1 bottleneck
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        # Two attention projections
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # Height attention branch
        x_h = self.pool_h(x)              # (B, C, H, 1)

        # Width attention branch
        x_w = self.pool_w(x)              # (B, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)     # (B, C, W, 1)

        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)

        # Shared transform
        y = self.act(self.bn1(self.conv1(y)))

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class C2CA(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, reduction=32):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.blocks = nn.ModuleList()

        if n >= 1:
            self.blocks.append(CoordinateAttention(self.c, reduction=reduction))

        for _ in range(max(0, n - 1)):
            self.blocks.append(Conv(self.c, self.c, 3, 1))

        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            y.append(blk(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        return x + out if self.shortcut else out


if __name__ == "__main__":
    x = torch.randn(2, 256, 32, 32)

    print("\n--- Coordinate Attention ---")
    ca = CoordinateAttention(256, reduction=32)
    y = ca(x)
    print("CA:", x.shape, "->", y.shape)

    print("\n--- C2CA ---")
    c2ca = C2CA(256, 256, n=1)
    y2 = c2ca(x)
    print("C2CA:", x.shape, "->", y2.shape)
