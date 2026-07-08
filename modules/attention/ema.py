import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class EMA(nn.Module):

    def __init__(self, channels, scales=(3,5,7), reduction=4):
        super().__init__()
        self.channels = channels
        self.scales = scales

        self.branches = nn.ModuleList()
        for k in scales:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True)
                )
            )

        # fuse after concat
        self.fuse = nn.Conv2d(len(scales) * channels, channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(channels)

        # lightweight channel recalibration
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        # multi-scale depthwise features
        feats = [b(x) for b in self.branches]
        cat = torch.cat(feats, dim=1)
        fused = self.fuse_bn(self.fuse(cat))

        # channel recalibration
        w = self.pool(fused)
        w = self.fc(w).sigmoid()

        return fused * w + x  # residual within block


class C2EMA(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, scales=(3,5,7), reduction=4):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.blocks = nn.ModuleList()
        if n >= 1:
            self.blocks.append(EMA(self.c, scales=scales, reduction=reduction))
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
    print("\n--- EMA ---")
    ema = EMA(256)
    y = ema(x)
    print("EMA:", x.shape, "->", y.shape)

    print("\n--- C2EMA ---")
    c2ema = C2EMA(256, 256, n=1)
    y2 = c2ema(x)
    print("C2EMA:", x.shape, "->", y2.shape)
    print("C2EMA params:", sum(p.numel() for p in c2ema.parameters()))
