import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.conv import Conv


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.sigmoid()


class C2ECA(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5,
                 gamma=2, b=1):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c1 * e) 

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.blocks = nn.ModuleList()
        self.extra = nn.ModuleList()

        if n >= 1:
            self.blocks.append(ECA(self.c, gamma, b))

        for _ in range(max(0, n - 1)):
            self.extra.append(Conv(self.c, self.c, 3, 1))

        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        for blk in self.blocks:
            y.append(blk(y[-1]))

        for blk in self.extra:
            y.append(blk(y[-1]))

        out = self.cv2(torch.cat(y, 1))
        return x + out if self.shortcut else out


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    model = C2ECA(64, 64, n=1)
    print("Output:", model(x).shape)
