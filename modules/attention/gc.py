import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class GlobalContext(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        self.mask_conv = nn.Conv2d(channels, 1, kernel_size=1)

        mid = max(8, channels // reduction)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        mask = self.mask_conv(x).view(b, 1, h * w)       # (B,1,HW)
        mask = torch.softmax(mask, dim=2)                # (B,1,HW)

        x_flat = x.view(b, c, h * w)                     # (B,C,HW)
        context = torch.bmm(x_flat, mask.transpose(1, 2))  # (B,C,1)

        context = context.view(b, c, 1, 1)   

        out = self.transform(context)
        return x + out

class C2GC(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.blocks = nn.ModuleList()

        if n >= 1:
            self.blocks.append(GlobalContext(self.c, reduction))

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

    print("\n--- Global Context ---")
    gc = GlobalContext(256, reduction=16)
    y = gc(x)
    print("GC:", x.shape, "->", y.shape)

    print("\n--- C2GC ---")
    c2gc = C2GC(256, 256, n=1)
    y2 = c2gc(x)
    print("C2GC:", x.shape, "->", y2.shape)
