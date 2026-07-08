import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

# Basic utility layers
class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ZPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)


class TripletAttention(nn.Module):
    """
    Triplet Attention
    Paper: Rotate to Attend — Triplet Attention Module
    """
    def __init__(self, no_spatial=False):
        super().__init__()
        self.no_spatial = no_spatial

        self.cw = AttentionGate()
        self.hc = AttentionGate()
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        #rotate for (C,W)
        x1 = x.permute(0, 2, 1, 3)
        out1 = self.cw(x1).permute(0, 2, 1, 3)

        #rotate for (C,H)
        x2 = x.permute(0, 3, 2, 1)
        out2 = self.hc(x2).permute(0, 3, 2, 1)

        #spatial (H,W)
        if not self.no_spatial:
            out3 = self.hw(x)
            return (out1 + out2 + out3) / 3
        
        # If spatial disabled
        return (out1 + out2) / 2


class AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        k = 7
        self.pool = ZPool()
        self.conv = BasicConv(2, 1, k, p=k//2, relu=False)

    def forward(self, x):
        attn = torch.sigmoid(self.conv(self.pool(x)))
        return x * attn

# C2f Wrapper
class C2Triplet(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, no_spatial=False):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)

        # identical to MSCA C2 wrapper
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n)*self.c, c2, 1)

        self.blocks = nn.ModuleList()

        # first block
        if n >= 1:
            self.blocks.append(TripletAttention(no_spatial=no_spatial))

        # remaining blocks
        for _ in range(max(0, n-1)):
            self.blocks.append(Conv(self.c, self.c, 3, 1))

        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        for blk in self.blocks:
            y.append(blk(y[-1]))

        out = self.cv2(torch.cat(y, 1))
        return x + out if self.shortcut else out


# Test
if __name__ == "__main__":
    x = torch.randn(2, 256, 32, 32)

    print("\n--- Triplet Attention ---")
    ta = TripletAttention()
    y = ta(x)
    print("Triplet:", x.shape, "→", y.shape)

    print("\n--- C2Triplet ---")
    c2ta = C2Triplet(256, 256, n=1)
    y2 = c2ta(x)
    print("C2Triplet:", x.shape, "→", y2.shape)
