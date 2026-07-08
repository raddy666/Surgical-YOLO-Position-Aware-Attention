import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module
    Paper: ICML 2021
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # Mean of each channel
        mu = x.mean(dim=[2, 3], keepdim=True)

        # Variance-like term
        x_mu2 = (x - mu) ** 2
        n = H * W - 1
        var = x_mu2.sum(dim=[2, 3], keepdim=True) / n

        # Energy function
        e = x_mu2 / (4 * (var + self.e_lambda)) + 0.5

        return x * self.act(e)


class C2SimAM(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, e_lambda=1e-4):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n)*self.c, c2, 1)

        self.blocks = nn.ModuleList()

        if n >= 1:
            self.blocks.append(SimAM(e_lambda=e_lambda))

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

    print("\n--- SimAM ---")
    sim = SimAM()
    y = sim(x)
    print("SimAM:", x.shape, "->", y.shape)
    print("Params:", sum(p.numel() for p in sim.parameters()), "(should be 0)")

    print("\n--- C2SimAM ---")
    c2sim = C2SimAM(256, 256, n=1)
    y2 = c2sim(x)
    print("C2SimAM:", x.shape, "->", y2.shape)
    print("Params:", sum(p.numel() for p in c2sim.parameters()))
