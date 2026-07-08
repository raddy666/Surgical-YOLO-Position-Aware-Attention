import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.conv import CBAM

class C2CBAM(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, reduction=16, kernel_size=7):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        self.blocks = nn.ModuleList()
        if n >= 1:
            self.blocks.append(CBAM(self.c, kernel_size))
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
    x = torch.randn(2, 64, 32, 32)
    model = CBAM(c1=64, kernel_size=7)
    y = model(x)
    print("CBAM:", x.shape, "->", y.shape)

    model = C2CBAM(c1=64, c2=64, n=1, e=0.5)
    y = model(x)
    print("C2CBAM:", x.shape, "->", y.shape)