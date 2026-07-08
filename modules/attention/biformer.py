import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# Basic Layers for BiFormer
class MLP(nn.Module):
    def __init__(self, dim, expansion_factor=4, bias=False):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # shapes (B,C,H,W)

        # compute spatial similarity map (B,1,H,W)
        attn = (q * k).sum(dim=1, keepdim=True)  # (B,1,H,W)

        attn = attn.view(B, 1, -1)                # (B,1,H*W)
        attn = torch.softmax(attn, dim=-1)       # normalize over H*W
        attn = attn.view(B, 1, H, W)             # (B,1,H,W) back

        out = v * attn                            # (B,C,H,W) * (B,1,H,W) -> (B,C,H,W)
        return self.proj(out)



# BiFormer Block
class BiFormer(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=4, bias=False):
        super().__init__()
        self.attn = BiLevelRoutingAttention(dim, num_heads, bias)
        self.mlp = MLP(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

# C2 Wrapper 
class C2BiFormer(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5,
                 num_heads=8, ffn_expansion_factor=4, bias=False):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.blocks = nn.ModuleList()
        self.blocks.append(BiFormer(self.c, num_heads, ffn_expansion_factor, bias))

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
    print("\n--- BiFormer ---")
    bf = BiFormer(256, num_heads=4)
    y = bf(x)
    print("BiFormer:", x.shape, "->", y.shape)

    print("\n--- C2BiFormer ---")
    c2bf = C2BiFormer(256, 256, n=1, num_heads=4)
    y2 = c2bf(x)
    print("C2BiFormer:", x.shape, "->", y2.shape)
    print("C2BiFormer params:", sum(p.numel() for p in c2bf.parameters()))
