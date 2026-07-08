import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class MSCA(nn.Module):
    def __init__(self, dim, num_heads=4, scales=(3,5,7), attn_ratio=0.5, sr_ratio=1):
        super(MSCA, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = max(1, int(sr_ratio))

        self.scales = scales
        self.ms_gates = nn.Parameter(torch.zeros(len(scales)))
        self.ms_branches = nn.ModuleList([
            nn.Sequential(
                Conv(dim, dim, k=1, s=1),
                nn.Conv2d(dim, dim, k, padding=k//2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True)
            ) for k in scales
        ])
        self.ms_fuse = Conv(dim * len(scales), dim, k=1, s=1)

        attn_dim = max(32, int(dim * attn_ratio))
        attn_dim = (attn_dim // num_heads) * num_heads
        self.attn_dim = attn_dim
        self.qkv = nn.Linear(dim, 3 * attn_dim, bias=False)
        self.proj = nn.Linear(attn_dim, dim, bias=False)

        self.fusion_logits = nn.Parameter(torch.zeros(2, dim))

        self.norm_in = nn.BatchNorm2d(dim)
        self.norm_out = nn.BatchNorm2d(dim)

    def _apply_ms_frontend(self, x):
        gates = torch.sigmoid(self.ms_gates)
        feats = [branch(x) * gates[i] for i, branch in enumerate(self.ms_branches)]
        return self.ms_fuse(torch.cat(feats, dim=1))

    def _axial_attention(self, x, axis):
        B, C, H, W = x.shape
        if axis == 'h':
            seq_len, x_seq = H, x.permute(0, 3, 2, 1).reshape(B*W, H, C)
        else:
            seq_len, x_seq = W, x.permute(0, 2, 3, 1).reshape(B*H, W, C)

        qkv = self.qkv(x_seq).chunk(3, dim=-1)
        q, k, v = qkv

        if self.sr_ratio > 1 and seq_len >= self.sr_ratio:
            k = F.avg_pool1d(k.transpose(1,2), self.sr_ratio, self.sr_ratio).transpose(1,2)
            v = F.avg_pool1d(v.transpose(1,2), self.sr_ratio, self.sr_ratio).transpose(1,2)

        D = self.attn_dim // self.num_heads
        def split_heads(t):
            N, L, _ = t.shape
            return t.view(N, L, self.num_heads, D).permute(0,2,1,3)
        q, k, v = map(split_heads, (q,k,v))

        attn = (q @ k.transpose(-2,-1)) * (D**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0,2,1,3).reshape(-1, seq_len, self.attn_dim)
        out = self.proj(out)

        if axis == 'h':
            return out.view(B, W, H, C).permute(0,3,2,1)
        else:
            return out.view(B, H, W, C).permute(0,3,1,2)

    def forward(self, x):
        identity = x
        x = self.norm_in(x)
        x_ms = self._apply_ms_frontend(x)
        out_h = self._axial_attention(x_ms, 'h')
        out_w = self._axial_attention(x_ms, 'w')
        fusion = torch.softmax(self.fusion_logits, dim=0)
        out = fusion[0].view(1,-1,1,1)*out_h + fusion[1].view(1,-1,1,1)*out_w
        return identity + self.norm_out(out)


class C2MSCA(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, **msca_kwargs):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.blocks = nn.ModuleList()
        if n >= 1:
            self.blocks.append(MSCA(self.c, **msca_kwargs))
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
    msca = MSCA(dim=64, num_heads=4, scales=(3,5,7), attn_ratio=0.5, sr_ratio=2)
    y = msca(x)
    print("MSCA:", x.shape, "->", y.shape)

    c2msca = C2MSCA(c1=64, c2=128, n=2, shortcut=False, scales=(3,5,7), num_heads=4, attn_ratio=0.5, sr_ratio=2)
    y2 = c2msca(x)
    print("C2MSCA:", x.shape, "->", y2.shape)
