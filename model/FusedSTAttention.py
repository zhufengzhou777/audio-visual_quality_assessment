import torch
import torch.nn as nn

from model.vit_feature_extractor import DropPath, MLP


class FusedSTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, T, N, C)
        qkv2 = qkv.permute(0, 1, 3, 4, 2, 5)  # 3BTHNC -> 3BHNTC
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale  # transpose
        attn2 = self.attn_drop(attn2.softmax(dim=-1))
        x2 = attn2 @ v2
        x2 = x2.permute(0, 3, 2, 1, 4)
        x2 = x2.reshape(B, T, N, C)
        x = self.proj((x + x2) * 0.5)
        x = self.proj_drop(x)
        return x


class FusedSTBlock(nn.Module):
    def __init__(self, num_heads=8, dim=768, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = self.attn = FusedSTAttention(dim=dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
