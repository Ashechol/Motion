import torch
from torch import nn
from model.modules.graph import GCN


class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C) instead of
    (B * T, J, C)
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qkv_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qkv_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)


class VelocityCrossAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.act = nn.GELU()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.kv = nn.Linear(dim_in, dim_in * 2, bias=qkv_bias)
        self.q = nn.Linear(dim_in, dim_in, bias=qkv_bias)

        self.conv1 = nn.Conv2d(dim_in, dim_in * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim_in * 2, dim_in, kernel_size=1)

        self.proj_drop = nn.Dropout(proj_drop)

    def get_velocity(self, x):
        b, t, j, c = x.shape

        x = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        u = torch.cat((x, x[:, 0].reshape(b, 1, j, -1)), dim=1)  # B T+1 J C
        v = torch.cat((x[:, 0].reshape(b, 1, j, -1), x), dim=1)

        # B T J C
        # u = (u - v)[:, 0:-1].transpose(1, 2)  # B J T C
        u = (u - v)[:, 0:-1].permute(0, 3, 1, 2)  # B C T J

        u = self.conv2(self.act(u))

        # return self.conv(uv).permute(0, 2, 1).reshape(b, j, t, c)
        return u.permute(0, 3, 2, 1)  # B T J C

    def forward(self, x):
        b, t, j, c = x.shape

        # (2, B, H, T, J, C)
        kv = self.kv(x).reshape(b, t, j, 2, self.num_heads, c // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        # (B, H, J, T, C)
        k, v = kv[0].transpose(2, 3), kv[1].transpose(2, 3)

        # (B, H, J, T, C)
        q = self.q(self.get_velocity(x)).reshape(b, j, t, self.num_heads, c // self.num_heads).permute(0, 3, 1, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(b, t, j, c)  # (B, T, J, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SGAttention(nn.Module):
    def __init__(self, dim_in, dim_out, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qkv_scale=None):
        super().__init__()

        self.spatial = Attention(dim_in, dim_out, num_heads, qkv_bias, qkv_scale, mode='spatial')
        self.graph = GCN(dim_in, dim_out, 17, mode='spatial')

        self.fusion = nn.Linear(dim_out * 2, 2)
        self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def adaptive_fusion(self, a, b):
        alpha = torch.cat((a, b), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)

        return a * alpha[..., 0:1] + b * alpha[..., 1:2]

    def forward(self, x):
        # B T J C

        return self.adaptive_fusion(self.spatial(x), self.graph(x))


class TVCAttention(nn.Module):
    """
    Temporal-Velocity Cross Attention
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qkv_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.temporal = Attention(dim_in, dim_out, num_heads, qkv_bias, qkv_scale, attn_drop, proj_drop, mode='temporal')
        self.velocity = VelocityCrossAttention(dim_in, dim_out, num_heads, qkv_bias, qkv_scale, attn_drop, proj_drop)

        self.fusion = nn.Linear(dim_out * 2, 2)
        self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def adaptive_fusion(self, a, b):
        alpha = torch.cat((a, b), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)

        return a * alpha[..., 0:1] + b * alpha[..., 1:2]

    def forward(self, x):
        # B T J C

        return self.adaptive_fusion(self.temporal(x), self.velocity(x))


if __name__ == '__main__':
    from thop import profile
    model = VelocityCrossAttention(128, 128)
    X = torch.randn(1, 243, 17, 128)

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()

    macs, _ = profile(model, inputs=(X,))

    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {macs:,}")
