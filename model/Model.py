from collections import OrderedDict

import torch
from torch import nn
from timm.models.layers import DropPath

from model.modules.attention import Attention, TVCAttention
from model.modules.graph import GCN
from model.modules.mlp import MLP


class FormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mixer_type='attention', n_frames=243):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop=drop, mode='spatial')
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim, num_nodes=17, mode='spatial')
        elif mixer_type == 'tvc':
            self.mixer = TVCAttention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MotionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, n_frames=243):
        super().__init__()

        self.att_spatial = FormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                       qk_scale, use_layer_scale, layer_scale_init_value,
                                       mixer_type='attention', n_frames=n_frames)

        self.graph_spatial = FormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                         qkv_bias, qk_scale, use_layer_scale, layer_scale_init_value,
                                         mixer_type="graph", n_frames=n_frames)

        self.att_tvc = FormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                   qk_scale, use_layer_scale, layer_scale_init_value,
                                   mixer_type='tvc', n_frames=n_frames)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
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
        """
        x: tensor with shape [B, T, J, C]
        """

        if self.use_adaptive_fusion:
            x = self.adaptive_fusion(self.att_spatial(x), self.graph_spatial(x))
        else:
            x = (self.att_spatial(x) + self.graph_spatial(x)) * 0.5

        x = self.att_tvc(x)

        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, n_frames=243):
    layers = []
    for _ in range(n_layers):
        layers.append(MotionBlock(dim=dim,
                                  mlp_ratio=mlp_ratio,
                                  act_layer=act_layer,
                                  attn_drop=attn_drop,
                                  drop=drop_rate,
                                  drop_path=drop_path_rate,
                                  num_heads=num_heads,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  qkv_bias=qkv_bias,
                                  qk_scale=qkv_scale,
                                  use_adaptive_fusion=use_adaptive_fusion,
                                  n_frames=n_frames))
    layers = nn.Sequential(*layers)

    return layers


class Model(nn.Module):
    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, num_joints=17, n_frames=243):
        super().__init__()

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    n_frames=n_frames)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C]
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        x = self.joints_embed(x)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)

        return x


if __name__ == '__main__':
    import warnings
    from thop import profile

    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    random_x = torch.randn((b, t, j, c)).to('cuda')

    model = Model(n_layers=16, dim_in=3, dim_feat=128, mlp_ratio=4, n_frames=t, qkv_bias=True).to('cuda')

    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()

    macs, _ = profile(model, inputs=(random_x,))

    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {macs:,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)

    import time

    num_iterations = 100
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")

    out = model(random_x)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"
