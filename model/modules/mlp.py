import torch
import torch.nn as nn
from model.modules.graph import GCN
from timm.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):
        """
        :param channel_first: if True, during forward the tensor shape is [B, C, T, J] and fc layers are performed with
                              1x1 convolutions.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if channel_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MLPLN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )

        self.act = act_layer()

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features)
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class MLPGCN(nn.Module):
    def __init__(self, dim, hidden_features=None, act_layer=nn.GELU, n_frames=243, drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,):
        super().__init__()

        hidden_features = hidden_features or dim
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gcn_s = GCN(dim, dim,
                         num_nodes=17,
                         mode='spatial')

        self.gcn_t = GCN(dim, dim,
                         num_nodes=n_frames,
                         neighbour_num=2,
                         mode='temporal',
                         use_temporal_similarity=True,
                         temporal_connection_len=1)

        self.mlp = MLPLN(dim, hidden_features, act_layer=act_layer)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x_gcn_s = self.gcn_s(x)
        x_mlp = self.mlp(x)

        if self.use_layer_scale:
            x = res + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                                     * (x_gcn_s + x_mlp))
        else:
            x = res + self.drop_path(x_gcn_s + x_mlp)

        res = x
        x = self.norm1(x)
        x_gcn_t = self.gcn_t(x)
        x_mlp = self.mlp(x)

        if self.use_layer_scale:
            x = res + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                                     * (x_gcn_t + x_mlp))
        else:
            x = res + self.drop_path(x_gcn_t + x_mlp)

        return x