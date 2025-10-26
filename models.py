from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils



class Encoder_fc(nn.Module):
    '''
        Linear encoder
    '''
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, out_channels)

        self.act_fn = F.leaky_relu
        # self.act_fn = F.relu
        # self.layer_norm = nn.LayerNorm(hidden_channels, eps=1e-6)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        self.fc.reset_parameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        return x



class Encoder(nn.Module):
    '''
        MLP encoder
    '''
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

        # self.act_fn = F.leaky_relu
        self.act_fn = F.relu
        self.layer_norm = nn.LayerNorm(hidden_channels, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



# Message-Passing (GCN: $D^{-1/2} A D^{-1/2}$)
class GCNAggr(pyg_nn.MessagePassing):
    '''
        source_2_target
    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        # add self-loops
        if not pyg_utils.contains_self_loops(edge_index):
            # edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index, edge_weight)
            edge_index, _ = pyg_utils.add_remaining_self_loops(edge_index)

        # calculate edge_weight
        if edge_weight is None:
            edge_index, edge_weight = pyg_nn.conv.gcn_conv.gcn_norm(edge_index)

        x = self.propagate(edge_index, edge_weight=edge_weight, x=x, size=None)

        return x

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return pyg_utils.spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

# 修改后的 NormProp 类
class NormProp(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, n_prototype: int,
                 polars: torch.Tensor, dropout: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.n_prototype = n_prototype
        self.drop_p = dropout
        self.K = 2
        # 确保原型是二维浮点张量
        assert isinstance(polars, torch.Tensor), "polars 必须是张量"
        self.prototypes = nn.Parameter(polars.clone().float())
        assert self.prototypes.dim() == 2, "原型维度错误"

        # 原有编码器和传播层
        self.encoder = Encoder(in_channels, 32, embedding_dim, dropout=dropout)
        self.propagation = GCNAggr()

        # 对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(embedding_dim, 256),  # 投影到对比空间
            nn.ELU(),
            nn.Linear(256, 128)
        )
        # self._initparameters()

    # def _initparameters(self) -> None:
    #     # 初始化编码器和传播层的参数
    #     # axivr初始化
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        h0 = self.encoder(x)
        h0 = F.normalize(h0, p=2, dim=-1)
        hs = [h0]
        for _ in range(self.K):
            h = self.propagation(hs[-1], edge_index, edge_weight)
            hs.append(h)

        # 返回多层嵌入和对比投影
        contrast_emb = self.contrast_proj(hs[-1])  # 使用最后一层
        return hs, F.normalize(contrast_emb, dim=-1)  # L2归一化


    def classification_based_on_cosine(self, h: torch.Tensor) -> torch.Tensor:
        # assert h.shape[1] == self.prototypes.shape[1]
        assert isinstance(h, torch.Tensor), "输入 h 必须是张量"
        assert h.dim() == 2, "输入 h 必须是二维张量"
        assert self.prototypes.shape[1] == h.shape[1], f"维度不匹配: 原型={self.prototypes.shape}, h={h.shape}"
        h = F.normalize(h, p=2, dim=-1)
        distance: torch.Tensor = h @ self.prototypes.T
        return distance
