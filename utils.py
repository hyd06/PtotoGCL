import random
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.datasets import Planetoid, Amazon, WikiCS, Coauthor
import torch_geometric.transforms as T




def load_data(dataset_str: str, path: str, n_shot: int = 10, device="cuda:0") -> Dict:
    '''
        :param dataset_str:
        :param path:
        :return: {x, edge_index, y, train_mask, val_mask, test_val, num_nodes, num_features, num_classes}
    '''

    dataset_str = dataset_str.lower()
    dataset = None

    # 加载 Planetoid 数据集（Cora/Citeseer/Pubmed）
    if dataset_str in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=path,
                            name=dataset_str.capitalize(),
                            transform=T.NormalizeFeatures())
        data = dataset[0].to(device)

        # # 核心修改：按类别采样n_shot个样本
        # train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
        # for cls in range(dataset.num_classes):
        #     idx = (data.y == cls).nonzero(as_tuple=True)[0]
        #     selected = idx[torch.randperm(len(idx))[:n_shot]]  # 随机选n_shot个
        #     print(f"Class {cls} selected {len(selected)} nodes")
        #     train_mask[selected] = True

        dataset_package = {
            "name": dataset_str,
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            # "train_mask": train_mask,
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        }

    # 加载 Amazon-Photo 数据集
    elif dataset_str == "photo":
        dataset = Amazon(root=path,
                         name='Photo',
                         transform=T.Compose([T.NormalizeFeatures(), T.ToDevice(device)]))
        data = dataset[0]

        # 划分数据集（如果没有预分割）
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        num_nodes = data.num_nodes
        train_num = int(num_nodes * train_ratio)
        val_num = int(num_nodes * (train_ratio + val_ratio))

        idx = np.arange(num_nodes )
        np.random.shuffle(idx)

        train_idx = torch.tensor(idx[:train_num]).long()
        val_idx = torch.tensor(idx[train_num:val_num]).long()
        test_idx = torch.tensor(idx[val_num:]).long()

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)

        dataset_package = {
            "name": "photo",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    # 加载 Amazon-Computers 数据集
    elif dataset_str == "computers":
        dataset = Amazon(root=path,
                         name='Computers',
                         transform=T.Compose([T.NormalizeFeatures(), T.ToDevice(device)]))
        data = dataset[0]

        # 划分数据集（同上）
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        num_nodes = data.num_nodes
        train_num = int(num_nodes * train_ratio)
        val_num = int(num_nodes * (train_ratio + val_ratio))

        idx = np.arange(num_nodes)
        np.random.shuffle(idx)

        train_idx = torch.tensor(idx[:train_num]).long()
        val_idx = torch.tensor(idx[train_num:val_num]).long()
        test_idx = torch.tensor(idx[val_num:]).long()

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)

        dataset_package = {
            "name": "computers",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }
        # 加载 coauthor-cs 数据集
    elif dataset_str == "cs":
        dataset = Coauthor(
            root=path,
            name='CS',
            transform=T.Compose([
                T.NormalizeFeatures(),
                T.ToDevice(device)
            ])
        )
        data = dataset[0]

        # 划分数据集
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        num_nodes = data.num_nodes
        train_num = int(num_nodes * train_ratio)
        val_num = int(num_nodes * (train_ratio + val_ratio))

        idx = np.arange(num_nodes)
        np.random.shuffle(idx)

        train_idx = torch.tensor(idx[:train_num]).long()
        val_idx = torch.tensor(idx[train_num:val_num]).long()
        test_idx = torch.tensor(idx[val_num:]).long()

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)

        dataset_package = {
            "name": "cs",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            "train_mask": train_mask.to(device),
            "val_mask": val_mask.to(device),
            "test_mask": test_mask.to(device),
            "num_nodes": num_nodes,
            "is_directed": False
        }
    elif dataset_str == "physics":
        dataset = Coauthor(root=path,
                         name='Physics',
                         transform=T.Compose([T.NormalizeFeatures(), T.ToDevice(device)]))
        data = dataset[0]

        # 划分数据集（同上）
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        num_nodes = data.num_nodes
        train_num = int(num_nodes * train_ratio)
        val_num = int(num_nodes * (train_ratio + val_ratio))

        idx = np.arange(num_nodes)
        np.random.shuffle(idx)

        train_idx = torch.tensor(idx[:train_num]).long()
        val_idx = torch.tensor(idx[train_num:val_num]).long()
        test_idx = torch.tensor(idx[val_num:]).long()

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)

        dataset_package = {
            "name": "physics",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    elif dataset_str == "wikics":
        dataset = WikiCS(
            root=path + '/WikiCS',
            transform=T.Compose([T.NormalizeFeatures(), T.ToDevice(device)])
        )
        data = dataset[0]

        num_folds = 20  # WikiCS 10 个 fold

        for fold_idx in range(num_folds):
            train_mask = data.train_mask[:, fold_idx].bool()  # [num_nodes]
            val_mask = data.val_mask[:, fold_idx].bool()
            test_mask = data.test_mask.bool()  # 固定测试集

        dataset_package = {
            "name": "wikics",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y.squeeze(),  # 修复维度问题
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            "train_mask": data.train_mask,  # [num_nodes, 20]
            "val_mask": data.val_mask,      # [num_nodes, 20]
            "test_mask": data.test_mask,          # 固定测试集
            "num_nodes": data.num_nodes,
            "is_directed": False
        }

        # 添加GCN归一化边权重
        edge_index, edge_weight = gcn_conv_norm(dataset_package["edge_index"], data.num_nodes)
        dataset_package.update({
            "edge_index": edge_index,
            "edge_weight": edge_weight
        })

    else:
        raise ValueError(
            f"未知数据集: {dataset_str}. 支持的数据集: cora, citeseer, pubmed, photo, computers,coauthor-cs")

    # 添加通用字段
    dataset_package.update({
        "num_nodes": data.num_nodes,
        "is_directed": data.is_directed
    })

    return dataset_package



def random_walk_norm(edge_index: torch.Tensor, num_nodes=None) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        edge_index: with self-loops
    '''
    edge_weight: torch.Tensor = torch.ones(edge_index.shape[1], device=edge_index.device)
    edge_weight = pyg_utils.softmax(src=edge_weight, index=edge_index[1])
    return edge_index, edge_weight



def gcn_conv_norm(edge_index: torch.Tensor, num_nodes=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not pyg_utils.contains_self_loops(edge_index=edge_index):
        edge_index, _ = pyg_utils.add_remaining_self_loops(edge_index)

    edge_index, edge_weight = pyg_nn.conv.gcn_conv.gcn_norm(edge_index)
    return edge_index, edge_weight


def generate_augmented_view(x: torch.Tensor, edge_index: torch.Tensor,
                            feat_drop_rate=0.2, edge_drop_rate=0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 生成增强视图并返回归一化的边权重 """
    # 特征增强
    aug_x = x.clone()
    mask = torch.rand(x.size(1), device=x.device) < feat_drop_rate
    aug_x[:, mask] = 0

    # 边增强（确保保留自环边）
    aug_edge_index, _ = pyg_utils.dropout_adj(
        edge_index,
        p=edge_drop_rate,
        num_nodes=x.size(0),  # 关键修复：指定节点数
        force_undirected=True  # 避免边方向错误
    )

    # 重新计算 GCN 归一化的边权重
    aug_edge_index, aug_edge_weight = gcn_conv_norm(aug_edge_index, num_nodes=x.size(0))

    return aug_x, aug_edge_index, aug_edge_weight
