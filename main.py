import argparse
import time
from typing import List, Tuple, Dict

import numpy as np

import psutil
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.utils
import torch_geometric.transforms

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import models
from models import NormProp


import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Choose dataset.')
parser.add_argument('--seed', type=int, default=42, help='Set random seed.')
parser.add_argument('--lr', type=float, default=0.00075, help='Learning rate.')
parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay (L2).')
parser.add_argument('--nhid', type=int, default=32, help='Dim of hidden layer.')
parser.add_argument('--epochs', type=int, default=500, help='Epochs.')
parser.add_argument('--path', type=str, default='./datasets/pyg/', help='Path of dataset cache')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout.')
parser.add_argument('--polars-path', type=str, default='./polars', help='The path of polars\'s numpy file.')
parser.add_argument('--mu', type=float, default=0.5, help='Weight of L2 norm loss.')
parser.add_argument('--ct', type=float, default=0.8, help='Weight of contrastive loss.')
parser.add_argument('--temperature', type=float, default=0.5, help='Temperature parameter for contrastive loss.')
parser.add_argument('--conf-threshold', type=float, default=0.92, help='Threshold of confidence.')
parser.add_argument('--batch-size', type=int, default=4096, help='Batch size.')
parser.add_argument('--warmup', type=int, default=10, help='Warmup epoch.')
parser.add_argument('--split', type=int, default=0, help='Train/Val/Test split id.')
parser.add_argument('--K', type=int, default=2, help='Aggregate K-hop neighborhoods.')
args = parser.parse_args()


# settings of training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_geometric.seed_everything(args.seed)

# load  dataset
dataset_package = utils.load_data(args.dataset, args.path)
dataset_name = dataset_package["name"]

if dataset_name == "wikics":
    num_folds = dataset_package["train_mask"].shape[1]
    all_acc = []
    for fold_idx in range(num_folds):
        train_mask = dataset_package["train_mask"][:, fold_idx].bool().to(device)
        val_mask = dataset_package["val_mask"][:, fold_idx].bool().to(device)
        test_mask = dataset_package["test_mask"].bool().to(device)
else:
    train_mask = dataset_package["train_mask"].bool().to(device)
    val_mask = dataset_package["val_mask"].bool().to(device)
    test_mask = dataset_package["test_mask"].bool().to(device)


x: torch.Tensor             = dataset_package['x']
y: torch.Tensor             = dataset_package['y']
edge_index: torch.Tensor    = dataset_package['edge_index']
edge_index, _               = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, num_nodes=x.shape[0])
# train_mask: torch.Tensor    = dataset_package['train_mask']
# val_mask: torch.Tensor      = dataset_package['val_mask']
# test_mask: torch.Tensor     = dataset_package['test_mask']
num_features: int           = dataset_package['num_features']
num_labels: int             = dataset_package['num_classes']

# _, edge_weight              = utils.random_walk_norm(edge_index, x.shape[0])
_, edge_weight              = utils.gcn_conv_norm(edge_index, x.shape[0])


polars                      = np.load(f"{args.polars_path}/polars-{num_labels}-{args.nhid}.npy")
assert polars.shape == (num_labels, args.nhid), "原型形状错误"
polars                      = torch.tensor(polars).to(device)


# settings of model
model = NormProp(num_features, args.nhid, num_labels, polars, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
norm_loss_flag: bool = False


# TODO: solve upper bound of Euclidean Norm
upper_bound_list: List[torch.Tensor] = list()
norm: torch.Tensor = torch.ones(x.shape[0], 1, device=x.device)
upper_bound_list.append(norm.view(-1))
for _ in range(model.K):
    norm = model.propagation(norm, edge_index, edge_weight)
    upper_bound_list.append(norm.view(-1))


# TODO: detect remote nodes
# from bias_utils import detect_remote_nodes
# distance_score: torch.Tensor = detect_remote_nodes(edge_index, edge_weight, train_mask, model.propagation, args.K)
# remote_mask: torch.Tensor = distance_score <= 0
# print(f"Remote nodes: {remote_mask.sum().item()}, min = {distance_score.min().item()}, max = {distance_score.max().item()}")


def random_sample(h: torch.Tensor, size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    random_perm: torch.Tensor = torch.randperm(h.shape[0], device=h.device)
    mask_idx: torch.Tensor = random_perm[: size]
    return h[mask_idx], mask_idx

# def plot_tsne(features, labels):
#     import pandas as pd
#     tsne = TSNE(n_components=2, init='pca', random_state=42)
#     import seaborn as sns
#     class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
#     tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
#     print('tsne_features的shape:', tsne_features.shape)
#     plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
#     df = pd.DataFrame()
#     df["y"] = labels
#     df["comp1"] = tsne_features[:, 0]
#     df["comp2"] = tsne_features[:, 1]
#     sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
#                     palette=sns.color_palette("hls", class_num),
#                     data=df)
#     plt.xlabel('t-SNE Dim-1', font={'family': 'Times New Roman', 'size': 12})
#     plt.ylabel('t-SNE Dim-2', font={'family': 'Times New Roman', 'size': 12})
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.show()

def plot_tsne(features, labels):
    import pandas as pd
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]
    
    sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df)
    
    # Set bold fonts
    plt.xlabel('t-SNE Dim-1', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
    plt.ylabel('t-SNE Dim-2', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
    
    # Set bold ticks
    plt.xticks(fontsize=18, weight='bold')
    plt.yticks(fontsize=18, weight='bold')
    
    # Set legend font to bold if needed
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(16)
    
    plt.show()


# def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature=0.3):
#     """ InfoNCE 对比损失 """
#     z1 = F.normalize(z1, p=2, dim=1)
#     z2 = F.normalize(z2, p=2, dim=1)

#     N = z1.size(0)
#     sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / temperature

#     pos_sim = torch.diag(sim_matrix)
#     logits = sim_matrix - torch.log(torch.exp(sim_matrix).sum(dim=1, keepdim=True))

#     loss = -pos_sim.mean() + logits.logsumexp(dim=1).mean()
#     return loss

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature=0.5):
    """ InfoNCE 对比损失 """
    # 特征归一化 (L2归一化)
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    
    batch_size = z1.size(0)
    
    # 计算相似度矩阵 (余弦相似度 / 温度系数)
    sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / temperature
    
    # 正样本为对角线元素
    pos_sim = torch.diag(sim_matrix)
    
    # 计算 logits（使用数值稳定的 logsumexp）
    logits = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    
    # 损失计算：交叉熵损失 (等价于 -pos_sim + logsumexp)
    loss = -pos_sim.mean() + logits.logsumexp(dim=1).mean()
    return loss

def train():
    model.train()
    optimizer.zero_grad()

    # 生成两个增强视图
    aug_x1, aug_edge1, aug_weight1 = utils.generate_augmented_view(x, edge_index)
    aug_x2, aug_edge2, aug_weight2 = utils.generate_augmented_view(x, edge_index)

    # 前向传播
    embedding, contrast_emb1 = model(aug_x1, aug_edge1, aug_weight1)
    _, contrast_emb2 = model(aug_x2, aug_edge2, aug_weight2)

    # embedding = model(x, edge_index, edge_weight)

    out = model.classification_based_on_cosine(embedding[-1])
    confidence = torch.max(out, dim=-1)[0]
    train_out = torch.gather(out[train_mask], dim=-1, index=y[train_mask].view(-1, 1))

    # 对比损失
    # contrast_loss = contrastive_loss(contrast_emb1, contrast_emb2)
    contrast_loss = contrastive_loss(contrast_emb1, contrast_emb2, temperature=args.temperature)

    embedding_norms = list()
    for emb in embedding:
        emb_norm = (emb * emb).sum(dim=-1).sqrt()
        embedding_norms.append(emb_norm)

    conf_mask = confidence > args.conf_threshold
    # conf_mask = conf_mask & remote_mask
    conf_mask = conf_mask & (~train_mask)
    conf_mask = conf_mask & (~val_mask)

    global norm_loss_flag
    if norm_loss_flag:
        if conf_mask.sum().item() > 0:
            norm_loss = embedding_norms[-1] / upper_bound_list[-1]
            # filter using confidence, e.g. max similarity
            norm_loss = norm_loss[conf_mask]
            # random sample
            random_idx: torch.Tensor = torch.randperm(norm_loss.shape[0], device=x.device)[ : args.batch_size]
            norm_loss = 1 - norm_loss[random_idx].mean()
            # print(f"debug: {confidence_mask.sum().item()}")
        else:
            norm_loss = 0
    else:
        norm_loss = 0
        conf_loss = 0

    cos_diff = 1 - train_out
    # loss = (cos_diff * cos_diff).mean() + (norm_loss * norm_loss) * args.mu + 0.2 * contrast_loss
    loss = (cos_diff).mean() + (norm_loss) * args.mu + args.ct * contrast_loss

    # 消融实验
    # loss = (cos_diff).mean()
    # loss = (cos_diff).mean() + 0.2 * contrast_loss
    # loss = (cos_diff).mean() + (norm_loss) * args.mu

    loss.backward()

    optimizer.step()
    return float(loss), model



@torch.no_grad()
def test(model):
    model.eval()
    embedding_list, _ = model(x, edge_index, edge_weight)
    predictions: List[torch.Tensor] = list()
    confidence: List[torch.Tensor] = list()
    norms: List[torch.Tensor] = list()

    for emb in range(len(embedding_list)):

        # 如果 emb 不是 torch.Tensor 类型，转换为张量
        if not isinstance(embedding_list[emb], torch.Tensor):
            embedding_list[emb] = torch.tensor(embedding_list[emb], dtype=torch.float32).to(device)

        # 调用 classification_based_on_cosine
        distance_of_polars: torch.Tensor = model.classification_based_on_cosine(embedding_list[emb])

        # 获取预测和置信度
        pred_dist, pred = torch.max(distance_of_polars, dim=-1)
        norm: torch.Tensor = (embedding_list[emb] * embedding_list[emb]).sum(dim=-1).sqrt()
        predictions.append(pred)
        confidence.append(pred_dist)
        norms.append(norm)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        accs.append(int((pred[mask] == y[mask]).sum()) / mask.sum().item())
    return accs, embedding_list[-1].detach().cpu().numpy()

def get_memory_usage():
    """返回当前进程的内存占用（单位：MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024**2  # 转换为 MB

# 在模型定义后添加参数量统计函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    best_val_acc = test_acc = 0
    best_embedding = None
    best_labels = None
    total_time = 0  # 总耗时统计
    peak_memory = 0  # 记录峰值内存

    # 添加参数量统计
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    max_params = total_params  # 初始化最大参数量

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()  # 记录 epoch 开始时间
        initial_mem = get_memory_usage()  # 记录初始内存

        if epoch > args.warmup:
            norm_loss_flag = True

        loss, model = train()
        (train_acc, val_acc, tmp_test_acc), current_embedding  = test(model)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_embedding = current_embedding
            best_labels = y.detach().cpu().numpy()
        elif val_acc == best_val_acc and tmp_test_acc > test_acc:
            test_acc = tmp_test_acc
        
        epoch_time = time.time() - epoch_start  # 计算 epoch 耗时
        total_time += epoch_time  # 累加总时间

        current_mem = get_memory_usage()
        epoch_memory = current_mem - initial_mem  # 本 epoch 内存增量
        peak_memory = max(peak_memory, current_mem)  # 更新峰值内存

        # 检查当前参数量
        current_params = count_parameters(model)
        max_params = max(max_params, current_params)
        

        print(f"---Epoch {epoch}, time = {epoch_time:.2f}s, Mem = {epoch_memory:.2f}MB (Peak: {peak_memory:.2f}MB), loss = {loss:.5f}, train-acc = {train_acc:.3f}, val = {val_acc:.3f}, test = {tmp_test_acc:.3f}")

    print(f"\nTotal training time = {total_time:.2f}s")
    print(f"Peak Memory Usage = {peak_memory:.2f}MB")
    print(f"Maximum Parameters: {max_params:,}")  # 打印最大参数量
    print(f"\nBest val test = {best_val_acc:.3f}\nTest acc = {test_acc:.4f}")

    print()
    print(test_acc)

    # 训练结束后可视化最佳模型的嵌入
    if best_embedding is not None and best_labels is not None:
        print("\nGenerating t-SNE visualization...")
        
        # 可视化所有节点
        plot_tsne(best_embedding, best_labels)
        
    #     # # 仅可视化测试集节点
    #     # test_mask_np = test_mask.cpu().numpy()
    #     # plot_tsne(best_embedding[test_mask_np], best_labels[test_mask_np])

