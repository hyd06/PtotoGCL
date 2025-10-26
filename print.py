import dgl
import torch

# 加载Cora数据集
dataset = dgl.data.CoraGraphDataset()
# 或者使用旧版本的名称
# dataset = dgl.data.CitationGraphDataset('cora')

# 获取图对象
g = dataset[0]

# 打印图的基本信息
print(g)

# 查看可用的节点特征和边特征
print("\n可用的节点特征:", g.ndata.keys())
print("可用的边特征:", g.edata.keys())

# 查看具体特征
print("\n节点特征 'feat':", g.ndata['feat'].shape)
print("节点标签 'label':", g.ndata['label'].shape)
print("训练集掩码 'train_mask':", g.ndata['train_mask'].shape)
print("验证集掩码 'val_mask':", g.ndata['val_mask'].shape)
print("测试集掩码 'test_mask':", g.ndata['test_mask'].shape)