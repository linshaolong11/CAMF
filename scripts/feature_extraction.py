import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# 1D Transformer for SMILES sequences
class transformer_1d(nn.Module):
    def __init__(self):
        super(transformer_1d, self).__init__()
        self.embedding = nn.Embedding(2587, 128)  # 2586 + 1 for mask
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=8, dim_feedforward=512)
    
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.transformer(x, mask)
        return x

# 2D GNN for molecular graphs
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
    
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, global_mean_pool(x, batch=None)

# 3D SchNet for molecular spatial structures
class SchNet(nn.Module):
    def __init__(self):
        super(SchNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
    
    def forward(self, x, pos, batch):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, global_mean_pool(x, batch)

# Transformer Encoder for 2D and 3D features
class Transformer_E(nn.Module):
    def __init__(self):
        super(Transformer_E, self).__init__()
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=256)
    
    def forward(self, h_node, mask):
        h_node = self.transformer(h_node, mask)
        return h_node

# Projection Head for Feature Extraction
class projection_head(nn.Module):
    def __init__(self):
        super(projection_head, self).__init__()
        self.fc1 = nn.Linear(128, 500)  # 降维到500维
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

# Feature Extraction Model
class FeatureExtractionModel(nn.Module):
    def __init__(self):
        super(FeatureExtractionModel, self).__init__()
        self.model_1d = transformer_1d()
        self.model_2d = GNN()
        self.model_3d = SchNet()
        self.transformer_2d = Transformer_E()
        self.transformer_3d = Transformer_E()
        self.projection_1d = projection_head()
        self.projection_2d = projection_head()
        self.projection_3d = projection_head()
    
    def forward(self, batch):
        # 1D Feature Extraction
        smiles_emb, smi_mask = batch.smiles, batch.mask
        emb_1d = self.model_1d(smiles_emb, smi_mask)
        emb_1d = global_mean_pool(emb_1d, batch=None)
        emb_1d = self.projection_1d(emb_1d)
        
        # 2D Feature Extraction
        emb_2d, emb_2d_pool = self.model_2d(batch.x, batch.edge_index, batch.edge_attr)
        emb_2d = self.transformer_2d(emb_2d, None)
        emb_2d = global_mean_pool(emb_2d, batch.batch)
        emb_2d = self.projection_2d(emb_2d)
        
        # 3D Feature Extraction
        emb_3d, emb_3d_pool = self.model_3d(batch.x[:, 0], batch.pos, batch.batch)
        emb_3d = self.transformer_3d(emb_3d, None)
        emb_3d = global_mean_pool(emb_3d, batch.batch)
        emb_3d = self.projection_3d(emb_3d)
        
        return emb_1d, emb_2d, emb_3d

# 特征提取函数
def extract_molecular_features(smiles_list):
    """从SMILES列表中提取分子特征"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractionModel().to(device)
    model.eval()
    
    features = []
    
    for smiles in smiles_list:
        # 将SMILES转换为分子图
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            features.append(np.zeros(500))  # 无效SMILES返回全零特征
            continue
        
        # 生成2D分子图特征
        x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float32)
        edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([[bond.GetBondTypeAsDouble()] for bond in mol.GetBonds()], dtype=torch.float32)
        
        # 生成3D分子坐标
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        
        # 创建PyG数据对象
        batch = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, smiles=torch.tensor([0]), mask=torch.ones(1, 1))
        batch = batch.to(device)
        
        # 提取特征
        with torch.no_grad():
            emb_1d, emb_2d, emb_3d = model(batch)
            feature = torch.cat([emb_1d, emb_2d, emb_3d], dim=1).cpu().numpy()  # 拼接1D、2D和3D特征
        
        features.append(feature.flatten())  # 展平为1D数组
    
    return np.array(features)
