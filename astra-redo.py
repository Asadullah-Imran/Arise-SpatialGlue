# Cell 2: Imports & Environment Configuration
import os
import sys
import argparse
import copy
import random
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from typing import Optional, Tuple, Dict, List

import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Using compute engine: {device}")
if torch.cuda.is_available():
    print(f"[GPU] Model: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

def init_r_mclust():
    try:
        import rpy2.robjects as robjects
        robjects.r['options'](warn=-1)
        robjects.r.library("mclust")
        return True
    except Exception as e:
        print(f"[Warning] R mclust unavailable ({e}). Fallback to KMeans.")
        return False
# Cell 3: Universal Preprocessing Engine
def clr_normalize_each_cell(adata, inplace=True):
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x)) if len(x) > 0 else 1.0
        return np.log1p(x / (exp + 1e-12))

    if not inplace:
        adata = adata.copy()

    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X)
    return adata

def tfidf(X):
    idf = X.shape[0] / (X.sum(axis=0) + 1e-12)
    if sp.issparse(X):
        tf = X.multiply(1.0 / (X.sum(axis=1) + 1e-12))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-12)
        return tf * idf

def preprocess_universal(adata_RNA, adata_omics2, dataset_name, n_hvg=3000):
    sc.pp.filter_genes(adata_RNA, min_cells=10)
    try:
        sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=n_hvg)
    except Exception:
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
        sc.pp.highly_variable_genes(adata_RNA, flavor="seurat", n_top_genes=n_hvg)

    if 'log1p' not in adata_RNA.uns:
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
    sc.pp.scale(adata_RNA)

    RNA_expression = adata_RNA[:, adata_RNA.var['highly_variable']].X
    if sp.issparse(RNA_expression):
        RNA_expression = RNA_expression.toarray()

    # Second Modality Preprocessing
    if dataset_name.startswith("10x"):
        # ADT Protein Modality
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        ADT_expression = adata_omics2.X.toarray() if sp.issparse(adata_omics2.X) else np.array(adata_omics2.X)
    else:
        # ATAC Epigenomic Modality
        adata_omics2.X = tfidf(adata_omics2.X)
        sc.pp.normalize_per_cell(adata_omics2, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2)
        n_comps = min(50, adata_omics2.shape[1])
        pca_model = PCA(n_components=n_comps, random_state=42)
        X_atac = adata_omics2.X.toarray() if sp.issparse(adata_omics2.X) else adata_omics2.X
        ADT_expression = pca_model.fit_transform(X_atac)

    return RNA_expression, ADT_expression, adata_RNA, adata_omics2
# Cell 4: Dual-Graph Topology & Spatial Adjacency Construction
class DualGraphData(Data):
    def __init__(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                 dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight):
        super().__init__()
        self.x_RNA = x_RNA
        self.x_ADT = x_ADT
        self.sim_edge_index = sim_edge_index
        self.sim_edge_weight = sim_edge_weight
        self.dist_edge_index = dist_edge_index
        self.dist_edge_weight = dist_edge_weight
        self.common_edge_index = common_edge_index
        self.common_edge_weight = common_edge_weight

def build_dual_graph(RNA_expression, ADT_expression, cell_positions, device='cpu', num_neighbors=15):
    # 1. Cosine Feature Similarity KNN Graph
    similarity_matrix = cosine_similarity(RNA_expression)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(RNA_expression)
    _, indices = nbrs.kneighbors(RNA_expression)

    adjacency_matrix = np.zeros_like(similarity_matrix, dtype=int)
    for i in range(len(RNA_expression)):
        for j in indices[i][1:]:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

    sim_edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long).to(device)
    sim_edge_weight = torch.tensor(similarity_matrix[adjacency_matrix > 0], dtype=torch.float).to(device)

    # 2. Spatial Distance KNN Graph
    knn_graph = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='distance', include_self=False)
    knn_graph = knn_graph.maximum(knn_graph.T)
    dist_edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_graph.data, dtype=torch.float).to(device)

    # 3. Common Edge Pure Intersection (Consensus)
    sim_edges = set(zip(sim_edge_index[0].tolist(), sim_edge_index[1].tolist()))
    dist_edges = set(zip(dist_edge_index[0].tolist(), dist_edge_index[1].tolist()))
    common_edges = sim_edges.intersection(dist_edges)

    common_edge_index = torch.tensor(list(zip(*common_edges)), dtype=torch.long).to(device)
    common_edge_weight = torch.ones(common_edge_index.size(1), dtype=torch.float).to(device)

    x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
    x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

    return DualGraphData(
        x_RNA, x_ADT,
        sim_edge_index, sim_edge_weight,
        dist_edge_index, dist_edge_weight,
        common_edge_index, common_edge_weight
    )

def compute_spatial_adj_norm(dist_edge_index, num_nodes, device='cpu'):
    """Construct row-normalized spatial adjacency matrix for Potts MRF consensus."""
    adj = torch.sparse_coo_tensor(dist_edge_index, torch.ones(dist_edge_index.size(1), device=device), size=(num_nodes, num_nodes)).to_dense()
    adj = adj + torch.eye(num_nodes, device=device)
    deg = adj.sum(dim=1, keepdim=True)
    adj_norm = adj / (deg + 1e-12)
    return adj_norm
# Cell 5: Neural Architecture & Gated Attention Fusion Modules
class GatedAttentionFusion(nn.Module):
    """
    Spot-Adaptive Gated Attention Fusion Module.
    Computes a learned, per-spot, feature-wise attention gate g in (0, 1)^d:
        g = Sigmoid(W_g [z1 || z2] + b_g)
        z_fused = g ⊙ z1 + (1 - g) ⊙ z2
    Followed by an output projection layer to refine the fused manifold.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z1, z2], dim=1)
        gate = self.gate_net(combined)
        fused = gate * z1 + (1.0 - gate) * z2
        fused = self.out_proj(fused)
        return fused, gate

class SpatialPottsDEC(nn.Module):
    """
    Spatial Potts Markov Random Field Consensus Deep Embedding Clustering (DEC).
    Smooths target distribution P using spatial neighborhood consensus to suppress salt-and-pepper noise
    while preserving transcriptomic cluster boundaries.
    """
    def __init__(self, num_clusters: int, latent_dim: int = 64, alpha: float = 1.0, lambda_spatial: float = 0.15):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.lambda_spatial = lambda_spatial
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def compute_q(self, z: torch.Tensor) -> torch.Tensor:
        # Student's t-distribution soft assignment
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / (torch.sum(q, dim=1, keepdim=True) + 1e-12)
        return q

    def compute_spatial_target_p(self, q: torch.Tensor, spatial_adj_norm: torch.Tensor) -> torch.Tensor:
        # Standard DEC target distribution
        weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True) + 1e-12)
        p_dec = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-12)

        # Spatial Potts neighborhood consensus
        spatial_consensus = torch.mm(spatial_adj_norm, q)

        # Potts modulated target
        p_spatial = p_dec * torch.exp(self.lambda_spatial * spatial_consensus)
        p_final = p_spatial / (torch.sum(p_spatial, dim=1, keepdim=True) + 1e-12)
        return p_final

    def forward(self, z: torch.Tensor, p_target: Optional[torch.Tensor] = None):
        q = self.compute_q(z)
        if p_target is not None:
            kl_loss = F.kl_div(q.log(), p_target, reduction='batchmean')
        else:
            kl_loss = torch.tensor(0.0, device=z.device)
        return q, kl_loss

class DualGCN_Gated(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, q_dim, dropout=0.0):
        super(DualGCN_Gated, self).__init__()
        # RNA dual branches (similarity & distance)
        self.x_RNA1 = GCNConv(in_channels, hidden_channels)
        self.x_RNA2 = GCNConv(in_channels, hidden_channels)
        self.protein3 = GCNConv(q_dim, out_channels)

        self.sim_conv = GCNConv(hidden_channels, out_channels)
        self.dist_conv = GCNConv(hidden_channels, out_channels)

        # Spot-Adaptive Gated Attention Fusion Modules
        # Fusion 1: Within-Modality (RNA Sim vs. RNA Dist)
        self.fusion_layer1 = GatedAttentionFusion(dim=out_channels)
        # Fusion 2: Between-Modality (Fused RNA vs. Auxiliary ADT/ATAC)
        self.fusion_layer2 = GatedAttentionFusion(dim=out_channels)

        # Decoupled Multi-Task Decoders
        self.dropout = dropout
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)
        self.deconv4 = nn.Linear(hidden_channels, q_dim)
        self.deconv5 = nn.Linear(hidden_channels, in_channels + q_dim)

    def forward(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight):
        xs = F.relu(self.x_RNA1(x_RNA, sim_edge_index, sim_edge_weight))
        xs = F.dropout(xs, self.dropout, training=self.training)

        xd = F.relu(self.x_RNA2(x_RNA, dist_edge_index, dist_edge_weight))
        xd = F.dropout(xd, self.dropout, training=self.training)

        x_sim = self.sim_conv(xs, sim_edge_index, sim_edge_weight)
        x_dist = self.dist_conv(xd, dist_edge_index, dist_edge_weight)
        pro = self.protein3(x_ADT, common_edge_index, common_edge_weight)

        # Dynamic Gated Fusion
        fused, gate_spatial = self.fusion_layer1(x_sim, x_dist)
        fused_pro, gate_modality = self.fusion_layer2(fused, pro)
        return x_sim, x_dist, fused, fused_pro, pro, gate_spatial, gate_modality

    def reconstruct_joint(self, z):
        return self.deconv5(F.relu(self.deconv1(z)))

    def reconstruct_rna(self, z):
        return self.deconv2(F.relu(self.deconv1(z)))

    def reconstruct_aux(self, z):
        return self.deconv4(F.relu(self.deconv1(z)))

class ASTRA_v1_DEC_Gated(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, q_dim, num_clusters,
                 beta=25.0, gamma=10.0, delta=1.0, lambda_dec=1.0, lambda_spatial=0.15, dropout=0.0):
        super(ASTRA_v1_DEC_Gated, self).__init__()
        self.gcn = DualGCN_Gated(in_channels, hidden_channels, out_channels, q_dim, dropout=dropout)
        self.dec = SpatialPottsDEC(num_clusters=num_clusters, latent_dim=out_channels, lambda_spatial=lambda_spatial)
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_dec = lambda_dec
        self.l1_lambda = 1e-4
        self.l2_lambda = 1e-3

    def init_cluster_centers(self, centers_numpy):
        """Initialize DEC prototype centers using KMeans in aligned latent coordinates."""
        self.dec.cluster_centers.data.copy_(torch.tensor(centers_numpy, dtype=torch.float32, device=self.dec.cluster_centers.device))

    def forward(self, data, compute_dec=False, p_target=None):
        x_sim, x_dist, fused, fused_pro, pro, gate_sp, gate_mod = self.gcn(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index, data.sim_edge_weight,
            data.dist_edge_index, data.dist_edge_weight,
            data.common_edge_index, data.common_edge_weight
        )
        if compute_dec:
            q, kl_loss = self.dec(fused_pro, p_target)
            return x_sim, x_dist, fused, fused_pro, pro, q, kl_loss, gate_sp, gate_mod
        return x_sim, x_dist, fused, fused_pro, pro, gate_sp, gate_mod

    def cosine_similarity(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
        mat = torch.div(mat, torch.matmul(norm, norm.T) + 1e-12)
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat - torch.diag_embed(torch.diag(mat))
        return mat

    def spatial_regularization_loss(self, emb, dist_edge_index, dist_edge_weight):
        num_nodes = emb.size(0)
        graph_nei = torch.sparse_coo_tensor(
            dist_edge_index,
            torch.ones_like(dist_edge_weight),
            size=(num_nodes, num_nodes)
        ).to_dense()
        graph_neg = 1.0 - graph_nei
        sim_mat = torch.sigmoid(self.cosine_similarity(emb))
        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1.0 - sim_mat + 1e-10)).mean()
        return -(neigh_loss + neg_loss) / 2.0

    def compute_losses(self, data, sim_z, dist_z, fused_z, fused_pro, pro, kl_loss=None, stage=1):
        combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)
        l_rec = F.mse_loss(combined_raw, self.gcn.reconstruct_joint(fused_pro))
        l_sim = F.mse_loss(data.x_RNA, self.gcn.reconstruct_rna(sim_z))
        l_dist = F.mse_loss(data.x_RNA, self.gcn.reconstruct_rna(dist_z))
        l_aux = F.mse_loss(data.x_ADT, self.gcn.reconstruct_aux(pro))

        l_spatial = self.spatial_regularization_loss(fused_z, data.dist_edge_index, data.dist_edge_weight)

        l1_reg = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2_reg = sum(torch.sum(p ** 2) for p in self.parameters())
        reg_loss = self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        # In Stage 2, moderate reconstruction weight so DEC KL divergence actively sharpens clusters
        beta_curr = self.beta if stage == 1 else (self.beta * 0.2)
        rec_term = beta_curr * (l_rec + l_sim + l_dist + l_aux)
        spatial_term = self.gamma * l_spatial
        reg_term = self.delta * reg_loss

        total_loss = rec_term + spatial_term + reg_term
        if stage == 2 and kl_loss is not None:
            total_loss = total_loss + self.lambda_dec * kl_loss

        return total_loss, l_rec, l_spatial, (kl_loss.item() if kl_loss is not None else 0.0)
# Cell 6: Clustering & Two-Stage Training Engine (400 Epochs)
def mclust_R(adata_or_embeddings, num_cluster, modelNames='EEE', use_pca=True, n_comps=20, random_seed=42):
    np.random.seed(random_seed)
    if init_r_mclust():
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import pandas2ri, default_converter
            from rpy2.robjects.conversion import localconverter
            robjects.r.library("mclust")
            robjects.r["set.seed"](random_seed)
            rmclust = robjects.r["Mclust"]

            X = np.array(adata_or_embeddings, dtype=np.float64)
            if use_pca and X.shape[1] > n_comps:
                comps = min(n_comps, X.shape[1], X.shape[0] - 1)
                pca_model = PCA(n_components=comps, random_state=random_seed)
                X = pca_model.fit_transform(X)

            df = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(X.shape[1])])
            subset_size = min(300, X.shape[0])
            subset_indices = robjects.IntVector(list(np.random.choice(range(1, X.shape[0] + 1), subset_size, replace=False)))
            init_list = robjects.ListVector({'subset': subset_indices})

            res = None
            with localconverter(default_converter + pandas2ri.converter):
                try:
                    res = rmclust(df, G=num_cluster, modelNames=modelNames, initialization=init_list)
                except Exception:
                    res = rmclust(df, G=num_cluster)

            if res is not None and hasattr(res, 'names') and 'classification' in res.names:
                return np.array(res['classification']).astype(int)
        except Exception as e:
            print(f"[Notice] mclust fallback ({e}). Using KMeans.")
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=num_cluster, n_init=10, random_state=random_seed).fit_predict(adata_or_embeddings)

def cluster_embeddings(embeddings, num_clusters, method='kmeans', random_seed=42):
    if method == 'mclust':
        return mclust_R(embeddings, num_cluster=num_clusters, random_seed=random_seed)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=num_clusters, n_init=10, random_state=random_seed).fit_predict(embeddings)

def train_model_dec(
    model,
    data,
    spatial_adj_norm,
    gt_labels=None,
    pretrain_epochs=250,
    finetune_epochs=150,
    lr=1e-3,
    num_clusters=10,
    tool='kmeans',
    seed=42,
    p_update_interval=5
):
    optimizer_stage1 = torch.optim.Adam(model.parameters(), lr=lr)

    best_sil = -1.0
    best_epoch = -1
    best_stage = "Pre-train"
    best_embeddings = None
    best_labels = None
    best_imputed_rna = None
    best_gates = None
    best_model_state_overall = None
    
    best_ari_overall = -1.0
    ari_at_best_sil = -1.0
    best_sil_stage1 = -1.0
    best_epoch_stage1 = -1
    best_model_state_stage1 = None
    best_embeddings_stage1 = None

    history = {
        'total_loss': [],
        'recon_loss': [],
        'spatial_loss': [],
        'kl_loss': [],
        'l_sim': [],
        'l_dist': [],
        'l_aux': [],
        'silhouette': [],
        'ari': []
    }

    print(f"\n{'='*75}\n[Stage 1/2] Gated Fusion Pre-training ({pretrain_epochs} Epochs | {tool})\n{'='*75}")
    for epoch in range(pretrain_epochs):
        model.train()
        optimizer_stage1.zero_grad()
        sim_z, dist_z, fused_z, fused_pro, pro, g_sp, g_mod = model(data, compute_dec=False)
        
        combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)
        l_rec = F.mse_loss(combined_raw, model.gcn.reconstruct_joint(fused_pro))
        l_sim = F.mse_loss(data.x_RNA, model.gcn.reconstruct_rna(sim_z))
        l_dist = F.mse_loss(data.x_RNA, model.gcn.reconstruct_rna(dist_z))
        l_aux = F.mse_loss(data.x_ADT, model.gcn.reconstruct_aux(pro))
        
        loss, l_rec_ret, l_sp, _ = model.compute_losses(data, sim_z, dist_z, fused_z, fused_pro, pro, stage=1)
        loss.backward()
        optimizer_stage1.step()

        model.eval()
        with torch.no_grad():
            sim_z_eval, dist_z_eval, fused_z_eval, fused_pro_eval, pro_eval, g_sp_eval, g_mod_eval = model(data, compute_dec=False)
            epoch_emb = fused_pro_eval.cpu().numpy()

        epoch_labels = cluster_embeddings(epoch_emb, num_clusters, method=tool, random_seed=seed)
        try:
            sil = float(silhouette_score(epoch_emb, epoch_labels))
        except Exception:
            sil = -1.0
            
        ari = -1.0
        if gt_labels is not None:
            ari = float(adjusted_rand_score(gt_labels, epoch_labels))

        history['total_loss'].append(loss.item())
        history['recon_loss'].append(l_rec.item())
        history['spatial_loss'].append(l_sp.item())
        history['kl_loss'].append(0.0)
        history['l_sim'].append(l_sim.item())
        history['l_dist'].append(l_dist.item())
        history['l_aux'].append(l_aux.item())
        history['silhouette'].append(sil)
        history['ari'].append(ari)

        if sil > best_sil:
            best_sil = sil
            ari_at_best_sil = ari
            best_epoch = epoch + 1
            best_stage = "Pre-train"
            best_embeddings = epoch_emb.copy()
            best_labels = epoch_labels.copy()
            best_gates = (g_sp_eval.cpu().numpy(), g_mod_eval.cpu().numpy())
            best_model_state_overall = copy.deepcopy(model.state_dict())
            with torch.no_grad():
                best_imputed_rna = model.gcn.reconstruct_rna(sim_z_eval).cpu().numpy()
                
        if ari > best_ari_overall:
            best_ari_overall = ari

        if sil > best_sil_stage1:
            best_sil_stage1 = sil
            best_epoch_stage1 = epoch + 1
            best_model_state_stage1 = copy.deepcopy(model.state_dict())
            best_embeddings_stage1 = epoch_emb.copy()

        if (epoch + 1) % 25 == 0 or epoch == 0 or (epoch + 1) == pretrain_epochs:
            print(f"Pretrain Ep {epoch+1:3d}/{pretrain_epochs} | Tot: {loss.item():.4f} | Rec: {l_rec.item():.4f} | Spat: {l_sp.item():.4f} | KL: 0.0000 | Sim: {l_sim.item():.4f} | Dist: {l_dist.item():.4f} | Aux: {l_aux.item():.4f} | Sil: {sil:.4f} (Best: {best_sil:.4f} @ Ep {best_epoch}) | ARI: {ari:.4f}")

    print(f"\n[Bridge Transition] Restoring peak Stage-1 model weights (Sil = {best_sil_stage1:.4f} @ Ep {best_epoch_stage1})...")
    if best_model_state_stage1 is not None:
        model.load_state_dict(best_model_state_stage1)

    model.eval()
    with torch.no_grad():
        _, _, _, fused_pro_best, _, _, _ = model(data, compute_dec=False)
        best_emb_aligned = fused_pro_best.cpu().numpy()

    kmeans_init = KMeans(n_clusters=num_clusters, n_init=20, random_state=seed)
    kmeans_init.fit(best_emb_aligned)
    model.init_cluster_centers(kmeans_init.cluster_centers_)
    print("DEC Cluster Prototypes successfully initialized in aligned latent space!")

    lr_stage2 = lr * 0.1
    optimizer_stage2 = torch.optim.Adam(model.parameters(), lr=lr_stage2)
    p_target = None

    print(f"\n{'='*75}\n[Stage 2/2] Gated Spatial Potts Consensus DEC Fine-Tuning ({finetune_epochs} Epochs | lr={lr_stage2:.1e})\n{'='*75}")
    for epoch in range(finetune_epochs):
        curr_epoch = pretrain_epochs + epoch + 1
        model.train()

        if epoch % p_update_interval == 0 or p_target is None:
            with torch.no_grad():
                _, _, _, f_pro_curr, _, _, _ = model(data, compute_dec=False)
                q_curr = model.dec.compute_q(f_pro_curr)
                p_target = model.dec.compute_spatial_target_p(q_curr, spatial_adj_norm)

        optimizer_stage2.zero_grad()
        sim_z, dist_z, fused_z, fused_pro, pro, q, kl_loss, g_sp, g_mod = model(data, compute_dec=True, p_target=p_target)
        
        combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)
        l_rec = F.mse_loss(combined_raw, model.gcn.reconstruct_joint(fused_pro))
        l_sim = F.mse_loss(data.x_RNA, model.gcn.reconstruct_rna(sim_z))
        l_dist = F.mse_loss(data.x_RNA, model.gcn.reconstruct_rna(dist_z))
        l_aux = F.mse_loss(data.x_ADT, model.gcn.reconstruct_aux(pro))
        
        loss, l_rec_ret, l_sp, kl_val = model.compute_losses(data, sim_z, dist_z, fused_z, fused_pro, pro, kl_loss=kl_loss, stage=2)
        loss.backward()
        optimizer_stage2.step()

        model.eval()
        with torch.no_grad():
            sim_z_eval, dist_z_eval, fused_z_eval, fused_pro_eval, pro_eval, g_sp_eval, g_mod_eval = model(data, compute_dec=False)
            q_eval = model.dec.compute_q(fused_pro_eval)
            epoch_emb = fused_pro_eval.cpu().numpy()
            preds_q = q_eval.argmax(dim=1).cpu().numpy()

        n_unique_q = len(np.unique(preds_q))
        if n_unique_q >= 2:
            try:
                sil_q = float(silhouette_score(epoch_emb, preds_q))
            except Exception:
                sil_q = -1.0
        else:
            sil_q = -1.0

        try:
            km_preds = KMeans(n_clusters=num_clusters, n_init=5, random_state=seed).fit_predict(epoch_emb)
            sil_km = float(silhouette_score(epoch_emb, km_preds))
        except Exception:
            sil_km = -1.0

        if sil_q >= sil_km and n_unique_q >= 2:
            sil = sil_q
            epoch_labels = preds_q
        else:
            sil = sil_km
            epoch_labels = km_preds
            
        ari = -1.0
        if gt_labels is not None:
            ari = float(adjusted_rand_score(gt_labels, epoch_labels))

        history['total_loss'].append(loss.item())
        history['recon_loss'].append(l_rec.item())
        history['spatial_loss'].append(l_sp.item())
        history['kl_loss'].append(kl_val)
        history['l_sim'].append(l_sim.item())
        history['l_dist'].append(l_dist.item())
        history['l_aux'].append(l_aux.item())
        history['silhouette'].append(sil)
        history['ari'].append(ari)

        if sil > best_sil:
            best_sil = sil
            ari_at_best_sil = ari
            best_epoch = curr_epoch
            best_stage = "DEC-Potts"
            best_embeddings = epoch_emb.copy()
            best_labels = epoch_labels.copy()
            best_gates = (g_sp_eval.cpu().numpy(), g_mod_eval.cpu().numpy())
            best_model_state_overall = copy.deepcopy(model.state_dict())
            with torch.no_grad():
                best_imputed_rna = model.gcn.reconstruct_rna(sim_z_eval).cpu().numpy()
                
        if ari > best_ari_overall:
            best_ari_overall = ari

        if (epoch + 1) % 25 == 0 or epoch == 0 or (epoch + 1) == finetune_epochs or sil > best_sil_stage1:
            print(f"DEC Ep {epoch+1:3d}/{finetune_epochs} (Total {curr_epoch:3d}) | Tot: {loss.item():.4f} | Rec: {l_rec.item():.4f} | Spat: {l_sp.item():.4f} | KL: {kl_val:.4f} | Sim: {l_sim.item():.4f} | Dist: {l_dist.item():.4f} | Aux: {l_aux.item():.4f} | Sil: {sil:.4f} (Peak: {best_sil:.4f} [{best_stage}] @ Ep {best_epoch}) | ARI: {ari:.4f}")

    if best_model_state_overall is not None:
        model.load_state_dict(best_model_state_overall)

    print(f"\n{'='*75}\n[Best Overall Checkpoint] Highest Silhouette = {best_sil:.4f} (ARI = {ari_at_best_sil:.4f}) achieved in [{best_stage}] at Epoch {best_epoch}.\n{'='*75}")
    return best_embeddings, best_labels, best_imputed_rna, best_gates, best_epoch, best_stage, history, best_sil, ari_at_best_sil, sil, ari


# Cell 7: Dataset Directory & Automatic Download Resolver
CHOICES = [
    ("10x_human_lymph_node_A1", "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1", "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1", "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1", "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1", "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1", "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

def resolve_dataset_files(dataset_name: str, folder_url: str):
    candidate_dirs = [
        f"data/{dataset_name}",
        f"/content/data/{dataset_name}",
        f"/content/drive/MyDrive/Colab/data/{dataset_name}",
        f"/content/drive/MyDrive/Colab/ARISE/data/{dataset_name}",
        f"/content/drive/MyDrive/Colab/{dataset_name}"
    ]
    base_dir = None
    for c_dir in candidate_dirs:
        if os.path.exists(c_dir):
            base_dir = c_dir
            break

    if base_dir is None:
        base_dir = f"data/{dataset_name}"
        os.makedirs(base_dir, exist_ok=True)
        print(f"Downloading dataset {dataset_name} using gdown...")
        os.system(f'gdown --folder "{folder_url}" --output "{base_dir}"')

    rna_path = os.path.join(base_dir, "adata_RNA.h5ad")
    if dataset_name.startswith("10x"):
        aux_path = os.path.join(base_dir, "adata_ADT.h5ad")
        anno_path = os.path.join(base_dir, "annotation.csv")
        gt_col = "manual-anno"
    else:
        aux_path = os.path.join(base_dir, "adata_ATAC.h5ad")
        anno_path = os.path.join(base_dir, "anno.csv")
        gt_col = "cluster"

    if not (os.path.exists(rna_path) and os.path.exists(aux_path) and os.path.exists(anno_path)):
        print(f"Re-downloading {dataset_name}...")
        os.system(f'gdown --folder "{folder_url}" --output "{base_dir}"')

    return rna_path, aux_path, anno_path, gt_col



# ==============================================================================
# Enhanced Plotting and Visualizations
# ==============================================================================

def plot_training_curves(training_results, dataset_name="Dataset", save_path=None, show=True):
    epochs_range = range(1, len(training_results['total_loss']) + 1)
    has_ari = len(training_results.get('ari', [])) > 0
    
    fig, axes = plt.subplots(1, 3 if has_ari else 2, figsize=(18 if has_ari else 12, 4.5), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
        
    # 1. Loss Curves (Plotting ALL losses)
    axes[0].plot(epochs_range, training_results['total_loss'], color='#1f77b4', linewidth=2.5, label='Total Loss')
    axes[0].plot(epochs_range, training_results['recon_loss'], color='#ff7f0e', linewidth=1.5, linestyle='--', label='Recon Loss')
    axes[0].plot(epochs_range, training_results['spatial_loss'], color='#2ca02c', linewidth=1.5, linestyle='-.', label='Spatial Loss')
    axes[0].plot(epochs_range, training_results['kl_loss'], color='#d62728', linewidth=1.5, linestyle=':', label='KL Loss')
    axes[0].plot(epochs_range, training_results['l_sim'], color='#9467bd', linewidth=1.0, alpha=0.7, label='L_sim')
    axes[0].plot(epochs_range, training_results['l_dist'], color='#8c564b', linewidth=1.0, alpha=0.7, label='L_dist')
    axes[0].plot(epochs_range, training_results['l_aux'], color='#e377c2', linewidth=1.0, alpha=0.7, label='L_aux')
    
    axes[0].set_title(f'Loss Curves - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss Value', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True, fontsize='small', loc='upper right')

    # 2. Silhouette Score Curve
    axes[1].plot(epochs_range, training_results['silhouette'], color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    axes[1].set_title(f'Silhouette Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, loc='lower right')
    
    # 3. ARI Curve
    if has_ari:
        axes[2].plot(epochs_range, training_results['ari'], color='#ff7f0e', linewidth=2.0, label='Epoch ARI')
        axes[2].set_title(f'ARI Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend(frameon=True, loc='lower right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_all_visualizations(
    adata_RNA,
    best_emb,
    best_labels,
    sil,
    ari,
    history,
    dataset_name="Dataset",
    seed=42,
    true_labels=None,
    output_dir="results",
    show=False
):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(plots_dir, "curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "umap"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "violin"), exist_ok=True)

    if true_labels is not None:
        adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
    elif 'ground_truth' not in adata_RNA.obs:
        adata_RNA.obs['ground_truth'] = pd.Categorical(best_labels.astype(str))

    adata_RNA.obsm['ASTRA_Emb'] = best_emb
    adata_RNA.obs['predicted_domain'] = pd.Categorical(best_labels.astype(str))

    # 1. Curves
    curve_path = os.path.join(plots_dir, "curves", f"{dataset_name}_seed{seed}_training_curves.png")
    plot_training_curves(history, dataset_name=f"{dataset_name} (Seed {seed})", save_path=curve_path, show=show)

    # 2. Spatial Domains
    spatial_coords = adata_RNA.obsm.get('spatial', None)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    if spatial_coords is not None:
        gt_cats = pd.Categorical(adata_RNA.obs['ground_truth'])
        pred_cats = pd.Categorical(adata_RNA.obs['predicted_domain'])
        axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=gt_cats.codes, cmap='tab20', s=10, alpha=0.9)
        axes[0].set_title(f'Ground Truth ({dataset_name})', fontsize=12, fontweight='bold')
        axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=pred_cats.codes, cmap='tab20', s=10, alpha=0.9)
        axes[1].set_title(f'ASTRA-Redo Domains (ARI: {ari:.4f})', fontsize=12, fontweight='bold')
    plt.suptitle(f"Spatial Domains Comparison - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "spatial", f"{dataset_name}_seed{seed}_spatial.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. UMAP
    sc.pp.neighbors(adata_RNA, use_rep='ASTRA_Emb')
    sc.tl.umap(adata_RNA)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc.pl.umap(adata_RNA, color='ground_truth', ax=axes[0], show=False, title='UMAP: Ground Truth')
    sc.pl.umap(adata_RNA, color='predicted_domain', ax=axes[1], show=False, title=f'UMAP: Predicted Domains (Sil: {sil:.4f})')
    plt.suptitle(f"UMAP Joint Representation - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "umap", f"{dataset_name}_seed{seed}_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Violin Plots
    from sklearn.metrics import silhouette_samples
    try:
        sample_sil_values = silhouette_samples(best_emb, best_labels)
        adata_RNA.obs['silhouette_coefficient'] = sample_sil_values
        adata_RNA.obs['Latent_Dim_1'] = best_emb[:, 0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        sns.violinplot(data=adata_RNA.obs, x='predicted_domain', y='silhouette_coefficient', palette='Set2', inner='quartile', ax=axes[0])
        axes[0].axhline(sil, color='red', linestyle='--', label=f'Mean Sil: {sil:.4f}')
        axes[0].set_title("Silhouette Coefficient per Predicted Domain", fontsize=12, fontweight='bold')
        sns.violinplot(data=adata_RNA.obs, x='predicted_domain', y='Latent_Dim_1', palette='tab10', inner='box', ax=axes[1])
        axes[1].set_title("Latent Dimension 1 Distribution per Domain", fontsize=12, fontweight='bold')
        plt.suptitle(f"Violin Plots: Cluster Profiles - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "violin", f"{dataset_name}_seed{seed}_violin.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Skipping Violin plots due to error: {e}")

# ==============================================================================
# CLI and Experiment Execution
# ==============================================================================

def run_experiment(datasets="all", seeds=None, pretrain_epochs=250, finetune_epochs=150, lr=1e-3, 
                   lambda_dec=1.0, lambda_spatial=0.15, output_dir="results/astra_redo_Results", 
                   visualize=True):
    os.makedirs(output_dir, exist_ok=True)
    if seeds is None: seeds = [42, 1234, 2024]
    
    if datasets == "all":
        dataset_names = [c[0] for c in CHOICES]
    else:
        dataset_names = datasets
        
    all_results = []
    
    for dataset_name in dataset_names:
        folder_url = next((c[1] for c in CHOICES if c[0] == dataset_name), None)
        if not folder_url:
            print(f"Dataset {dataset_name} not found in CHOICES. Skipping.")
            continue
            
        print(f"
{'='*80}
 STARTING DATASET: {dataset_name}
{'='*80}")
        rna_path, aux_path, anno_path, gt_col = resolve_dataset_files(dataset_name, folder_url)
        adata_RNA = sc.read_h5ad(rna_path)
        adata_aux = sc.read_h5ad(aux_path)
        anno_df = pd.read_csv(anno_path, index_col=0)
        
        gt_values = anno_df[gt_col] if isinstance(anno_df[gt_col], pd.Series) else anno_df[gt_col].iloc[:, 0]
        adata_RNA.obs['ground_truth'] = gt_values.values
        adata_aux.obs['ground_truth'] = gt_values.values
        
        RNA_data, ADT_data, adata_RNA, adata_aux = preprocess_universal(adata_RNA, adata_aux, dataset_name)
        cell_positions = adata_RNA.obsm['spatial']
        graph_data = build_dual_graph(RNA_data, ADT_data, cell_positions, device=device)
        spatial_adj_norm = compute_spatial_adj_norm(graph_data.dist_edge_index, RNA_data.shape[0], device=device)
        n_clusters = int(adata_RNA.obs['ground_truth'].dropna().nunique())
        
        dataset_results = []
        for seed in seeds:
            print(f"
{'-'*65}
 Dataset: {dataset_name} | Seed: {seed}
{'-'*65}")
            set_seed(seed)
            model = ASTRA_v1_DEC_Gated(
                in_channels=RNA_data.shape[1], hidden_channels=512, out_channels=64,
                q_dim=ADT_data.shape[1], num_clusters=n_clusters,
                beta=25.0, gamma=10.0, delta=1.0, lambda_dec=lambda_dec, lambda_spatial=lambda_spatial
            ).to(device)
            
            gt_labels_array = adata_RNA.obs['ground_truth'].astype(str).values
            best_emb, best_labels, best_imputed_rna, best_gates, best_ep, best_stg, hist, best_sil, ari_at_best_sil, last_sil, last_ari = train_model_dec(
                model=model, data=graph_data, spatial_adj_norm=spatial_adj_norm,
                gt_labels=gt_labels_array, pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                lr=lr, num_clusters=n_clusters, tool='kmeans', seed=seed
            )
            
            if visualize:
                plot_all_visualizations(
                    adata_RNA=adata_RNA, best_emb=best_emb, best_labels=best_labels,
                    sil=best_sil, ari=ari_at_best_sil, history=hist,
                    dataset_name=dataset_name, seed=seed, true_labels=gt_labels_array,
                    output_dir=output_dir, show=False
                )
            
            res_dict = {
                'dataset': dataset_name, 'seed': seed,
                'Best_Silhouette': best_sil, 'Best_Sil_Epoch': best_ep,
                'Best_ARI': ari_at_best_sil, 'Best_Stage': best_stg,
                'Last_Silhouette': last_sil, 'Last_ARI': last_ari
            }
            dataset_results.append(res_dict)
            all_results.append(res_dict)
            
        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"ASTRA_Redo_{dataset_name}_results.csv"), index=False)
        
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "ASTRA_Redo_all_results.csv"), index=False)
    print("
Experiment Completed! Results saved to", output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASTRA-Redo Multi-Omics Model CLI")
    parser.add_argument('--datasets', nargs='+', default=['all'], help="Datasets to run (names or 'all')")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 2024], help="Random seeds")
    parser.add_argument('--pretrain_epochs', type=int, default=250, help="Stage 1 Epochs")
    parser.add_argument('--finetune_epochs', type=int, default=150, help="Stage 2 Epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--lambda_dec', type=float, default=1.0, help="DEC loss weight")
    parser.add_argument('--lambda_spatial', type=float, default=0.15, help="Spatial consensus weight")
    parser.add_argument('--output_dir', type=str, default='results/astra_redo_Results', help="Output directory")
    parser.add_argument('--no_visualize', action='store_true', help="Disable plotting")
    
    args = parser.parse_args()
    
    run_experiment(
        datasets=args.datasets if args.datasets[0] != 'all' else 'all',
        seeds=args.seeds,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        lambda_dec=args.lambda_dec,
        lambda_spatial=args.lambda_spatial,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )
