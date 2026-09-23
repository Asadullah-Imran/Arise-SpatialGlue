# !pip install scanpy anndata scikit-misc

# ------------------------------ Model CODE ----------------------

import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """
    Construct neighbor graphs, including feature graph and spatial graph.
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data

def pca(adata, use_reps=None, n_comps=10):

    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray())
       else:
          feat_pca = pca.fit_transform(adata.X)

    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):

    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "connectivity", metric="correlation", include_self=False):

    """Constructing feature neighbor graph according to expresss profiles"""

    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""

    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)

    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)

    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)

    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)

    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)

    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }

    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    """
    Overall encoder.
    """
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)

        # graph2
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)

        # between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # consistency encoding
        emb_latent_omics1_across_recon = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
        emb_latent_omics2_across_recon = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)

        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_latent_omics1_across_recon':emb_latent_omics1_across_recon,
                   'emb_latent_omics2_across_recon':emb_latent_omics2_across_recon,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha':alpha_omics_1_2
                   }

        return results


class Encoder(Module):
    """
    Modality-specific GNN encoder.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class Decoder(Module):
    """
    Modality-specific GNN decoder.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class AttentionLayer(Module):
    """
    Attention layer.
    """
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha
import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter
    import pandas as pd

    np.random.seed(random_seed)

    robjects.r.library("mclust")

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)

    rmclust = robjects.r["Mclust"]

    # Get the data
    X = np.array(adata.obsm[used_obsm], dtype=np.float64)

    print("Input shape:", X.shape)

    # Convert to DataFrame with column names
    df = pd.DataFrame(
        X,
        columns=[f'PC{i+1}' for i in range(X.shape[1])]
    )

    # Use subset for initialization to dramatically speed up mclust for large datasets
    subset_size = min(300, X.shape[0])
    subset_indices = robjects.IntVector(list(np.random.choice(range(1, X.shape[0] + 1), subset_size, replace=False)))
    init_list = robjects.ListVector({'subset': subset_indices})

    with localconverter(default_converter + pandas2ri.converter):
        res = rmclust(
            df,
            G=num_cluster,
            modelNames=modelNames,
            initialization=init_list
        )

    if hasattr(res, 'rx2'):
        mclust_res = np.array(res.rx2('classification'))
    elif hasattr(res, 'getbyname'):
        mclust_res = np.array(res.getbyname('classification'))
    else:
        mclust_res = np.array(res['classification'])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('str')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20, random_seed=2020):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.
    random_seed : int, optional
        Random seed for clustering. The default is 2020.

    Returns
    -------
    None.

    """

    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == 'mclust':
       if use_pca:
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters, random_seed=random_seed)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters, random_seed=random_seed)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca:
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca:
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!."

    return res

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values

  """
  import pandas as pd

  df = pd.DataFrame(columns=[modality1, modality2, 'label'])
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2)

  plt.tight_layout(w_pad=0.05)
  plt.show()
import torch
from tqdm import tqdm
import torch.nn.functional as F


class Train_SpatialGlue:
    def __init__(self,
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        epochval=None,
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 5, 1, 1]
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.loss_history = []

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        print("epochval -->", epochval)

        if self.datatype == 'SPOTS':
           self.epochs = 600
           self.weight_factors = [1,5,1,1]

        elif self.datatype == 'Stereo-CITE-seq':
           self.epochs = 1500
           self.weight_factors = [1,10,1,10]

        elif self.datatype == '10x':
           self.epochs = 200
           self.weight_factors = [1,5,1,10]

        elif self.datatype == 'Spatial-epigenome-transcriptome':
           self.epochs = 1600
           self.weight_factors = [1,5,1,1]

        if epochval is not None:
            self.epochs = epochval



    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # correspondence loss
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            loss = self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_recon_omics2 + self.weight_factors[2]*self.loss_corr_omics1 + self.weight_factors[3]*self.loss_corr_omics2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch == 0:
               print(f'Epoch {epoch} - Initial Loss: {loss.item():.4f}')

            self.loss_history.append(loss.item())

        print(f"Model training finished! Final Loss: {loss.item():.4f}\n")

        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1, self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'SpatialGlue': emb_combined.detach().cpu().numpy(),
                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy(),
                  'loss_history': self.loss_history}

        return output










# ------------------------------ Run Configuration ----------------------

choices = [
    ("10x_human_lymph_node_A1",
     "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1",
     "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1",
     "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1",
     "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1",
     "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1",
     "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

# Run all 6 datasets (indices 0 to 5)
DATASET_INDICES = [0, 1, 2, 3, 4, 5]

# Define seeds to loop over (20 seeds)
# Define seeds to loop over
SEEDS = [
    42, 0, 1, 7, 123, 1234, 2022, 2023, 2024, 1337
]

# Set R_HOME and path before any rpy2 operations
import os
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['PATH'] = '/usr/lib/R/bin:' + os.environ['PATH']

# Import rpy2 and setup mclust
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter
import rpy2.robjects.conversion as cv

cv.set_conversion(default_converter + pandas2ri.converter)

# Suppress warnings
robjects.r.options(warn=-1)

# Install mclust only if it is not already installed
robjects.r("""
if (!requireNamespace("mclust", quietly = TRUE)) {
    install.packages("mclust", repos="https://cloud.r-project.org")
}
library(mclust)
""")

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

all_results = []
tool = 'mclust'  # mclust, leiden, and louvain
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for dataset_idx in DATASET_INDICES:
    dataset_name, folder_url = choices[dataset_idx]

    print("\n" + "#"*80)
    print(f" STARTING DATASET: {dataset_name} ".center(80, "#"))
    print("#"*80)

    # ------------------ Download & Path Configuration ------------------
    base = f"data/{dataset_name}"
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")

    # Human datasets have ADT; Mouse datasets have ATAC
    if dataset_name.startswith("10x"):
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_column = "manual-anno"
        data_type = '10x'
    else:
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_column = "cluster"
        data_type = 'Spatial-epigenome-transcriptome'

    # Check if files already exist locally before triggering download
    if not os.path.exists(rna_path) or not os.path.exists(other_path) or not os.path.exists(annotation_path):
        print(f"Downloading dataset files into: {base}")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{folder_url}" --output "{base}"')
    else:
        print(f"Dataset files already exist at {base}. Skipping download.")

    # ------------------ Load AnnData & Annotations ------------------
    adata_omics1 = sc.read_h5ad(rna_path)
    adata_omics2 = sc.read_h5ad(other_path)

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    adata_omics1.obs['ground_truth'] = anno_df[gt_column]
    adata_omics2.obs['ground_truth'] = anno_df[gt_column]

    # ------------------ Preprocessing ------------------
    # RNA preprocessing
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    if not dataset_name.startswith("10x"):
        sc.pp.filter_cells(adata_omics1, min_genes=200)

    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]

    # Protein or ATAC preprocessing
    if dataset_name.startswith("10x"):
        adata_omics1.obsm["feat"] = pca(
            adata_omics1_high,
            n_comps=adata_omics2.n_vars - 1
        )
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm["feat"] = pca(
            adata_omics2,
            n_comps=adata_omics2.n_vars - 1
        )
    else:
        adata_omics1.obsm["feat"] = pca(
            adata_omics1_high,
            n_comps=50
        )
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        if "X_lsi" not in adata_omics2.obsm:
            sc.pp.highly_variable_genes(
                adata_omics2,
                flavor="seurat_v3",
                n_top_genes=3000
            )
            lsi(
                adata_omics2,
                use_highly_variable=False,
                n_components=51
            )
        adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

    # ------------------ Spatial Neighbor Graph Construction ------------------
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

    # ------------------ Seed Iterations ------------------
    dataset_results = []
    n_ground_truth = adata_omics1.obs["ground_truth"].nunique()
    print(f"Number of ground truth classes: {n_ground_truth}")
    n_cluster = n_ground_truth

    for seed in SEEDS:
        print(f"\n" + "-"*60)
        print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
        print("-"*60)

        # 1. Set seed
        fix_seed(seed)

        # 2. Model Training
        # We need a fresh deep copy of the data dictionary for each seed run to avoid side-effects
        import copy
        data_copy = {
            'adata_omics1': data['adata_omics1'].copy(),
            'adata_omics2': data['adata_omics2'].copy()
        }

        # Train model
        model = Train_SpatialGlue(data_copy, datatype=data_type, device=device, random_seed=seed)
        output = model.train()

        adata = data_copy['adata_omics1'].copy()
        adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
        adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
        adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
        adata.obsm['alpha'] = output['alpha']
        adata.obsm['alpha_omics1'] = output['alpha_omics1']
        adata.obsm['alpha_omics2'] = output['alpha_omics2']

        # 3. Clustering
        clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=n_cluster, method=tool, use_pca=True, random_seed=seed)

        # 4. Score Generating
        y_true = adata.obs['ground_truth'].astype(str)
        y_pred = adata.obs['SpatialGlue'].astype(str)

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)

        # Calculate Silhouette, CHI, DBI Scores using robust LabelEncoder
        joint_feat = adata.obsm['SpatialGlue']
        le = LabelEncoder()
        y_pred_int = le.fit_transform(y_pred)
        sil_score = silhouette_score(joint_feat, y_pred_int)
        chi_score = calinski_harabasz_score(joint_feat, y_pred_int)
        dbi_score = davies_bouldin_score(joint_feat, y_pred_int)

        print(f"\nResult for dataset: {dataset_name} | seed: {seed}")
        print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil_score:.4f}")

        # Store seed results
        res_dict = {
            'dataset': dataset_name,
            'seed': seed,
            'ARI': ari,
            'NMI': nmi,
            'AMI': ami,
            'Silhouette': sil_score,
            'CHI': chi_score,
            'DBI': dbi_score,
            'no_cluster': n_cluster
        }
        dataset_results.append(res_dict)
        all_results.append(res_dict)

    # Write dataset summary statistics and save csv for this dataset
    df_ds = pd.DataFrame(dataset_results)
    print("\n" + "="*80)
    print(f" RESULTS SUMMARY FOR {dataset_name} ".center(80, "="))
    print("="*80)
    print(df_ds.to_string(index=False))
    print("-"*80)
    print("Summary Statistics:")
    print(df_ds.describe().loc[['mean', 'std']])
    print("="*80)

    # Generate text summary with Mean ± SD format for this dataset
    summary_lines = [f"Summary Stats for {dataset_name}:"]
    metrics_list = ['ARI', 'NMI', 'AMI', 'Silhouette', 'CHI', 'DBI']
    for metric in metrics_list:
        mean_val = df_ds[metric].mean()
        std_val = df_ds[metric].std()
        summary_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    print("\n".join(summary_lines))
    print("="*80)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df_ds.to_csv(f"results/SpatialGlue_{dataset_name}_results.csv", index=False)

# Write overall summary to file
df_all = pd.DataFrame(all_results)
df_all.to_csv("results/SpatialGlue_all_results.csv", index=False)
print("\n" + "="*80)
print(" ALL DATASETS COMPLETED ".center(80, "="))
print("="*80)
print(df_all.groupby('dataset')[['ARI', 'NMI', 'Silhouette']].mean())
print("="*80)










