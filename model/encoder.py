import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn import MessagePassing, SAGEConv, GATConv, GCNConv, GINConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
import torch_sparse
from torch_scatter import scatter_add
from typing import Iterable
from typing_extensions import Final

from utils.link_utils import adjoverlap
# a vanilla message passing layer 
class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x
    

convdict = {
    "gcn":
    GCNConv,
    "gcn_cached":
    lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max":
    lambda indim, outdim: GCNConv(
        indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": 
    lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none":
    None
}

predictor_dict = {}

# Edge dropout
class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]

# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool] # whether to rescale edge weight
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args,
            in_dropout=0.0, dropout=0.5, heads=1, beta=-1, pre_ln=False, gnn='gcn'):
        super(GNN, self).__init__()

        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.task = args.task
        local_layers = args.local_layers

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)
            
        self.vqs = torch.nn.ModuleList()
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList() 
        self.lins = torch.nn.ModuleList() 
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if gnn=='gat':
                self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        #self.linear_gnn = torch.nn.Linear(heads*hidden_channels, local_layers*3)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, hidden_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        id_list = []
        quantized_list = []
        total_commit_loss = 0
        token_list = []

        ## equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)

            x = local_conv(x, edge_index) + self.lins[i](x)
            # x = self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x_local = x_local + x

        #gnn_id = self.linear_gnn(x_local)
        x = self.pred_local(x_local)
        # print(x)
        return x
    



class InnerProductDecoder(torch.nn.Module):

    def __init__(self, hidden_dim=None, output_dim=None):
        super().__init__()
        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
    def forward_pairwise(self, z: Tensor, x: Tensor, sigmoid: bool = False) -> Tensor:
        """z: [N, d], x: [N, d] → similarity per node"""
        z = F.normalize(z, dim=-1)
        x = F.normalize(x, dim=-1)
        sim = (z * x).sum(dim=-1)  # [N]
        return torch.sigmoid(sim) if sigmoid else sim
    
    def forward_pairwise_batch(self, z: Tensor, x: Tensor, sigmoid: bool = False) -> Tensor:
        """z: [N, d], x: [N, k, d] → similarity [N, k]"""
        z = F.normalize(z, dim=-1)

        x = F.normalize(x, dim=-1)
        z = z.unsqueeze(1)  # [N, 1, d]
        sim = (z * x).sum(dim=-1)  # [N, k]
        return torch.sigmoid(sim) if sigmoid else sim


class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            lnfn(in_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        # print(x)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])

predictor_dict = {
    "cn1": CNLinkPredictor,
}