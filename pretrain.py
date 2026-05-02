import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.utils import mask_feature, dropout_edge
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import k_hop_subgraph
from utils.data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits, mask_node_feature, compute_laplacian_pe
from utils.args import get_args_pretrain
from torch_geometric.utils import to_scipy_sparse_matrix
from dataset import load_dataset
from model.encoder import GNN, InnerProductDecoder
from vector_quantize_pytorch import SimVQ
from model.vq import VectorQuantize
from model.pt_model import JointModel
import networkx as nx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph
import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict

def jaccard_overlap(edge_index, num_nodes):
    neighbors = defaultdict(set)
    for src, dst in edge_index.t().tolist():
        neighbors[src].add(dst)
        neighbors[dst].add(src) 

    scores = []
    for u in range(num_nodes):
        for v in neighbors[u]:
            if u >= v: continue
            inter = neighbors[u] & neighbors[v]
            union = neighbors[u] | neighbors[v]
            if len(union) == 0:
                continue
            score = len(inter) / len(union)
            scores.append(score)
    return np.mean(scores)

def pca_components_for_variance(x: torch.Tensor, threshold: float = 0.95):
    x_np = x.cpu().numpy()
    pca = PCA(n_components=min(x_np.shape))
    pca.fit(x_np)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, threshold) + 1
    return n_components

def average_degree(edge_index, num_nodes):
    deg = torch.bincount(edge_index[0], minlength=num_nodes)
    return deg.float().mean().item()

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

def compute_graph_avg_jaccard(edge_index, num_nodes, num_sample=1000):

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tolil()

    jaccards = []
    for _ in range(num_sample):
        i, j = np.random.randint(0, num_nodes, 2)
        n_i = set(adj.rows[i])
        n_j = set(adj.rows[j])

        if len(n_i) == 0 and len(n_j) == 0:
            jacc = 1.0
        elif len(n_i.union(n_j)) == 0:
            jacc = 0.0
        else:
            jacc = len(n_i.intersection(n_j)) / len(n_i.union(n_j))
        jaccards.append(jacc)

    return np.mean(jaccards)
def anneal_temperature(global_step, max_steps, start_temp=1.0, final_temp=0.1):
    # exponential decay
    ratio = global_step / max_steps
    temp = start_temp * (final_temp / start_temp) ** ratio
    return max(temp, final_temp)

def compute_structural_entropy(edge_index, num_nodes):
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G_nx = to_networkx(data, to_undirected=True)
    
    edge_list = list(G_nx.edges())  # list of (u, v)
    grakel_graph = Graph(edge_list, node_labels={i: str(i) for i in G_nx.nodes()})

    # 3. WL kernel (labels will be updated inside)
    wl_kernel = WeisfeilerLehman(n_iter=2)
    wl_kernel.fit_transform([grakel_graph])  # triggers label updates
    labels = grakel_graph.node_labels  # dict: node_id → WL label

    # 4. Count WL label frequencies
    values = list(labels.values())
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    return entropy

def pretrain_an_epoch(args, epoch, x, edge_index, aug_graph, opt, model):
    model_path = osp.join(args.model_dir, f'{args.dataset}_with_emb_{args.use_emb_recon_loss}_feat_{args.use_feat_recon_loss}_topo_{args.use_topo_recon_loss}_contrastive_{args.contrastive_recon}')
    os.makedirs(model_path, exist_ok=True)

    loss_train_b = 0
    
    loss_weights = {
        'feat_recon_loss': 100.0,
        'emb_recon_loss': 1.0,
        'topo_recon_loss': 0.01,
        'commit_loss': 1.0,
        'topo_contrastive': 1.0
    }
    opt.zero_grad()
    temperature = anneal_temperature(epoch, args.epochs, start_temp=1.0, final_temp=0.1)
    if args.stage == 1:
        out, losses = model(x, edge_index, aug_graph, args.stage, temperature, opt)

        total_loss = sum(
            loss_weights[k] * v
            for k, v in losses.items()
            if v is not None and k in loss_weights
        )

        loss_train_b = total_loss.item()

        total_loss.backward()
        opt.step()

        if epoch % 5 == 0:
                print(f"[Epoch {epoch:03d}]: "
                    f"loss: {loss_train_b} "
                    f"commit_loss:{losses['commit_loss'].item()}"
                    f"feat_loss:{losses['feat_recon_loss'].item()}"
                    f"topo_loss:{losses['topo_recon_loss'].item()}"
                    f"emb_loss:{losses['emb_recon_loss'].item()}")

        return {'loss': loss_train_b} 
    
        
    # out, quantized, embed_ind, commit_loss, codebook_emb
    if args.stage == 2:
        out, quantized, embed_ind, losses = model(x, edge_index, aug_graph, args.stage, temperature, opt)
        total_loss = sum(
            loss_weights[k] * v
            for k, v in losses.items()
            if v is not None and k in loss_weights and k != 'topo_contrastive'
        )


        loss_train_b = total_loss.item()

        # if not args.contrastive_recon:
        #     total_loss.backward()
        #     if not args.contrastive_recon:
        total_loss.backward()   
            
        opt.step()
        

        with torch.no_grad():
            embed_ind = embed_ind.view(-1, 1)  # [N, H]
            num_codes_total = args.num_codes
            heads = embed_ind.shape[1]

            all_stats = []

            for h in range(heads):
                head_indices = embed_ind[:, h]
                used_code_ids = head_indices.unique()
                num_codes_used = used_code_ids.numel()
                code_usage_rate = num_codes_used / num_codes_total

                token_counts = torch.bincount(head_indices, minlength=num_codes_total).float()
                token_probs = token_counts / token_counts.sum()
                nonzero_probs = token_probs[token_probs > 0]

                entropy = -torch.sum(nonzero_probs * torch.log2(nonzero_probs)).item()
                perplexity = 2 ** entropy
                normalized_perplexity = perplexity / num_codes_total

                all_stats.append({
                    'head': h,
                    'used': num_codes_used,
                    'usage_rate': code_usage_rate,
                    'entropy': entropy,
                    'perplexity': perplexity,
                    'norm_perplexity': normalized_perplexity,
                    'all_losses': {
                                    k: (v if k == 'topo_contrastive' else v.item())
                                    for k, v in losses.items() if v is not None}})
        
            if epoch % 5 == 0:
                for stat in all_stats:
                    print(f"[Epoch {epoch:03d}] Head {stat['head']}: "
                        f"loss: {loss_train_b} "
                        f"commit_loss:{losses['commit_loss'].item()}"
                        f"feat_loss:{losses['feat_recon_loss'].item()}"
                        f"topo_loss:{losses['topo_recon_loss'].item()}"
                        f"emb_loss:{losses['emb_recon_loss'].item()}"
                        f"topo_contra_loss:{losses['topo_contrastive']}"
                        f"Used {stat['used']}/{num_codes_total} "
                        f"({stat['usage_rate']:.2%}) | "
                        f"Entropy: {stat['entropy']:.4f} | "
                        f"Perplexity: {stat['perplexity']:.2f} "
                        f"({stat['norm_perplexity']:.2%})")
                # if args.use_feat_recon_loss:
                #     feat_recon = F.mse_loss(model.feat_recon(quantized[aug_graph[2]]), x[aug_graph[2]])
                #     print(f'feat_recon_MSE: {feat_recon}')
                # if args.use_topo_recon_loss:
                #     topo_recon = model.mse_link_recon_loss(quantized, edge_index[:, aug_graph[3]])
                #     print(f'topo_recon_MSE: {topo_recon}')
            
                
            if epoch % 5 == 0 and args.stage == 2:
                torch.save({
                    'quantized': quantized.cpu(),           # [N, D]
                    'indices': embed_ind.cpu()             # [N]
                }, osp.join('../GQT/pretrained_codes/', f'{args.dataset}_vq_{args.num_codes}_reg_{epoch}_size_{args.num_codes}_dim_{args.hidden_channels}.pt'))
                print("codes saved!")
            


        return {'loss': loss_train_b, 'collapse_stats': all_stats, 'quantized_idx':embed_ind}

def run():
    args = get_args_pretrain()

    device = torch.device("cuda:" + str(3)) if torch.cuda.is_available() else torch.device("cpu")

    ## load pre-training datasets
    dataset = load_dataset(args.data_dir, args.dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(
            dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

    # pos_enc = compute_laplacian_pe(dataset.graph['edge_index'], n, k=args.pos_dim)

    dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

    ## initial models
    loss_flags = {
        'commit_loss': args.use_commit_loss,
        'emb_recon_loss': args.use_emb_recon_loss,
        'feat_recon_loss': args.use_feat_recon_loss,
        'topo_recon_loss': args.use_topo_recon_loss,
        'contrastive_recon': args.contrastive_recon
    }
    if args.encoder == 'gnn':
        encoder = GNN(d, args.hidden_channels, c, args,
                    in_dropout=args.in_dropout, dropout=args.dropout,
                    heads=args.num_heads, pre_ln=args.pre_ln, gnn='gat')
    elif args.encoder == 'mlp':
        encoder = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, args.hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_channels, args.hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(args.hidden_channels),
            nn.Linear(args.hidden_channels, args.hidden_channels)
        )
    
    # gnn_path = osp.join('pretrained_gnn/', f'{args.dataset}_feature_rec_{args.use_feat_recon_loss}_link_rec_{args.use_topo_recon_loss}')
    if args.encoder == 'gnn':
        gnn_path = osp.join('pretrained_gnn/', f'{args.dataset}_feature_rec_{True}_link_rec_{True}')
    elif args.encoder == 'mlp':
        gnn_path = osp.join('pretrained_mlp/', f'{args.dataset}_feature_rec_{True}_link_rec_{True}')
    os.makedirs(gnn_path, exist_ok=True)
    if args.stage == 2 and args.pretrained_gnn == 'True':
        print('pretrained gnn loaded')
        model_path = osp.join('pretrained_gnn', f'{args.dataset}_feature_rec_{args.use_feat_recon_loss}_link_rec_{args.use_topo_recon_loss}')
        load_gnn_path = osp.join(model_path, f'epoch_80.pt')
        state_dict = torch.load(load_gnn_path)
        encoder.load_state_dict(state_dict)
    if args.ema_update == 'True':
        vq = VectorQuantize(separate_codebook_per_head=True, heads=args.num_heads, dim=args.hidden_channels, codebook_size=args.num_codes, kmeans_init=True, use_cosine_sim=True)
    elif args.sim == 'True':
        vq = SimVQ(dim=args.hidden_channels,codebook_size=args.num_codes,rotation_trick = True)
    else:
        vq = VectorQuantize(separate_codebook_per_head=True, heads=args.num_heads, dim=args.hidden_channels, codebook_size=args.num_codes, kmeans_init=True, use_cosine_sim=True, ema_update=False, learnable_codebook=True
    )
    feat_recon_decoder = None
    topo_recon_decoder = None
    contrastive_feat_recon_decoder = None
    contrastive_emb_recon_decoder = None

    decoder = nn.Sequential(
            nn.Linear(args.hidden_channels, args.hidden_channels),
            nn.ReLU(),
            nn.Linear(args.hidden_channels, args.hidden_channels),
            nn.ReLU(),
            nn.Linear(args.hidden_channels, args.hidden_channels),
            nn.ReLU()
                )

    if args.use_feat_recon_loss:
        # feat_recon_decoder = nn.Linear(args.hidden_channels + args.pos_dim, d)
        feat_recon_decoder = nn.Linear(args.hidden_channels, d)
        # feat_recon_decoder = nn.Linear(args.hidden_channels, args.hidden_channels)

    if args.use_topo_recon_loss:
        topo_recon_decoder = InnerProductDecoder(hidden_dim=args.hidden_channels, output_dim=args.hidden_channels)

    if args.contrastive_recon:
        feat_recon_decoder = nn.Linear(args.hidden_channels, d)
        contrastive_feat_recon_decoder = InnerProductDecoder(hidden_dim=args.hidden_channels, output_dim=d)
    


    model = JointModel(encoder, vq, feat_recon_decoder, topo_recon_decoder, decoder, contrastive_feat_recon_decoder, contrastive_emb_recon_decoder, args.kmeans, args.encoder, loss_flags)

    if args.stage == 1:
        for param in model.vq.parameters():
            param.requires_grad = False
    elif args.stage == 2:
        for param in model.vq.parameters():
            param.requires_grad = True
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.001, weight_decay=0.00001)

    ## training loop
    for run in range(args.runs):
        x = dataset.graph['node_feat']
        edge_index = dataset.graph['edge_index']
        def feature_variance(x):
            return x.var(dim=0).mean().item()
        from sklearn.metrics.pairwise import cosine_similarity
        N = x.shape[0]
        pca_dim = pca_components_for_variance(x)
        avg_deg = average_degree(edge_index, x.size(0))
        avg_jaccard = jaccard_overlap(edge_index, x.size(0))

        print(f"PCA@95% = {pca_dim}, AvgDegree = {avg_deg:.2f}, AvgJaccard = {avg_jaccard:.4f}")

        def feature_similarity_entropy(x, num_sample=1000):
            x = x.cpu().numpy()
            N = x.shape[0]
            sims = []
            for _ in range(num_sample):
                i, j = np.random.randint(0, N, 2)
                sim = cosine_similarity(x[i:i+1], x[j:j+1])[0][0]
                sims.append(sim)

            # binning + entropy
            hist, _ = np.histogram(sims, bins=10, range=(0, 1), density=False)

            # normalize to probability
            total = hist.sum()
            probs = hist / total

            # filter out 0s to avoid log(0)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        deg = degree(edge_index[0], num_nodes=N)
        print(feature_variance(x), feature_similarity_entropy(x), deg.mean().item(), compute_graph_avg_jaccard(edge_index, N))
        aug_x, mask_feat = mask_feature(x, p=args.feat_p)
        aug_edge_index, mask_link = dropout_edge(
                edge_index, p=args.edge_p, force_undirected=False, training=True
            )
        mask_x, mask = mask_node_feature(x, p=args.feat_p)
        print(edge_index)
        x = x.to(device)
        edge_index = edge_index.to(device)
        aug_x = aug_x.to(device)
        aug_edge_index = aug_edge_index.to(device)
        model.to(device)
        # aug_graph = [aug_x, aug_edge_index, mask_feat, mask_link]
        aug_graph = [mask_x.to(device), aug_edge_index, mask.to(device), mask_link]

        if args.contrastive_recon:
            # model.prepare_nodewise_contrastive_tensors(aug_edge_index, x.size(0))
            model.prepare_nodewise_contrastive_tensors(aug_edge_index, x, x.size(0))

        stats_log = []

        for epoch in range(args.epochs):
            
            model.train()

            #epoch, x, edge_index, labels, criterion, opt, clf, model, device
            stats = pretrain_an_epoch(args, epoch, x, edge_index, aug_graph, optimizer, model)

            if epoch % 20 == 0 and args.stage == 1:
                save_path = osp.join(gnn_path, f'epoch_{epoch}.pt')
                torch.save(model.encoder.state_dict(), save_path)
                print('gnn saved !')

            stats_log.append(stats)
            # if epoch % 5 == 0:
            #     vq_path = osp.join('../GQT/pretrained_vq', f'vq_{args.dataset}_contrastive_recon_{args.contrastive_recon}_{epoch}.pt')
            #     encoder_path = osp.join('../GQT/pretrained_gnn', f'vq_{args.dataset}_contrastive_recon_{args.contrastive_recon}_{epoch}.pt')
            #     model.save_vq(vq_path)
            #     model.save_encoder(encoder_path)

        return stats_log
   

if __name__ == "__main__":
    stas = run()