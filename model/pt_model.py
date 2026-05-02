from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv

from torch_geometric.utils import negative_sampling

EPS = 1e-15

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') +
                F.kl_div(q.log(), m, reduction='batchmean'))

def token_usage_entropy(soft_weights):
    # soft_weights: [1, N, K]
    token_probs = soft_weights.squeeze(0) 
    usage = token_probs.mean(dim=0)       
    entropy = -(usage * torch.log(usage + 1e-8)).sum()
    return entropy

def get_feat_negatives(x, k=5):
    """
    x: [N, d] node features
    Returns: [N, k] index of k most dissimilar nodes per node
    """
    x = F.normalize(x, dim=1)
    sim = torch.matmul(x, x.T)  # [N, N], cosine similarity
    neg_sim = -sim  # dissimilarity
    neg_idx = torch.topk(neg_sim, k=k, dim=1).indices  # top-k least similar
    return neg_idx

class JointModel(nn.Module):
    def __init__(self, encoder, vq, feat_recon_decoder, topo_recon_decoder, decoder, contrastive_feat_recon_decoder, contrastive_emb_recon_decoder, kmeans, encoder_type, loss_flags=None):
        super().__init__()
        self.encoder = encoder
        self.kmeans = kmeans
        self.sem_encoder = None
        if loss_flags is None:

            loss_flags = {
                'commit_loss': True,
                'emb_recon_loss': True,
                'feat_recon_loss': True,
                'topo_recon_loss': True,
                'contrastive_recon': True
            }

        self.emb_rec = loss_flags.get('emb_recon_loss', False)
        self.feat_rec = loss_flags.get('feat_recon_loss', False)
        self.topo_rec = loss_flags.get('topo_recon_loss', False)
        self.contrastive_recon = loss_flags.get('contrastive_recon', False)
        self.decoder = decoder
        self.encoder_type = encoder_type
        if self.kmeans:
            self.vq = vq

        if self.emb_rec:
            self.sem_encoder = deepcopy(self.encoder)
            self.sem_projector = nn.Linear(self.encoder.hidden_channels, self.encoder.hidden_channels)

        if self.feat_rec:
            self.feat_recon_decoder = feat_recon_decoder

        if self.topo_rec:
            self.topo_recon_decoder = topo_recon_decoder 

        if self.contrastive_recon:
            self.contrastive_recon_decoder = contrastive_feat_recon_decoder
            self.feat_recon_decoder = feat_recon_decoder

        # if self.contrastive_emb_recon:
        #     self.sem_encoder = deepcopy(self.encoder)
        #     self.sem_projector = nn.Linear(self.encoder.hidden_channels, self.encoder.hidden_channels)
        #     self.contrastive_emb_recon_decoder = contrastive_emb_recon_decoder
    def compute_feature_similarity_matrix(self, x):
        x = F.normalize(x, dim=-1)
        sim_matrix = x @ x.T  # [N, N]
        return sim_matrix
    
    def prepare_nodewise_contrastive_tensors(self, edge_index, x, num_nodes, k_pos=50, k_neg=50, topk_feat=20):
        import random
        import torch
        import torch.nn.functional as F
        from collections import defaultdict

        device = edge_index.device
        edge_index = edge_index.cpu()
        x_cpu = x.cpu()


        neighbors = defaultdict(set)
        for src, dst in zip(edge_index[0], edge_index[1]):
            i, j = src.item(), dst.item()
            neighbors[i].add(j)
            neighbors[j].add(i)


        x_norm = F.normalize(x_cpu, dim=-1)
        sim_matrix = x_norm @ x_norm.T  # [N, N]

        all_nodes = set(range(num_nodes))
        pos_index = []
        neg_index = []

        for i in range(num_nodes):
            pos_list_struct = list(neighbors[i]) 


            topk_sim = torch.topk(sim_matrix[i], topk_feat + 1).indices.tolist()
            pos_list_feat = [j for j in topk_sim if j != i and j not in neighbors[i]][:topk_feat]


            pos_candidates = list(set(pos_list_struct + pos_list_feat))
            if len(pos_candidates) == 0:
                pos_candidates = [i]

            pos_sample = random.choices(pos_candidates, k=k_pos) if len(pos_candidates) < k_pos else random.sample(pos_candidates, k=k_pos)


            exclude_set = set(pos_candidates + [i])
            neg_candidates = list(all_nodes - exclude_set)

            neg_sample = random.choices(neg_candidates, k=k_neg) if len(neg_candidates) < k_neg else random.sample(neg_candidates, k=k_neg)

            pos_index.append(pos_sample)
            neg_index.append(neg_sample)

        self.pos_index = torch.tensor(pos_index, device=device, dtype=torch.long)
        self.neg_index = torch.tensor(neg_index, device=device, dtype=torch.long)


    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def save_vq(self, path):
        torch.save(self.vq.state_dict(), path)

    # Reconstructing the tree representation
    def sem_recon_loss(self, emb, quantized, eta=1.0):

        z = emb
        h = self.sem_projector(quantized)

        z = F.normalize(z, dim=-1, p=2)  # N * D
        h = F.normalize(h, dim=-1, p=2)  # N * D

        loss = (1 - (z * h).sum(dim=-1)).pow_(eta)
        loss = loss.mean()

        return loss

    def ema_update_sem_encoder(self, decay=0.99):
        for param_q, param_k in zip(self.encoder.parameters(), self.sem_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1 - decay)


    def feat_recon_loss(self, z, x, mask_feat):
        return F.mse_loss(self.feat_recon_decoder(z)[~mask_feat], x[~mask_feat])
        # return F.mse_loss(z[~mask_feat], x[~mask_feat])
    
    
    def contrastive_loss_with_negatives_batched(self, z, x, same_token_mask, tau=0.2, beta=10.0, batch_size=52):
        total_loss = torch.tensor(0.0, device=z.device)
        total_examples = 0
        num_batches = (z.size(0) + batch_size - 1) // batch_size
        x = F.normalize(x, dim=-1)
        all_losses = []

        for i in range(0, z.size(0), batch_size):
            # Batch slice
            z_batch = z[i:i+batch_size]  # [b, d]
            z_batch = F.normalize(z_batch, dim=-1)
            pos_index_batch = self.pos_index[i:i+batch_size]
            neg_index_batch = self.neg_index[i:i+batch_size]

            x_pos_batch = x[pos_index_batch]   # [b, k, d]
            x_neg_batch = x[neg_index_batch]    # [b, k, d]

            # Normalize + similarity
            pos_sim = self.contrastive_recon_decoder.forward_pairwise_batch(z_batch, x_pos_batch, sigmoid=True) / tau
            neg_sim = self.contrastive_recon_decoder.forward_pairwise_batch(z_batch, x_neg_batch, sigmoid=True) / tau  # [N, k]
            
            w_pos = (~same_token_mask[i:i+batch_size]).float()            
            w_softneg = same_token_mask[i:i+batch_size].float() 

            numerator = torch.sum(torch.exp(pos_sim) * w_pos, dim=1)  # [N]
            # numerator = torch.sum(torch.exp(pos_sim), dim=1)
            numerator = torch.clamp(numerator, min=1e-8)

            softneg_part = beta * torch.sum(torch.exp(pos_sim) * w_softneg, dim=1)  # [N]
            neg_part = torch.sum(torch.exp(neg_sim), dim=1)  # [N]
            denom = softneg_part + neg_part + 1e-8

            loss = -torch.log(numerator / denom)
            loss_batch = loss/loss.size(0)
            loss_batch.backward
            total_loss += loss.detach().sum()
            total_examples += loss.size(0)
        return total_loss/total_examples
    
    def contrastive_loss_soft_weights_batched(self, soft_weights, same_token_mask, tau=0.2, beta=10.0, batch_size=512):
        token_probs = F.normalize(soft_weights.squeeze(0), dim=-1)  # [N, K]
        total_loss = torch.tensor(0.0, device=token_probs.device)
        total_examples = 0
        all_losses = []

        for i in range(0, token_probs.size(0), batch_size):
            pi = token_probs[i:i+batch_size]                          # [b, K]
            pos_idx = self.pos_index[i:i+batch_size]                  # [b, k]
            neg_idx = self.neg_index[i:i+batch_size]                  # [b, k]
            same_token = same_token_mask[i:i+batch_size]              # [b, k]

            pj_pos = token_probs[pos_idx]                             # [b, k, K]
            pj_neg = token_probs[neg_idx]                             # [b, k, K

            # Compute similarity: [b, k]
            pos_sim = torch.einsum('bd,bkd->bk', pi, pj_pos) / tau
            neg_sim = torch.einsum('bd,bkd->bk', pi, pj_neg) / tau

            w_pos = (~same_token).float()       
            w_softneg = same_token.float()        

            numerator = torch.sum(torch.exp(pos_sim) * w_pos, dim=1)  # [b]
            softneg_part = beta * torch.sum(torch.exp(pos_sim) * w_softneg, dim=1)  # [b]
            neg_part = torch.sum(torch.exp(neg_sim), dim=1)            # [b]

            denom = torch.clamp(numerator + softneg_part + neg_part + 1e-8, min=1e-8)
            loss = -torch.log(numerator / denom)

            # total_loss += loss.sum()
            # total_examples += loss.size(0)
            all_losses.append(loss.mean())
        # return total_loss / total_examples
        return torch.stack(all_losses).mean() 

    def structure_token_loss_2(self, soft_weights):
        # token_probs = soft_weights.squeeze(0)  # [N, K]
        # token_probs = F.normalize(token_probs, dim=-1)  # cosine-friendly
        # anchor = torch.arange(token_probs.size(0), device=token_probs.device)  # [N]
        # anchor = anchor.unsqueeze(1).expand_as(self.pos_index)  # [N, M]

        # # [N, M, K]
        # pi_anchor = token_probs[anchor]        
        # pi_pos = token_probs[self.pos_index]      
        # pi_neg = token_probs[self.neg_index]      

        # # pos_sim = (pi_anchor * pi_pos).sum(dim=-1)  # [N, M]
        # neg_sim = (pi_anchor * pi_neg).sum(dim=-1)  # [N, M]
        # # pos_dist = 1 - pos_sim
        # neg_dist = 1 - neg_sim

        loss = - 1 * token_usage_entropy(soft_weights)
        # loss.backward()
        return loss
    
    def structure_token_loss_multi_pos(self, soft_weights, temperature=0.1):

        token_probs = soft_weights.squeeze(0)  # [N, K]
        token_probs = F.normalize(token_probs, dim=-1)  # [N, K]

        N, K = token_probs.shape
        M = self.pos_index.size(1)

        anchor = torch.arange(N, device=token_probs.device).unsqueeze(1).expand(-1, M)  # [N, M]

        pi_anchor = token_probs[anchor]             # [N, M, K]
        pi_pos    = token_probs[self.pos_index]     # [N, M, K]
        pi_neg    = token_probs[self.neg_index]     # [N, M, K]


        pos_sim = (pi_anchor * pi_pos).sum(dim=-1) / temperature
        neg_sim = (pi_anchor * pi_neg).sum(dim=-1) / temperature


        pos_exp = torch.exp(pos_sim)                # [N, M]
        neg_exp = torch.exp(neg_sim)                # [N, M]

        numerator = pos_exp.sum(dim=1)              # [N]
        denominator = numerator + neg_exp.sum(dim=1)  # [N]

        loss = -torch.log(numerator / (denominator + 1e-8)).mean()
        return loss
    
    def structure_token_loss_multi_pos_minibatch(
    self,
    soft_weights,                # [1, N, K] or [N, K]
    opt,
    temperature=0.1,
    batch_size=256,
    optimizer=None,              # optional optimizer step per batch
    retain_graph=True            # default True since we'll backprop multiple times
):
        token_probs = soft_weights.squeeze(0) if soft_weights.dim() == 3 else soft_weights  # [N, K]
        token_probs = F.normalize(token_probs, dim=-1)
        N, K = token_probs.shape
        M = self.pos_index.size(1)

        total_loss = torch.tensor(0.0, device=token_probs.device)
        total_examples = 0
        all_losses = []

        for i in range(0, N, batch_size):
            b = min(batch_size, N - i)

            # Batch anchor
            pi = token_probs[i:i + b]                            # [b, K]

            # Batch pos/neg indices: [b, M]
            pos_idx = self.pos_index[i:i + b]
            neg_idx = self.neg_index[i:i + b]

            # Pos/neg token distributions: [b, M, K]
            pj_pos = token_probs[pos_idx]
            pj_neg = token_probs[neg_idx]

            pi_expand = pi.unsqueeze(1).expand(-1, M, -1)        # [b, M, K]

            # Cosine similarity: [b, M]
            pos_sim = (pi_expand * pj_pos).sum(dim=-1) / temperature
            neg_sim = (pi_expand * pj_neg).sum(dim=-1) / temperature

            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)

            numerator = pos_exp.sum(dim=1)              # [b]
            denominator = numerator + neg_exp.sum(dim=1) + 1e-8  # [b]

            batch_loss = -torch.log(numerator / denominator)     # [b]
            batch_loss = batch_loss.mean()

            all_losses.append(batch_loss.detach().item())
            total_loss += batch_loss.detach().item()  * b
            total_examples += b

            batch_loss.backward(retain_graph=retain_graph)

        return total_loss / total_examples



    # Reconstructing tree structure, similar to graph reconstruction.
    def topo_recon_loss(self, z, pos_edge_index, neg_edge_index=None, ratio=1.0):

        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            # Randomly sample positive edges
            num_pos_edges = int(pos_edge_index.size(1) * ratio)
            num_pos_edges = max(num_pos_edges, 1)
            perm = torch.randperm(pos_edge_index.size(1))
            perm = perm[:num_pos_edges]
            pos_edge_index = pos_edge_index[:, perm]

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        # pos_loss = -torch.log(self.topo_recon_decoder(self.decoder(z), pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.topo_recon_decoder(self.decoder(z), neg_edge_index, sigmoid=True) + EPS).mean()

        return neg_loss
    

    def forward(self, x, edge_index, aug_g, stage, temperature, opt, topo_recon_ratio=1.0, pretrain=True):
        aug_x, aug_edge_index, mask_feat, mask_link = aug_g[0], aug_g[1], aug_g[2], aug_g[3]
        pos_edge_index = edge_index[:, ~mask_link]
        if self.encoder_type == 'gnn':
            z = self.encoder(aug_x, aug_edge_index)
        elif self.encoder_type == 'mlp':
            z = self.encoder(aug_x)
        losses = {}
        losses['commit_loss'] = torch.tensor(0.0, device=z.device)
        losses['emb_recon_loss'] = torch.tensor(0.0, device=z.device)
        losses['feat_recon_loss'] = torch.tensor(0.0, device=z.device)
        losses['topo_recon_loss'] = torch.tensor(0.0, device=z.device)
        losses['topo_contrastive'] = torch.tensor(0.0, device=z.device)
        
        if self.kmeans and pretrain is True and stage == 2:
            #quantize, embed_ind, loss, orig_quantize
            quantized, embed_ind, commit_loss, original_quantized, soft_weights = self.vq(z)
            # quantized, embed_ind, commit_loss = self.vq(z)
            losses['commit_loss'] = commit_loss
        
        if self.emb_rec and (self.sem_encoder is not None):
            emb_reconstruct_loss = self.sem_recon_loss(z, quantized)
            losses['emb_recon_loss'] = emb_reconstruct_loss
        
        if self.feat_rec and (self.feat_recon_decoder is not None):
            # feat_reconstruct_loss = self.feat_recon_loss(quantized, x, mask_feat)
            if stage == 2:
                # feat_reconstruct_loss = self.feat_recon_loss(torch.cat([quantized, pos_enc], dim=-1), aug_x, mask_feat)
                # feat_reconstruct_loss = self.feat_recon_loss(quantized + pos_enc, aug_x, mask_feat)
                feat_reconstruct_loss = self.feat_recon_loss(quantized, aug_x, mask_feat)
                # feat_reconstruct_loss = self.feat_recon_loss_cosine(quantized, aug_x, mask_feat)
            elif stage == 1:
                feat_reconstruct_loss = self.feat_recon_loss(z, aug_x, mask_feat)
            losses['feat_recon_loss'] = feat_reconstruct_loss

        if self.topo_rec and (self.topo_recon_decoder is not None):
            if stage == 2:
                # topo_recon_loss = self.topo_recon_loss(torch.cat([quantized, pos_enc], dim=-1), edge_index, ratio=topo_recon_ratio)
                # topo_recon_loss = self.topo_recon_loss(quantized + pos_enc, edge_index, ratio=topo_recon_ratio)
                topo_recon_loss = self.topo_recon_loss(quantized, edge_index, ratio=topo_recon_ratio)
            elif stage == 1:
                topo_recon_loss = self.topo_recon_loss(z, edge_index, ratio=topo_recon_ratio)
            # topo_recon_loss = self.mse_link_recon_loss(quantized, pos_edge_index)
            losses['topo_recon_loss'] = topo_recon_loss

        if (self.contrastive_recon 
            and self.contrastive_recon_decoder is not None
            and self.feat_recon_decoder is not None):
            token_diversity_loss = self.structure_token_loss_multi_pos_minibatch(soft_weights, opt)
            # losses['topo_contrastive'] = 0.1 * token_diversity_loss
            # losses['topo_contrastive'] = 0.1 * token_diversity_loss
            # print(token_diversity_loss.size())

        if stage == 1:
            return z, losses
        elif stage == 2:
            return z, quantized, embed_ind, losses