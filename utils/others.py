import os.path as osp
from pathlib import Path
import numpy as np
import torch

from model.encoder import GNN
from model.vq import VectorQuantize

def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_device_from_model(model):
    return next(model.parameters()).device

def load_params(model, path):
    if isinstance(model, GNN):
        model.load_state_dict(torch.load(path))
    elif isinstance(model, VectorQuantize):
        z = torch.randn(100, model.dim)
        model(z)
        model.load_state_dict(torch.load(path))
    return model


def active_code(encoder, vq, data):
    z = encoder(data.x, data.edge_index, data.edge_attr)
    _, indices, _, _ = vq(z)
    codebook_size = vq.codebook_size
    codebook_head = vq.heads
    return indices.unique(), indices.unique().numel() / (codebook_size * codebook_head)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def visualize(embedding, label=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label, cmap='tab10')
    plt.show()


def load_vq_from_codebook(
    codebook_path: str,
    model_path: str,
    args,
    device='cuda'
):


    checkpoint = torch.load(codebook_path)
    embed_ind = checkpoint['embed_ind'].to(device)   # shape: [N, H]
    codebook = checkpoint['codebook'].to(device)     # shape: [H, C, D] or [C, D]


    vq = VectorQuantize(
        separate_codebook_per_head=True, 
        use_cosine_sim = args.use_cosine_sim, 
        heads=args.num_heads, 
        dim=args.hidden_channels, 
        codebook_size=args.num_codes,
        learnable_codebook=args.learnable_codebook,   
        ema_update=args.ema_update            
    ).to(device)

    vq.load_state_dict(torch.load(model_path))

    if not args.learnable_codebook:
        vq.eval()
        for p in vq.parameters():
            print(p)
            p.requires_grad = False


    vq.codebook = codebook

    return vq, embed_ind  # shape: [N, D]



def sample_proto_instances(labels, split, num_instances_per_class=10):
    y = labels.cpu().numpy()
    target_y = y[split]
    classes = np.unique(target_y)

    class_index = []
    for i in classes:
        c_i = np.where(y == i)[0]
        c_i = np.intersect1d(c_i, split)
        class_index.append(c_i)

    proto_idx = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        proto_idx = np.concatenate((proto_idx, idx[:num_instances_per_class]))

    return proto_idx.astype(int)


def sample_proto_instances_for_graph(labels, split, num_instances_per_class=10):
    y = labels
    ndim = y.ndim
    if ndim == 1:
        y = y.reshape(-1, 1)

    # Map class and instance indices

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    target_y = y[split]
    task_list = target_y.shape[1]

    # class_index_pos = {}
    # class_index_neg = {}
    task_index_pos, task_index_neg = [], []
    for i in range(task_list):
        c_i = np.where(y[:, i] == 1)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_pos.append(c_i)

        c_i = np.where(y[:, i] == 0)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_neg.append(c_i)

    assert len(task_index_pos) == len(task_index_neg)

    # Randomly select instances for each task

    proto_idx, proto_labels = {}, {}
    for task, (idx_pos, idx_neg) in enumerate(zip(task_index_pos, task_index_neg)):
        tmp_proto_idx, tmp_labels = np.array([]), np.array([])

        # Randomly select instance for the task

        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)
        idx_pos = idx_pos[:num_instances_per_class]
        idx_neg = idx_neg[:num_instances_per_class]

        # Store the randomly selected instances

        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_pos))
        tmp_labels = np.concatenate((tmp_labels, np.ones(len(idx_pos))))
        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_neg))
        tmp_labels = np.concatenate((tmp_labels, np.zeros(len(idx_neg))))

        proto_idx[task] = tmp_proto_idx.astype(int)
        proto_labels[task] = tmp_labels.astype(int)

    return proto_idx, proto_labels


def mask2idx(mask):
    return torch.where(mask == True)[0]


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask
