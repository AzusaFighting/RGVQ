import numpy as np

import torch
from torch_geometric.loader import NeighborLoader

from utils.others import get_device_from_model, sample_proto_instances, mask2idx
from utils.eval import evaluate, task2metric


def ft_node(model, dataset, loader, optimizer, split, args):
    model.train()

    device = get_device_from_model(model)
    num_classes = model.num_classes

    use_proto_clf =  args.proto_clf
    use_lin_clf = args.lin_clf
    use_z_in_predict = args.use_z_in_predict
    lambda_act = args.lambda_act
    lambda_proto = 1.0 - lambda_act

    proto_loss = torch.tensor(0.0).to(device)
    act_loss = torch.tensor(0.0).to(device)

    mini_batch = loader is not None

    if not mini_batch:
        # Encode
        x = dataset.graph['node_feat'].to(device)
        edge_index = dataset.graph['edge_index'].to(device)
        y = dataset.label.to(device)

        z = model.encode(x, edge_index)
        train_mask = split["train"]
        z_train, y_train = z[train_mask], y[train_mask]

        if use_proto_clf:
            # Compute Prototypes
            code_train, commit_loss = model.get_codes(z_train, use_orig_codes=True)

            proto_emb = model.get_class_prototypes(code_train, y_train, num_classes).detach() #[h, c, d]
            query_emb = z_train if use_z_in_predict else code_train

            # Compute Losses

            proto_loss = model.compute_proto_loss(query_emb, proto_emb, y_train) * lambda_proto

        if use_lin_clf:
            act_loss = model.compute_activation_loss(z_train, y_train) * lambda_act

        loss = proto_loss + act_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
        'proto_loss': proto_loss.item(),
        'act_loss': act_loss.item(),
        'loss': loss.item(),
    }
        

# Evaluation
def eval_node(model, dataset, loader, split, args):
    model.eval()

    device = get_device_from_model(model)
    setting = args.setting
    labels = dataset.label

    num_classes = model.num_classes

    use_proto_clf =  args.proto_clf
    use_lin_clf = args.lin_clf
    use_z_in_predict = args.use_z_in_predict
    lambda_act = args.lambda_act
    lambda_proto = 1.0 - lambda_act

    pred_proto = 0
    pred_lin = 0

    mini_batch = loader is not None
    if not mini_batch:
        # Encode
        x = dataset.graph['node_feat'].to(device)
        edge_index = dataset.graph['edge_index'].to(device)
        y = dataset.label.to(device)

        z = model.encode(x, edge_index)

        if setting == "standard":

            if use_proto_clf:
                # Compute Prototypes
                train_mask = split["train"]
                code, _ = model.get_codes(z, use_orig_codes=True)
                code_train, y_train = code[train_mask], y[train_mask]

                proto_emb = model.get_class_prototypes(code_train, y_train, num_classes).detach()
                query_emb = z if model.use_z_in_predict else code

                pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

            if use_lin_clf:
                pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            # Evaluate
            train_mask, val_mask, test_mask = split["train"], split["valid"], split["test"]
            train_value = evaluate(pred, y, train_mask, args)
            val_value = evaluate(pred, y, val_mask, args)
            test_value = evaluate(pred, y, test_mask, args)

            return {
                'train': train_value,
                'val': val_value,
                'test': test_value,
                'metric': task2metric[args.task]
            }

        elif setting == "few_shot":
            n_task = len(split["valid"]["support"])
            train_values, val_values, test_values = [], [], []

            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                code, _ = model.get_codes(z, use_orig_codes=True)
                code_support, y_support = code[s_mask], y[s_mask]
                z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]


                if use_proto_clf:
                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes).detach()
                    query_emb = z_query if use_z_in_predict else code_query

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate

                value = evaluate(pred, y_query, mask=None, args=args)
                train_values.append(value)
                val_values.append(value)

            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]
                
                code, _ = model.get_codes(z, use_orig_codes=True)
                code_support, y_support = code[s_mask], y[s_mask]
                z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]


                if use_proto_clf:
                    # Compute Prototypes
                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes).detach()

                    query_emb = z_query if model.use_z_in_predict else code_query

                    # Compute logits

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate
                value = evaluate(pred, y_query, mask=None, args=args)
                test_values.append(value)

            return {
                'train': np.mean(train_values),
                'val': np.mean(val_values),
                'test': np.mean(test_values),
                'metric': task2metric[args.task]
            }

    else:
        # The standard setting and the remaining settings are handled differently
        if setting == "standard":

            if use_proto_clf:
                # Define Prototype Loader
                # Prototype instance sampling only for standard setting
                proto_idx = sample_proto_instances(labels, mask2idx(split["train"]),
                                                   num_instances_per_class=model.num_instances_per_class)
                proto_loader = NeighborLoader(
                    dataset,
                    num_neighbors=args.num_neighbors,
                    input_nodes=proto_idx,
                    batch_size=256,
                    num_workers=8,
                )

                # Encode Prototypes

                code_list, y_list = [], []
                for batch in proto_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_list.append(code.detach())
                    y_list.append(y)

                code = torch.cat(code_list, dim=0)
                y = torch.cat(y_list, dim=0)
                proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

            # Do Prediction

            pred_list, y_list = [], []
            for batch in loader:
                batch = batch.to(device)
                bs = batch.batch_size

                # Encode

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]

                y = batch.y[:bs]
                z = model.encode(x, edge_index, edge_attr)[:bs]

                if use_proto_clf:
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    query_emb = z if model.use_z_in_predict else code
                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                pred_list.append(pred.detach())
                y_list.append(y)

            pred = torch.cat(pred_list, dim=0)
            y = torch.cat(y_list, dim=0)

            train_mask, val_mask, test_mask = split["train"], split["valid"], split["test"]
            train_value = evaluate(pred, y, train_mask, args)
            val_value = evaluate(pred, y, val_mask, args)
            test_value = evaluate(pred, y, test_mask, args)

            return {
                'train': train_value,
                'val': val_value,
                'test': test_value,
                'metric': task2metric[args.task]
            }

        elif setting == "few_shot":
            n_task = len(split["valid"]["support"])
            train_values, val_values, test_values = [], [], []

            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                if use_proto_clf:
                    # Define Loaders for Support and Query Sets
                    # Prototype loader for support set
                    # Query loader for query set
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=args.num_neighbors,
                        input_nodes=mask2idx(s_mask),
                        batch_size=256,
                        num_workers=8,
                    )
                    query_loader = NeighborLoader(
                        dataset,
                        num_neighbors=args.num_neighbors,
                        input_nodes=mask2idx(q_mask),
                        batch_size=256,
                        num_workers=8,
                    )

                    # Construct Prototypes based on Support Set

                    code_list, y_list = [], []
                    for batch in proto_loader:
                        batch = batch.to(device)
                        bs = batch.batch_size

                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]

                        y = batch.y[:bs]
                        z = model.encode(x, edge_index, edge_attr)[:bs]

                        code, _ = model.get_codes(z, use_orig_codes=True)
                        code_list.append(code.detach())
                        y_list.append(y)
                    code = torch.cat(code_list, dim=0)
                    y = torch.cat(y_list, dim=0)

                    proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

                # Compute logits

                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    if use_proto_clf:
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        query_emb = z if model.use_z_in_predict else code
                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                    pred_list.append(pred.detach())
                    y_list.append(y)

                pred = torch.cat(pred_list, dim=0)
                y = torch.cat(y_list, dim=0)

                value = evaluate(pred, y, mask=None, args=args)
                train_values.append(value)
                val_values.append(value)

            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                if use_proto_clf:
                    # Define Loaders for Support and Query Sets
                    # Prototype loader for support set
                    # Query loader for query set
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=args.num_neighbors,
                        input_nodes=mask2idx(s_mask),
                        batch_size=256,
                        num_workers=8,
                    )
                    query_loader = NeighborLoader(
                        dataset,
                        num_neighbors=args.num_neighbors,
                        input_nodes=mask2idx(q_mask),
                        batch_size=256,
                        num_workers=8,
                    )

                    # Construct Prototypes based on Support Set

                    code_list, y_list = [], []
                    for batch in proto_loader:
                        batch = batch.to(device)
                        bs = batch.batch_size

                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]

                        y = batch.y[:bs]
                        z = model.encode(x, edge_index, edge_attr)[:bs]

                        code, _ = model.get_codes(z, use_orig_codes=True)
                        code_list.append(code.detach())
                        y_list.append(y)
                    code = torch.cat(code_list, dim=0)
                    y = torch.cat(y_list, dim=0)

                    proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

                # Compute logits

                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    if use_proto_clf:
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        query_emb = z if model.use_z_in_predict else code
                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)

                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                    pred_list.append(pred.detach())
                    y_list.append(y)

                pred = torch.cat(pred_list, dim=0)
                y = torch.cat(y_list, dim=0)

                value = evaluate(pred, y, mask=None, args=args)
                test_values.append(value)

            return {
                'train': np.mean(train_values),
                'val': np.mean(val_values),
                'test': np.mean(test_values),
                'metric': task2metric[args.task]
            }
