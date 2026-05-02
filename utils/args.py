import argparse

import os
import os.path as osp

def get_args_pretrain():
    parser = argparse.ArgumentParser('Pretrain')

    parser.add_argument('--dataset', type=str, default='coauthor-cs')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')

    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    # model
    parser.add_argument('--method', type=str, default='gat')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=4,
                        help='number of layers for local attention')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='ckpt/pretrained_model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')
        
    parser.add_argument('--kmeans', type=int,
                        default=1)
    parser.add_argument('--num_codes', type=int,
                        default=512)
    parser.add_argument('--norm_type', type=str, default='none')

    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--num_id', type=int, default=15)

    # pre-training task parameters
    parser.add_argument('--feat_p', type=float, default=0.8)
    parser.add_argument('--edge_p', type=float, default=0.2)
    parser.add_argument('--use_commit_loss', type=bool, default=True)
    parser.add_argument('--use_emb_recon_loss', type=bool, default=False)
    parser.add_argument('--use_feat_recon_loss', type=bool, default=True)
    parser.add_argument('--use_topo_recon_loss', type=bool, default=True)
    parser.add_argument('--contrastive_recon', type=bool, default=True)
    parser.add_argument('--ema_update', type=str, default='False')
    parser.add_argument('--pretrained_gnn', type=str, default='False')
    parser.add_argument('--encoder', type=str, default='gnn')
    parser.add_argument('--sim', type=str, default='False')

    parser.add_argument('--commit_weight', type=float, default=1.0)
    parser.add_argument('--task', type=str, default='Node')
    parser.add_argument('--stage', type=int, default=2)   
    parser.add_argument('--pos_dim', type=int, default=256)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args_pretrain()
    os.makedirs(osp.join(args.model_dir, f'{args.dataset}'), exist_ok=True)
    print(args.model_dir)