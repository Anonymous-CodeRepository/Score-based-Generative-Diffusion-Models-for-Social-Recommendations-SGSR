# source /home/lkcy/anaconda3/bin/activate pytorch
# python main.py --dataset ciao --ciao 1
import torch
import argparse
import yaml
import os

import sys

def load_arguments_from_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='available datasets: [lastfm]')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epoches', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--topks', nargs='?',default="[20]",help="@k test list")
parser.add_argument('--M', type=int, default=10)
parser.add_argument('--N', type=int, default=10)
parser.add_argument('--epsilon', type=int, default=10)
parser.add_argument('--R', type=float, default=0.5)

parser.add_argument('--emb_dim', type=int, default=16, help='the dimension of the embedding vector')
parser.add_argument('--ui_n_layers', type=int, default=3, help='number of layers for interaction encoder')
parser.add_argument('--ui_dropout', type=bool, default=False)
parser.add_argument('--keep_prob_ui', type=float, default=0.9, help='keep probability for social')
parser.add_argument('--keep_prob', type=float, default=0.9, help='keep probability for social')

parser.add_argument('--ui_encoder', type=str, default='none', help='ui_encoder')
parser.add_argument('--social_encoder', type=str, default='none', help='social encoder')
parser.add_argument('--social_n_layers', type=int, default=3, help='number of layers for social encoder')
parser.add_argument('--condition_n_layers', type=int, default=3, help='number of layers for social encoder')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for bpr')
parser.add_argument('--weight_decay_social', type=float, default=0.0, help='weight decay for bpr')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for ui encoder')
# parser.add_argument('--wd', type=float, default=0.0, help='weight decay for opt')
parser.add_argument('--ema', type=float, default=0.999, help='')
parser.add_argument('--grad_norm', type=float, default=1.0, help='')

parser.add_argument('--sde', type=str, default='VP', help='type of diffusion model')
parser.add_argument('--beta_min', type=float, default=0.1)
parser.add_argument('--beta_max', type=float, default=1.0)
parser.add_argument('--continuous', type=bool, default=False)
parser.add_argument('--reduce_mean', type=bool, default=True)

parser.add_argument('--predictor', type=str, default='none', help='PC predictor')
parser.add_argument('--corrector', type=str, default='none', help='PC corrector')
parser.add_argument('--snr', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--ssl_wd', type=float, default=0.01)
parser.add_argument('--interOrIntra', type=str, default='inter')

parser.add_argument('--test_mode', type=int, default=1)
parser.add_argument('--core', type=int, default=0)

args = parser.parse_args()

device = (
        torch.device("cuda:"+str(args.core))
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

config = load_arguments_from_yaml(f'configures/{args.dataset}.yaml')
for arg, value in config.items():
    setattr(args, arg, value)

dataset = args.dataset
seed = args.seed

keep_prob = args.keep_prob

config_sde = {}
config_sde['beta_min'] = args.beta_min
config_sde['beta_max'] = args.beta_max
config_sde['num_scale'] = args.num_scale  # T

topks = eval(args.topks)
