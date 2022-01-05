from __future__ import division
from __future__ import print_function
import argparse
from utils import save_res, set_seed
from trainer import Trainer
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "model params")
    parser.add_argument("--gpu", type=int, default=0,
                            help="choose which GPU")
    parser.add_argument("--dataset", "-d", type=str, default='twitter',
                            help="choose the dataset: 'snippets' or 'twitter'")
    parser.add_argument("--file_dir", "-f_dir", type=str, default='./',
                            help="choose the file directory")
    parser.add_argument("--data_path", "-d_path", type=str, default='./data/',
                            help="choose the data path if necessary")
    parser.add_argument("--save_path", type=str, default="./",
                            help="save path")
    parser.add_argument("--save_name", type=str, default="./result.json",
                            help="save name")
    parser.add_argument('--disable_cuda', action='store_true',
                            help='disable CUDA')
    parser.add_argument("--seed", type=int, default=111, 
                            help="seeds for random initial")
    parser.add_argument("--hidden_size", type=int, default=200, 
                            help="hidden size")                        
    parser.add_argument("--threshold", type=float, default=2.5,
                            help="threshold for graph construction")
    parser.add_argument("--lr", type=float, default=1e-3,
                            help="learning rate of the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0,
                            help="adjust the learning rate via epochs")
    parser.add_argument("--drop_out", type=float, default=0.5,
                            help="dropout rate")
    parser.add_argument("--max_epoch", type=int, default=1000,
                            help="max numer of epochs")
    parser.add_argument("--type_num_node", type=list, default=['query', 'tag', 'word', 'entity'],
                            help="type number of the nodes (a list)")
    parser.add_argument("--concat_word_emb", type=bool, default=True,
                            help="concat word embedding with pretrained model")
    params = parser.parse_args()
    params.data_path = params.data_path + './{}_data/'.format(params.dataset)
    params.save_name = params.save_path + './result_{}.json'.format(params.dataset)
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    if params.dataset == 'snippets':
        params.seed = 120
        params.threshold = 2.7
        params.weight_decay = 0.001
    set_seed(params.seed)
    trainer = Trainer(params)
    test_acc,best_f1 = trainer.train()
    save_res(params, test_acc, best_f1)
    del trainer