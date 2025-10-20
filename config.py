import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
import string
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = ''
assert(word_emb_path is not None)
username = Path.home().name


project_dir = Path(__file__).resolve().parent.parent


sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = { 
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

def get_args():
    parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')

    # Dataset
    parser.add_argument('--dataset', type=str, default='mosei', choices=['mosi','mosei'],
                        help='dataset to use (default: mosi)')
    parser.add_argument('--data_path', type=str, default='aligned_50.pkl', 
                        help='path for storing the dataset')
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased',
                        help='path for storing the dataset')
    # Dropouts
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--TopK', type=int, default=3, help='K')
    # parser.add_argument('--bottleneck', type=int, default=64, help='bottleneck of parallel adapters')
    parser.add_argument('--rank', type=int, default=32, help='rank of parallel adapters')
    parser.add_argument('--lora_rank', type=int, default=32, help='rank of lora')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel_size')
    parser.add_argument('--lb_loss_hyp', type=int, default=0.001,
                        help='kernel_size')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_lora', type=float, default=1e-3,
                        help='initial learning rate for lora parameters (default: 1e-3)')
    parser.add_argument('--lr_adapter', type=float, default=1e-3,
                        help='initial learning rate for lora parameters (default: 1e-3)')
    
    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main optimizer')
    parser.add_argument('--weight_decay_lora', type=float, default=1e-4,
                        help='L2 penalty factor of the lora optimizer')
    parser.add_argument('--weight_decay_adapter', type=float, default=1e-4,
                        help='L2 penalty factor of the adapter optimizer')
        
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer to use (default: AdamW)')
    parser.add_argument('--audio_dim', type=int, default=5,
                        help='audio feature dim')
    parser.add_argument('--vision_dim', type=int, default=20,
                        help='vision feature dim')
    parser.add_argument('--num_epochs', type=int, default=25,  
                        help='number of epochs (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')
    parser.add_argument('--when', type=int, default=9,
                        help='learning rate decay')
    parser.add_argument('--start_fusion_layer', type=int, default=0,
                        help='start_fusion_layer')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=7777, 
                        help='random seed')
    
    args = parser.parse_args()
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32):
    config = Config(data=dataset, mode=mode)
    
    config.dataset = dataset
    config.batch_size = batch_size

    return config
