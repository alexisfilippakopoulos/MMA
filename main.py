import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU

import torch
import argparse
import numpy as np
import logging
from dataset import *
from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from utils.logs import set_arg_log

logging.getLogger ().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, filename='MMA.log')

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor') 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(seed) 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False  

        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())  
    
    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    dataLoader = MMDataLoader(args)

    train_loader = dataLoader['train']
    valid_loader = dataLoader['valid']
    test_loader = dataLoader['test']

    torch.autograd.set_detect_anomaly(True)
    solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader)

    logging.info(f'Runing code on the {args.dataset} dataset.')
    set_arg_log(args)
    best_dict = solver.train_and_eval()

    logging.info(f'Training complete')
    logging.info('--'*50)
    logging.info('\n')