import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import *
import logging
# import wandb

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params  # args
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.is_train = is_train
        self.model = model = MMA(hp)
        model.print_trainable_parameters()
        # Training hyperarams

        self.update_batch = hp.update_batch  

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        else:
            self.device = torch.device("cpu")

        self.criterion = nn.L1Loss(reduction="mean")  
        # self.criterion = nn.HuberLoss(reduction='mean')
        
        # optimizer
        self.optimizer={}

        if self.is_train:
            # mmilb_param = []
            main_param = []
            bert_param = []
            lora_param = []
            adapter_param = []

            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'lora_' in name: 
                        lora_param.append(p)
                    elif 'adapter' in name:
                        adapter_param.append(p)
                    else: 
                        main_param.append(p)
                
            for p in main_param:
                if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p) 

        
        optimizer_main_group = [
            {'params': lora_param, 'weight_decay': hp.weight_decay_lora, 'lr': hp.lr_lora},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main},
            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter}
        ]


        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        self.scheduler_main = StepLR(self.optimizer_main, step_size=self.hp.when, gamma=0.1)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        model = model.cuda()
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main

        # criterion for downstream task
        criterion = self.criterion

        def train(model, optimizer, criterion):
            epoch_loss = 0

            model.train()
            proc_loss, proc_size = 0, 0
            main_loss = 0.0
            multi_con_loss = 0.0
            exp_frequencies = []
            left_batch = self.update_batch
            expert_distribution = []
            for i_batch, batch_data in enumerate(self.train_loader):
                raw_text = batch_data['raw_text']
                vision = batch_data['vision']
                audio = batch_data['audio']
                text = batch_data['text']
                y = batch_data['labels']['M']
                #print(f"Vision data shape: {vision.shape}, Audio data shape: {audio.shape}, Text data shape: {text.shape}, Label shape: {y.shape}")
                
                model.zero_grad()
                with torch.cuda.device(0):
                    vision, audio, text, y = vision.cuda(), audio.cuda(), text.cuda(), y.cuda()
                
                batch_size = y.size(0)               
                preds, LBLoss, exp_freqs = model(vision, audio, text)
                exp_frequencies.append(exp_freqs.detach().cpu())
                y_loss = criterion(preds, y)
                loss = y_loss + 0.001*LBLoss
                loss.backward()
                
                # -------------------------------------------------------- #
                left_batch -= 1  
                if left_batch == 0: 
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)  
                    optimizer.step()
                # -------------------------------------------------------- #

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                main_loss +=y_loss.item() * batch_size
                
            exp_frequencies = torch.cat(exp_frequencies, dim=0)     
            return epoch_loss, exp_frequencies

        def evaluate(model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            main_loss = 0.0        
            results = []
            truths = []
            exp_frequencies = []
            with torch.no_grad():
                for batch_data in loader:
                    raw_text = batch_data['raw_text']
                    vision = batch_data['vision']
                    audio = batch_data['audio']
                    text = batch_data['text']
                    y = batch_data['labels']['M']

                    with torch.cuda.device(0):
                        vision, audio, text, y = vision.cuda(), audio.cuda(), text.cuda(), y.cuda()
                    batch_size = y.size(0)    
                    preds , _, exp_freqs = model(vision, audio, text)
                    exp_frequencies.append(exp_freqs.detach().cpu())         
                    criterion = nn.L1Loss()
                    main_loss += criterion(preds, y).item() * batch_size   
                    results.append(preds)
                    truths.append(y)           
            
            exp_frequencies = torch.cat(exp_frequencies, dim=0)
            results = torch.cat(results)
            truths = torch.cat(truths)
            test_preds = results.view(-1).cpu().detach().numpy()
            test_truth = truths.view(-1).cpu().detach().numpy()
            avg_main_loss =  np.mean(np.absolute(test_preds - test_truth))
            return avg_main_loss, results, truths, exp_frequencies

        best_valid = 1e8
        patience = self.hp.patience
        all_train_exp_freqs = []
        all_val_exp_freqs = []
        all_test_exp_freqs = []
        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            logging.info(f'epoch {epoch}:')

            self.epoch = epoch

            train_main_loss, train_exp_freqs = train(model, optimizer_main, criterion)  
            val_loss, results_val, truths_val, val_exp_freqs = evaluate(model, criterion, test=False) 
            test_loss, results, truths, test_exp_freqs = evaluate(model, criterion, test=True)  

            all_train_exp_freqs.append(train_exp_freqs)
            all_val_exp_freqs.append(val_exp_freqs)
            all_test_exp_freqs.append(test_exp_freqs)

            test_preds = results.view(-1).cpu().detach().numpy()
            test_truth = truths.view(-1).cpu().detach().numpy()
            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  
            binary_truth_non0 = test_truth[non_zeros] > 0
            binary_preds_non0 = test_preds[non_zeros] > 0
            acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

            val_preds = results_val.view(-1).cpu().detach().numpy()
            val_truth = truths_val.view(-1).cpu().detach().numpy()
            non_zeros_val = np.array([i for i, e in enumerate(val_truth) if e != 0])
            binary_truth_non0_val = val_truth[non_zeros_val] > 0
            binary_preds_non0_val = val_preds[non_zeros_val] > 0
            acc_2_non0_val = accuracy_score(binary_truth_non0_val, binary_preds_non0_val)
            
            end = time.time()
            duration = end-start
            # scheduler_main.step(val_loss)    # Decay learning rate by validation loss
            
            scheduler_main.step()
            learning_rate = optimizer_main.state_dict()['param_groups'][0]['lr']
            
            # logging.info(f'learnng rate: {learning_rate}')
            logging.info("-"*50)
            logging.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            logging.info("-"*50)

            # print(f'training on epoch {epoch}')
            # print(f'learnng rate: {learning_rate}')
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if self.hp.dataset in ["mosei_senti", "mosei"]:
                eval_mosei_senti(results, truths, True)
            elif self.hp.dataset == 'mosi':
                eval_mosi(results, truths, True)
            print(f"Mean Expert Usage\n\tTrain: {[round(freq, 3) for freq in train_exp_freqs.mean(dim=0).tolist()]}\n\tVal: {[round(freq, 3) for freq in val_exp_freqs.mean(dim=0).tolist()]}\n\tTest: {[round(freq, 3) for freq in test_exp_freqs.mean(dim=0).tolist()]}")
            if val_loss < best_valid:
                # model.model.save()
               
                patience = self.hp.patience
                best_valid = val_loss
                best_epoch = epoch
                best_mae = test_loss
                best_results = results
                best_truths = truths

            else:
                patience -= 1
                if patience == 0:
                    break

        # print(f'Best epoch: {best_epoch}')
        logging.info(f'Best epoch: {best_epoch}')

        if self.hp.dataset in ["mosei_senti", "mosei"]:
            best_dict = eval_mosei_senti(best_results, best_truths, True)  
        elif self.hp.dataset == 'mosi':
            best_dict = eval_mosi(best_results, best_truths, True)
        sys.stdout.flush() 
        return best_dict