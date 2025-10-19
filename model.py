import torch
from torch import nn
import torch.nn.functional as F
from modules.encoders import *
from transformers import BertTokenizer, BertConfig, T5Tokenizer, T5Config, BertModel
from peft import LoraConfig, TaskType, get_peft_model
from modules.xbert_1loc_1glob_temp_aware import XBertModel

class MMA_Bert(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(hp.bert_path)
        config = BertConfig.from_pretrained(hp.bert_path)
        if not hasattr(config,'TopK'):
            setattr(config,'TopK',hp.TopK)
        if not hasattr(config,'kernel_size'):
            setattr(config,'kernel_size',hp.kernel_size)
        if not hasattr(config,'rank'):
            setattr(config,'rank',hp.rank)
        if not hasattr(config,'audio_dim'):
            setattr(config,'audio_dim',hp.audio_dim)
        if not hasattr(config,'vision_dim'):
            setattr(config,'vision_dim',hp.vision_dim)
        if not hasattr(config,'start_fusion_layer'):
            setattr(config,'start_fusion_layer',hp.start_fusion_layer)
        self.rank = hp.rank
        peft_config = LoraConfig(inference_mode=False, r=hp.lora_rank, lora_alpha=32, lora_dropout=0.1)
        model = XBertModel.from_pretrained(hp.bert_path, config=config)
        self.model = get_peft_model(model, peft_config)  
        self.bert_model = self.model
        for n, p in self.bert_model.named_parameters():
            if 'adapter' in n:
                p.requires_grad=True
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, text, vision, audio):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        last_hidden_states, LBLoss, ep_d = self.bert_model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids,
                                            vision=vision,
                                            audio=audio)
        return last_hidden_states, LBLoss, ep_d

class MMA(nn.Module):
    def __init__(self,hp):
        super(MMA, self).__init__()
        self.model = MMA_Bert(hp)
        self.cls_head = SubNet(in_size=768, hidden_size=128, n_class=1, dropout=0.2)
        self.config = BertConfig.from_pretrained(hp.bert_path)
        
    def forward(self, vision, audio, text):   
        batch_size = text.shape[0]
        input_ids, input_mask = text[:,:,0].long(), text[:,:,1].float()
        hidden_states, LBLoss, ep_d = self.model(text, vision, audio)
        embedding = hidden_states[:,0,:]
        pred = self.cls_head(embedding)
        return pred, LBLoss
    
    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
