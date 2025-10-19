# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import BertConfig
from modules.adapters import *
from modules.loss import *
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]




class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs.
            token_type_ids (torch.LongTensor, optional): Token type IDs for segment embeddings.
            position_ids (torch.LongTensor, optional): Positional IDs.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings.
            past_key_values_length (int, optional): Length of past key values for incremental decoding. Defaults to 0.

        Returns:
            torch.Tensor: The combined embeddings.
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """
    Standard BERT self-attention mechanism.

    This module computes scaled dot-product attention with multiple heads. It includes
    linear projections for query, key, and value, and supports relative positional embeddings.
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    """
    The output sub-layer for the BERT self-attention mechanism.

    It applies a dense layer, dropout, and a residual connection with layer normalization
    to the output of the self-attention module.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    A complete BERT attention block, combining self-attention and self-output sub-layers.
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    """
    The intermediate feed-forward layer in a BERT transformer block.
    It consists of a single dense layer followed by a non-linear activation function.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    The output layer of the feed-forward network in a BERT transformer block.
    It applies a dense layer, dropout, and a residual connection with layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

from modules.loss import Load_Balancing_loss
from modules.adapters import NoParamMultiHeadAttention
from modules.loss import Load_Balancing_loss
from modules.encoders import TemporalAwareRouter, RouterPFSelfAttention, TemporalStatisticalRouter


class XBertLayer(nn.Module):
    """
    A modified BERT layer that incorporates the Mixture of Multimodal Adapters (MMA).

    This layer extends a standard BERT transformer layer with mechanisms for multimodal
    fusion. It performs token-level fusion via cross-attention and channel-level fusion
    using a mixture of modality-specific experts (adapters) controlled by a router.
    """
    def __init__(self, config, add_adapter=True):
        """
        Initializes the XBertLayer.

        Args:
            config (BertConfig): The configuration object for the model.
            add_adapter (bool, optional): If True, multimodal adapters are added to this layer. Defaults to True.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.num_experts = 12
        self.add_adapter = add_adapter
        #  -----------------adapters-------------------------
        if add_adapter==True:

            # ---------------------cross attention----------------------#
            self.adapter_conv_audio = nn.Conv1d(in_channels=config.audio_dim, out_channels=config.hidden_size, kernel_size=config.kernel_size)
            self.adapter_conv_vision = nn.Conv1d(in_channels=config.vision_dim, out_channels=config.hidden_size, kernel_size=config.kernel_size)
            self.text_audio_atten = NoParamMultiHeadAttention(embed_size=config.hidden_size)
            self.text_vision_atten = NoParamMultiHeadAttention(embed_size=config.hidden_size)
        
            # ---------------------cross attention----------------------#
            self.audio_dim = config.audio_dim
            self.vision_dim = config.vision_dim
            self.rank = config.rank
        # self.alpha = config.alpha
            self.TopK = config.TopK
            """self.adapter_vision_1 = Adapter_Layer(bottleneck=self.rank)
            self.adapter_vision_2 = Adapter_Layer(bottleneck=self.rank)
        
            self.adapter_audio_1 = Adapter_Layer(bottleneck=self.rank)
            self.adapter_audio_2 = Adapter_Layer(bottleneck=self.rank) 

            self.adapter_1 = Adapter_Layer(bottleneck=self.rank)
            self.adapter_2 = Adapter_Layer(bottleneck=self.rank)"""

            self.txt_local_exp1 = LocalTemporalExpert(bottleneck=self.rank)
            self.txt_local_exp2 = LocalTemporalExpert(bottleneck=self.rank)
            self.txt_global_exp1 = GlobalTemporalExpert(bottleneck=self.rank)
            self.txt_global_exp2 = GlobalTemporalExpert(bottleneck=self.rank)
            
            self.au_local_exp1 = LocalTemporalExpert(bottleneck=self.rank)
            self.au_local_exp2 = LocalTemporalExpert(bottleneck=self.rank)
            self.au_global_exp1 = GlobalTemporalExpert(bottleneck=self.rank)
            self.au_global_exp2 = GlobalTemporalExpert(bottleneck=self.rank)

            self.vis_local_exp1 = LocalTemporalExpert(bottleneck=self.rank)
            self.vis_local_exp2 = LocalTemporalExpert(bottleneck=self.rank)
            self.vis_global_exp1 = GlobalTemporalExpert(bottleneck=self.rank)
            self.vis_global_exp2 = GlobalTemporalExpert(bottleneck=self.rank)
        
            self.adapter_atten_gate = RouterPFSelfAttention()
            self.temporal_adapter = TemporalAwareRouter(embed_dim=768, num_experts=self.num_experts, kernel_size=3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        vision: Optional[bool] = None,
        audio: Optional[bool] = None,
        vision_length = None,
        audio_length = None
    ) -> Tuple[torch.Tensor]:
        """
        Forward pass for the XBertLayer.

        Args:
            hidden_states (torch.Tensor): Input text hidden states.
            attention_mask (torch.FloatTensor, optional): Mask for self-attention.
            head_mask (torch.FloatTensor, optional): Mask for attention heads.
            encoder_hidden_states (torch.FloatTensor, optional): Not used in this implementation.
            encoder_attention_mask (torch.FloatTensor, optional): Not used in this implementation.
            past_key_value (Tuple, optional): Cached key-value states for attention.
            output_attentions (bool, optional): Whether to return attention weights.
            vision (torch.Tensor, optional): Vision feature tensor.
            audio (torch.Tensor, optional): Audio feature tensor.
            vision_length: Not used.
            audio_length: Not used.

        Returns:
            Tuple[torch.Tensor]: A tuple containing:
                - The output hidden states of the layer.
                - The load balancing loss for the adapter routing.
                - The expert distribution frequency.
                - Optional attention outputs.
        """
        #print(f"Hidden states inside XBertLayer: {hidden_states.shape}")
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # bs x seq_len x d_model
        attention_output = self_attention_outputs[0]
        #-----------------------------adapter_change----------token fusion--------------------#
        N = self.num_experts / 3
        if self.add_adapter == True:
            key_padding_mask = attention_mask.squeeze()
            key_padding_mask = torch.where(key_padding_mask == 0, torch.tensor(1), key_padding_mask)
            key_padding_mask = torch.where(key_padding_mask != 1, torch.tensor(0), key_padding_mask)
            # Audio/Visual Features After Conv1D proj (bs x L_a x d_model and (bs x L_v x d_model))
            audio_hat = self.adapter_conv_audio(audio.permute(0,2,1)).permute(0,2,1)
            vision_hat = self.adapter_conv_vision(vision.permute(0,2,1)).permute(0,2,1)
            # Audio/Visual features cross attention with text features (bs x L_t x d_model)
            audio_t = self.text_audio_atten(queries=attention_output,keys=audio_hat,values=audio_hat,mask=key_padding_mask[:,:audio_hat.shape[1]])
            vision_t = self.text_vision_atten(queries=attention_output,keys=vision_hat,values=vision_hat,mask=key_padding_mask[:,:vision_hat.shape[1]])
    
             #-----------------------------adapter_change------------channel fusion------------------#
            topK=self.TopK
            N = self.num_experts // 3

            # every token position (across all sequences) is a separate sample of size [hidden_dim]
            """batch_size, sequence_length, hidden_dim = attention_output.shape
            T = batch_size*sequence_length
            # bs * seq_len x d_model
            attention_output = attention_output.view(-1,hidden_dim)
            audio_t = audio_t.contiguous().view(-1,hidden_dim)
            vision_t = vision_t.contiguous().view(-1,hidden_dim)
            #print("Meta to transpose", attention_output.shape, audio_t.shape, vision_t.shape)  # (bs*len, dim)

            
            # per-token activation score for all experts
            gate_logits = self.adapter_atten_gate(
                torch.stack((attention_output, 
                             audio_t + attention_output, 
                             vision_t + attention_output), 
                            dim=1)
                        ).view(-1, N*3)
            gate_logits = self.temporal_adapter(attention_output, audio_t + attention_output, vision_t + attention_output)
            # per-token probabilities for each expert
            gate_p = F.softmax(gate_logits, dim=1, dtype=torch.float).to(attention_output.dtype)
            #print("Gate P", gate_p.shape)  # (bs*len, num_experts)
            weights, selected_experts = torch.topk(gate_logits, topK)  # T, K
            #print("Selected experts", selected_experts.shape, weights.shape)  # (bs*len, K)
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(attention_output.dtype)
            #print("Weights after norm", weights.shape)
            results = torch.zeros_like(attention_output)

            experts = nn.ModuleList([
                                 self.txt_local_exp, self.txt_global_exp,
                                 self.au_local_exp, self.au_global_exp,
                                 self.vis_local_exp, self.vis_global_exp,
                                 ])
            f = torch.zeros(N*3)
            P = torch.zeros(N*3)
            for i, expert in enumerate(experts):
                f[i] = torch.sum(selected_experts == i).item() /T
                P[i] = torch.sum(gate_p[:,i]) /T
                batch_idx, nth_expert = torch.where(selected_experts == i)
                if i<N:
                    expert_output = expert(attention_output[batch_idx])
                    #print("Text Expert", expert_output.shape)
                elif i>(N-1) and i<2*N:
                    expert_output = expert(audio_t[batch_idx]+attention_output[batch_idx])
                    #print("Audio Expert", expert_output.shape)
                else:
                    expert_output = expert(vision_t[batch_idx]+attention_output[batch_idx])
                    #print("Vision Expert", expert_output.shape)
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_output
            
            attention_output = attention_output.reshape(batch_size, sequence_length, hidden_dim)
            results = results.reshape(batch_size, sequence_length, hidden_dim)
            results = (32/self.rank) * results"""

            batch_size, sequence_length, hidden_dim = attention_output.shape
            T = batch_size * sequence_length

            # routing scores
            gate_logits = self.temporal_adapter(
                attention_output, 
                audio_t + attention_output, 
                vision_t + attention_output
            )

            # probabilities and select top-K
            gate_p = F.softmax(gate_logits, dim=1, dtype=torch.float).to(attention_output.dtype) # T x num_experts
            weights, selected_experts = torch.topk(gate_logits, topK)  # T x K
            weights = F.softmax(weights, dim=1, dtype=torch.float).to(attention_output.dtype)

            # group per modality(t,v,a) and per type
            expert_outputs = [
                self.txt_local_exp1(attention_output), 
                self.txt_local_exp2(attention_output),                    
                self.txt_global_exp1(attention_output),  
                self.txt_global_exp2(attention_output),                   
                self.au_local_exp1(audio_t + attention_output), 
                self.au_local_exp2(audio_t + attention_output),           
                self.au_global_exp1(audio_t + attention_output),  
                self.au_global_exp2(audio_t + attention_output),          
                self.vis_local_exp1(vision_t + attention_output),
                self.vis_local_exp2(vision_t + attention_output),         
                self.vis_global_exp1(vision_t + attention_output),
                self.vis_global_exp2(vision_t + attention_output),        
            ]

            # num_experts x B x L_t x d_model
            expert_outputs = torch.stack(expert_outputs, dim=0)
            # num_experts x T x d_model)
            expert_outputs = expert_outputs.view(self.num_experts, T, hidden_dim)
            # T x num_experts x d_model)
            expert_outputs = expert_outputs.permute(1, 0, 2)

            # T x K x d_model
            selected_experts_expanded = selected_experts.unsqueeze(-1).expand(-1, -1, hidden_dim)

            # T x K x d_model
            selected_outputs = torch.gather(expert_outputs, 1, selected_experts_expanded)

            # T x K x 1
            weights_expanded = weights.unsqueeze(-1)  # (T, K, 1)
            # T x d_model
            weighted_sum = (weights_expanded * selected_outputs).sum(dim=1)

            # B x L_t x d_model
            results = weighted_sum.view(batch_size, sequence_length, hidden_dim)
            results = (32 / self.rank) * results

            # f: fraction of tokens dispatched to each expert
            presence = torch.zeros(T, self.num_experts, dtype=torch.bool)
            presence.scatter_(1, selected_experts, True)   # T x K)
            f = presence.float().mean(dim=0)          

            # P: average routing probability for each expert
            gate_p = F.softmax(gate_logits, dim=1)
            P = gate_p.mean(dim=0)                         # (num_experts,)

            # --- Load balancing ---
            lbloss = Load_Balancing_loss()
            num_modalities = 3       # text, audio, vision
            num_types = 2            # global, local
            N = self.num_experts // (num_modalities * num_types)

            lb_loss = 0.0
            for m in range(num_modalities):
                for tau in range(num_types):
                    start = m * (num_types * N) + tau * N
                    end = start + N
                    lb_loss += lbloss(f[start:end], P[start:end])

            lb_loss = lb_loss * N

        #------------------------------adapter_change------------------------------#

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        if self.add_adapter == True:
            layer_output = layer_output + results
        else:
            lb_loss = 0.
        outputs = (layer_output,) + outputs

        return outputs, lb_loss, f

    def feed_forward_chunk(self, attention_output):
        """
        Processes the attention output through the feed-forward network in chunks.

        Args:
            attention_output (torch.Tensor): The output from the attention mechanism.

        Returns:
            torch.Tensor: The output of the feed-forward network.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    """
    The BERT encoder, composed of a stack of `XBertLayer` layers.

    This module iteratively processes the input sequence through multiple `XBertLayer`
    layers, allowing for deep fusion of multimodal information when adapters are enabled.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.start_fusion_layer = config.start_fusion_layer
        self.layer = nn.ModuleList([XBertLayer(config, add_adapter=(i>=self.start_fusion_layer)) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        vision: Optional[bool] = None,
        audio: Optional[bool] = None,
        vision_length = None,
        audio_length = None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Forward pass for the BertEncoder.

        Args:
            hidden_states (torch.Tensor): Input sequence embeddings.
            attention_mask (torch.FloatTensor, optional): Attention mask for the input sequence.
            head_mask (torch.FloatTensor, optional): Mask for attention heads.
            encoder_hidden_states (torch.FloatTensor, optional): Not used.
            encoder_attention_mask (torch.FloatTensor, optional): Not used.
            past_key_values (Tuple, optional): Cached attention states.
            use_cache (bool, optional): Whether to use and return cached states.
            output_attentions (bool, optional): Whether to return all self-attention weights.
            output_hidden_states (bool, optional): Whether to return all hidden states.
            return_dict (bool, optional): Whether to return a `ModelOutput` object.
            vision (torch.Tensor, optional): Vision feature tensor, passed to each `XBertLayer`.
            audio (torch.Tensor, optional): Audio feature tensor, passed to each `XBertLayer`.
            vision_length: Not used.
            audio_length: Not used.

        Returns:
            Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]: The encoder's output,
            including the final hidden state, total load balancing loss, and expert frequencies.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        batch_size = hidden_states.shape[0]
        LBLoss = 0.0
        f = torch.zeros((12, self.layer[0].num_experts))
        # layer_experts_outputs = torch.zeros((12, batch_size*3*50, 769))
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs, LBLoss_layer, f_layer = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    vision=vision,
                    audio=audio,
                    vision_length = vision_length,
                    audio_length = audio_length
                    # layer = i
                )
            LBLoss = LBLoss + LBLoss_layer
            f[i] = f_layer

            hidden_states = layer_outputs[0]
        LBLoss = LBLoss / (self.config.num_hidden_layers - self.start_fusion_layer)
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
                LBLoss,
                f
                # layer_experts_outputs
                # Load
            ]
            if v is not None
        )
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class BertPooler(nn.Module):
    """
    Pools the model output by taking the hidden state of the first token ([CLS]).

    This is a standard component in BERT for sentence-level classification tasks.
    It applies a linear layer and a Tanh activation function to the [CLS] token's representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """
    A transformation layer used in the BERT prediction head (e.g., for masked language modeling).

    It consists of a dense layer, an activation function, and layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class XBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision: Optional[bool] = None,
        audio: Optional[bool] = None,
        audio_length=None, vision_length=None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        # validation of args and setup
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # prevent attention to padding tokens
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # word_embeddings + position_embeddings + token_type_embeddings → dropout → output
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        #print(f"Embedding output shape: {embedding_output.shape}")
        # batch_size, seq_length, hidden_size].
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision=vision,
            audio=audio,
            vision_length = vision_length,
            audio_length=audio_length
        )
        sequence_output = encoder_outputs[0]
        # expert_outputs  = encoder_outputs[-1]
        Loss = encoder_outputs[-2]
        f = encoder_outputs[-1]
        # print(LBLoss)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # if not return_dict:
        # return (sequence_output, pooled_output) + encoder_outputs[1:]
        return sequence_output, Loss, f
        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )

