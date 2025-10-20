import torch
import torch.nn.functional as F
import math
from torch import nn

def init_bert_weights(module):
    """Initialize the weights of the module according to BERT's initialization strategy."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Adapter_Layer(nn.Module):
    """
    Adapter Layer for the MMA architecture.

    This layer introduces a bottleneck structure to facilitate efficient parameter tuning.
    It consists of a down-projection, a non-linear activation function, and an up-projection.
    The layer supports learnable scaling and optional layer normalization.

    Args:
        config: Configuration object containing model parameters.
        d_model (int): Dimension of the input embeddings (default: 768).
        bottleneck (int): Dimension of the bottleneck layer (default: 64).
        dropout (float): Dropout probability (default: 0.2).
        init_option (str): Initialization strategy for the weights (default: "bert").
        adapter_scalar (str): Type of scaling for the adapter (default: "learnable_scalar").
        adapter_layernorm_option (str): Layer normalization option (default: "in").
    """
    def __init__(self,
                 config=None,
                 d_model=768,
                 bottleneck=64,
                 dropout=0.2,
                 init_option="bert",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.ca_heads = 4
        self.pivot_dim = self.down_size

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        """
        Forward pass through the adapter layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_embd).
            add_residual (bool): Whether to add the residual connection (default: False).
            residual (torch.Tensor): Residual tensor to add (default: None).

        Returns:
            torch.Tensor: Output tensor after applying the adapter layer.
        """
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class NoParamMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism without learnable parameters.

    This class implements a multi-head attention mechanism that does not require additional parameters,
    making it suitable for lightweight models. It performs scaled dot-product attention across multiple heads.

    Args:
        num_heads (int): Number of attention heads (default: 12).
        embed_size (int): Dimension of the input embeddings (default: 768).
    """
    def __init__(self, num_heads=12, embed_size=768):
        super(NoParamMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

    def forward(self, queries, values, keys, mask=None):
        """
        Forward pass through the multi-head attention mechanism.

        Args:
            queries (torch.Tensor): Query tensor of shape (batch_size, query_len, embed_size).
            values (torch.Tensor): Value tensor of shape (batch_size, value_len, embed_size).
            keys (torch.Tensor): Key tensor of shape (batch_size, key_len, embed_size).
            mask (torch.Tensor, optional): Mask tensor for attention scores (default: None).

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention.
        """
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention_scores = attention_scores / (self.embed_size ** (1/2))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention_scores, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)

        return out
    

class LocalTemporalExpert(nn.Module):
    def __init__(self, config=None, d_model=768,
                 bottleneck=64, dropout=0.2, 
                 init_option="bert", kernel_size=3,
                 dilation=1, 
                 adapter_scalar="learnable_scalar", 
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None and config is not None else d_model
        self.down_size = config.attn_bn if bottleneck is None and config is not None else bottleneck
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dilation = dilation

        # layer norm (optional)
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # scaling
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        
        # down-projection
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        
        # up-projection
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # temporal convolution depthwise separable
        padding = (kernel_size // 2) * dilation
        
        # depthwise convolution: one filter per channel
        self.depthwise_conv = nn.Conv1d(
            in_channels=self.down_size,
            out_channels=self.down_size,
            kernel_size=kernel_size,
            groups=self.down_size,
            padding=padding,
            dilation=dilation
        )
        
        # pointwise convolution: mixes across channels
        self.pointwise_conv = nn.Conv1d(
            in_channels=self.down_size,
            out_channels=self.down_size,
            kernel_size=1
        )

        # temporal gating
        self.gate_proj = nn.Linear(self.down_size, self.down_size)

        # weight initialization
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
                nn.init.kaiming_uniform_(self.depthwise_conv.weight, a=math.sqrt(5))
                nn.init.zeros_(self.depthwise_conv.bias)
                nn.init.kaiming_uniform_(self.pointwise_conv.weight, a=math.sqrt(5))
                nn.init.zeros_(self.pointwise_conv.bias)
                nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.gate_proj.bias)
        
    def forward(self, x, add_residual=False, residual=None):
        """
        Args:
            x: (batch_size, seq_len, d_model) - X_m^(l)
            add_residual: whether to add residual connection
            residual: optional residual tensor
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) - E_m^(l,n)
        """
        residual = x if residual is None else residual

        # optional pre-layernorm
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        # down-projection 
        H = self.down_proj(x)  # (batch_size, seq_len, down_size)
        H = self.non_linear_func(H)

        # (batch, seq_len, channels) -> (batch, channels, seq_len)
        H_t = H.transpose(1, 2)
        
        # depthwise separable convolution
        H_temp_t = self.depthwise_conv(H_t)
        H_temp_t = F.relu(H_temp_t)
        H_temp_t = self.pointwise_conv(H_temp_t)
        H_temp_t = F.relu(H_temp_t)
        H_temp_t = F.dropout(H_temp_t, p=self.dropout, training=self.training)
        
        # transpose back: (batch, channels, seq_len) -> (batch, seq_len, channels)
        H_temp = H_temp_t.transpose(1, 2)

        # temporal gating
        G = torch.sigmoid(self.gate_proj(H))  # (batch_size, seq_len, down_size)
        
        # gated combination
        H_tilde = G * H_temp + (1 - G) * H

        # up-projection
        up = self.up_proj(H_tilde)
        up = up * self.scale

        # optional post-layernorm
        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    
class GlobalTemporalExpert(nn.Module):
    def __init__(self, config=None, d_model=768,
                 bottleneck=64, dropout=0.2, 
                 num_heads=4, max_relative_position=32,
                 init_option="bert",
                 adapter_scalar="learnable_scalar", 
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None and config is not None else d_model
        self.down_size = config.attn_bn if bottleneck is None and config is not None else bottleneck
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
    
        assert self.down_size % self.num_heads == 0
        self.head_dim = self.down_size // self.num_heads
        # layer norm (optional)
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # scaling
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        
        # down-projection
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        
        # up-projection
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # multi-head attn projections 
        self.W_q = nn.Linear(self.down_size, self.down_size, bias=False)
        self.W_k = nn.Linear(self.down_size, self.down_size, bias=False)
        self.W_v = nn.Linear(self.down_size, self.down_size, bias=False)
        self.W_o = nn.Linear(self.down_size, self.down_size, bias=False)

        # relative position bias
        self.relative_pos_bias = nn.Parameter(torch.zeros((num_heads, 2 * max_relative_position + 1)))

        self.attn_dropout = nn.Dropout(p=dropout)
        self.output_dropout = nn.Dropout(p=dropout)

        # Weight initialization
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj.bias)
                
                nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_o.weight, a=math.sqrt(5))
                
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.up_proj.bias)
                
                nn.init.zeros_(self.relative_pos_bias)

    def get_relative_position_bias(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # clip(i-j, -L_max, L_max)
        relative_positions = torch.clamp(relative_positions, -self.max_relative_position, self.max_relative_position)
        # shift by L_max
        relative_positions_indices = relative_positions + self.max_relative_position
        bias = self.relative_pos_bias[:, relative_positions_indices]
        return bias
    
    def forward(self, x, add_residual=False, residual=None, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x if residual is None else residual

        # optional pre-layernorm
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        # down proj
        H = self.down_proj(x)
        H = self.non_linear_func(H)

        # attn projections
        Q = self.W_q(H)
        K = self.W_k(H)
        V = self.W_v(H)
        
        # reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        rel_pos_bias = self.get_relative_position_bias(seq_len, x.device)
        attn_scores = attn_scores + rel_pos_bias.unsqueeze(0)

        # apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul((attn_weights), V)

        # reshape back: (batch_size, seq_len, down_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.down_size)

        # output projection
        H_attn = self.W_o(attn_output)
        H_attn = self.output_dropout(H_attn)
        
        # residual connection
        H_tilde = H + H_attn
        
        # up-projection
        up = self.up_proj(H_tilde)
        up = up * self.scale

        # optional post-layernorm
        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
        
class SynchronyExpert(nn.Module):
    def __init__(self, config=None, d_model=768, d_au=5,
                 d_vis=20, bottleneck=64, dropout=0.2,
                 temporal_window=7,
                 init_option="bert",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.audio_dim = d_au
        self.vision_dim = d_vis
        self.down_size = bottleneck
        self.dropout = dropout
        self.temporal_window = temporal_window

        # layer norm (optional)
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # scaling
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # unimodal down-projections
        self.down_proj_audio = nn.Linear(self.audio_dim, self.down_size)
        self.down_proj_video = nn.Linear(self.vision_dim, self.down_size)
        self.non_linear_func = nn.ReLU()

        # temporal cross attn
        self.W_q_audio = nn.Linear(self.down_size, self.down_size, bias=False)
        self.W_k_video = nn.Linear(self.down_size, self.down_size, bias=False)
        self.W_v_video = nn.Linear(self.down_size, self.down_size, bias=False)

        self.proximity_bias = nn.Parameter(torch.zeros(2 * temporal_window + 1))

        # sync fusion
        self.W_1_fusion = nn.Linear(2 * self.down_size, self.down_size)
        self.W_2_fusion = nn.Linear(self.down_size, self.down_size)
        self.fusion_layer_norm = nn.LayerNorm(2 * self.down_size)

        # text alignment
        self.W_q_align = nn.Linear(self.n_embd, self.down_size, bias=False)
        self.W_k_align = nn.Linear(self.down_size, self.down_size, bias=False)
        self.W_v_align = nn.Linear(self.down_size, self.down_size, bias=False)

        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.attn_dropout = nn.Dropout(p=dropout)

        # weight initialization
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj_video.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj_video.bias)
                nn.init.kaiming_uniform_(self.down_proj_audio.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj_audio.bias)

                nn.init.kaiming_uniform_(self.W_q_audio.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_k_video.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_v_video.weight, a=math.sqrt(5))

                nn.init.zeros_(self.proximity_bias)

                nn.init.kaiming_uniform_(self.W_1_fusion.weight, a=math.sqrt(5))
                nn.init.zeros_(self.W_1_fusion.bias)
                nn.init.kaiming_uniform_(self.W_2_fusion.weight, a=math.sqrt(5))
                nn.init.zeros_(self.W_2_fusion.bias)

                nn.init.kaiming_uniform_(self.W_q_align.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_k_align.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_v_align.weight, a=math.sqrt(5))

                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.up_proj.bias)


    def get_proximity_bias(self, L_a, L_v, device):
        audio_positions = torch.arange(L_a, device=device).float()
        video_positions = torch.arange(L_v, device=device).float()

        # scale audio positions to video frame rate
        audio_in_video_scale = audio_positions.unsqueeze(1) * (L_v / L_a)
        video_positions_expanded = video_positions.unsqueeze(0)

        # temporal distance |i - j|
        temporal_distance = torch.abs(audio_in_video_scale - video_positions_expanded)
        temporal_distance_int = torch.floor(temporal_distance).long()

        # bias matrix init with -inf (outside window)
        bias = torch.full((L_a, L_v), float('-inf'), device=device)

        # for positions within window, use learnable bias
        within_window = temporal_distance_int <= self.temporal_window

        # clamp  [0, 2w]
        clamped_indices = torch.clamp(temporal_distance_int, 0, 2 * self.temporal_window)

        # apply learnable bias
        bias[within_window] = self.proximity_bias[clamped_indices[within_window]]

        return bias


    def forward(self, x_text, video_features, audio_features,
                add_residual=False, residual=None):

        batch_size, L_t, _ = x_text.shape
        _, L_v, _ = video_features.shape
        _, L_a, _ = audio_features.shape

        residual = x_text if residual is None else residual

        # optional pre-layernorm on text input
        if self.adapter_layernorm_option == "in":
            x_text = self.adapter_layer_norm_before(x_text)

        # unimodal down-projections
        print(video_features.shape)
        H_v = self.down_proj_video(video_features)
        H_v = self.non_linear_func(H_v)

        H_a = self.down_proj_audio(audio_features)
        H_a = self.non_linear_func(H_a)

        # cross-modal temporal attn
        Q_a = self.W_q_audio(H_a)
        K_v = self.W_k_video(H_v)
        V_v = self.W_v_video(H_v)

        attn_scores = torch.matmul(Q_a, K_v.transpose(-2, -1)) / math.sqrt(self.down_size)
        proximity_bias = self.get_proximity_bias(L_a, L_v, x_text.device)
        attn_scores = attn_scores + proximity_bias.unsqueeze(0)

        A = F.softmax(attn_scores, dim=-1)
        A = self.attn_dropout(A)
        H_v_tilde = torch.matmul(A, V_v)

        # sync fusion
        H_av = torch.cat([H_a, H_v_tilde], dim=-1)
        H_av_norm = self.fusion_layer_norm(H_av)
        H_fused = self.W_1_fusion(H_av_norm)
        H_fused = self.non_linear_func(H_fused)
        H_sync = self.W_2_fusion(H_fused)

        # text alignment
        Q_text = self.W_q_align(x_text)
        K_sync = self.W_k_align(H_sync)
        V_sync = self.W_v_align(H_sync)

        align_scores = torch.matmul(Q_text, K_sync.transpose(-2, -1)) / math.sqrt(self.down_size)
        A_align = F.softmax(align_scores, dim=-1)
        A_align = self.attn_dropout(A_align)
        H_align = torch.matmul(A_align, V_sync)

        # up-projection
        up = self.up_proj(H_align)
        up = up * self.scale

        # optional post-layernorm
        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class LocalTemporalExpert_2(nn.Module):
    def __init__(self, config=None, d_model=768,
                 bottleneck=64, dropout=0.2, 
                 init_option="bert", kernel_sizes=[3, 5],
                 dilation=1, 
                 adapter_scalar="learnable_scalar", 
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None and config is not None else d_model
        self.down_size = config.attn_bn if bottleneck is None and config is not None else bottleneck
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.dilation = dilation

        # layer norm (optional)
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # scaling
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        
        # down-projection
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        
        # up-projection
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # temporal convolution depthwise separable
        pad3 = (3 // 2) * dilation
        pad5 = (5 // 2) * dilation
        
        # depthwise convolution: one filter per channel
        self.depthwise_conv1 = nn.Conv1d(
            in_channels=self.down_size,
            out_channels=self.down_size,
            kernel_size=self.kernel_sizes[0],
            groups=self.down_size,
            padding=pad3,
            dilation=dilation
        )

        self.depthwise_conv2 = nn.Conv1d(in_channels=self.down_size,
                                         out_channels=self.down_size,
                                         kernel_size=self.kernel_sizes[1],
                                         padding=pad5,
                                         dilation=dilation,
                                         groups=self.down_size)
        
        # pointwise convolution: mixes across channels
        self.pointwise_conv = nn.Conv1d(
            in_channels=self.down_size,
            out_channels=self.down_size,
            kernel_size=1
        )

        # temporal gating
        self.gate_proj = nn.Linear(self.down_size, self.down_size)

        # weight initialization
        if init_option == "bert":
            self.apply(init_bert_weights)
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
                nn.init.kaiming_uniform_(self.depthwise_conv1.weight, a=math.sqrt(5))
                nn.init.zeros_(self.depthwise_conv1.bias)
                nn.init.kaiming_uniform_(self.depthwise_conv2.weight, a=math.sqrt(5))
                nn.init.zeros_(self.depthwise_conv2.bias)
                nn.init.kaiming_uniform_(self.pointwise_conv.weight, a=math.sqrt(5))
                nn.init.zeros_(self.pointwise_conv.bias)
                nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.gate_proj.bias)
        
    def forward(self, x, add_residual=False, residual=None):
        """
        Args:
            x: (batch_size, seq_len, d_model) - X_m^(l)
            add_residual: whether to add residual connection
            residual: optional residual tensor
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) - E_m^(l,n)
        """
        residual = x if residual is None else residual

        # optional pre-layernorm
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        # down-projection 
        H = self.down_proj(x)  # (batch_size, seq_len, down_size)
        H = self.non_linear_func(H)

        # (batch, seq_len, channels) -> (batch, channels, seq_len)
        H_t = H.transpose(1, 2)
        H_res = H_t
        # depthwise separable convolution
        H1 = self.depthwise_conv1(H_t)
        H1 = F.relu(H1)
        H1 = F.dropout(H1, p=self.dropout, training=self.training)
        H2 = self.depthwise_conv2(H1)
        H2 = F.relu(H2)
        H2 = self.pointwise_conv(H2)
        H2 = F.relu(H2)
        H2 = F.dropout(H2, p=self.dropout, training=self.training)
        
        H_temp_t = H_res + H2
        # transpose back: (batch, channels, seq_len) -> (batch, seq_len, channels)
        H_temp = H_temp_t.transpose(1, 2)

        # temporal gating
        G = torch.sigmoid(self.gate_proj(H))  # (batch_size, seq_len, down_size)
        
        # gated combination
        H_tilde = G * H_temp + (1 - G) * H

        # up-projection
        up = self.up_proj(H_tilde)
        up = up * self.scale

        # optional post-layernorm
        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    