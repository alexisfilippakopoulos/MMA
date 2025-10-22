# filepath: /MMA/MMA/modules/encoders.py

"""
This module defines the encoder components used in the MMA (Mixture of Multimodal Adapters) architecture for sentiment analysis. 
The MMA architecture integrates multiple modalities, including text, vision, and audio, to enhance sentiment classification tasks.

Classes:
- SubNet: A subnetwork designed for processing audio and video inputs in the pre-fusion stage. It consists of fully connected layers with dropout for regularization.
- RouterSelfAttention: Implements self-attention mechanisms that allow the model to focus on different parts of the input sequence, enhancing the representation of multimodal data.
- RouterPFSelfAttention: A variant of self-attention that processes inputs without additional linear projections, optimizing the attention mechanism for efficiency.
- RouterPFMultiHeadAttention: A multi-head attention mechanism that allows the model to attend to different parts of the input across multiple heads, facilitating better representation learning.

The encoder components are crucial for extracting meaningful features from the multimodal inputs, which are then fused and processed by the MMA architecture for sentiment analysis.
"""

import torch
import torch.nn.functional as F
import time
import math
from torch import nn

class SubNet(nn.Module):
    '''
    The subnetwork that is used in the MMA architecture for processing audio and video inputs in the pre-fusion stage.

    Args:
        in_size (int): Input dimension.
        hidden_size (int): Hidden layer dimension.
        n_class (int): Number of output classes.
        dropout (float): Dropout probability for regularization.

    Output:
        A tensor of shape (batch_size, hidden_size) representing the processed features.
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout):
        super(SubNet, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Forward pass for the SubNet.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, n_class) representing the class scores.
        '''
        dropped = self.drop(x)
        y_1 = torch.relu(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.relu(fusion)
        y_3 = self.linear_3(y_2)
        return y_3

class RouterSelfAttention(nn.Module):
    '''
    Implements self-attention mechanism for the MMA architecture.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        attn_dim (int): Dimension of the attention mechanism.

    Output:
        A tensor representing the attended features.
    '''

    def __init__(self, embed_dim=768, attn_dim=2):
        super(RouterSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim if attn_dim else embed_dim
        
        self.query = nn.Linear(embed_dim, self.attn_dim)
        self.key = nn.Linear(embed_dim, self.attn_dim)
        self.value = nn.Linear(embed_dim, self.attn_dim)
        self.out = nn.Linear(self.attn_dim, self.attn_dim)
        
    def forward(self, x):
        '''
        Forward pass for the RouterSelfAttention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, attn_dim) representing the attended features.
        '''
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Embedding dimension must match"
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        output = self.out(attn_output)
        
        return output
    
class RouterPFSelfAttention(nn.Module):
    '''
    Implements a variant of self-attention for the MMA architecture without additional linear projections.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        attn_dim (int): Dimension of the attention mechanism.

    Output:
        A tensor representing the attended features.
    '''

    def __init__(self, embed_dim=768, attn_dim=2):
        super(RouterPFSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim if attn_dim else embed_dim
        
        self.out = nn.Linear(self.embed_dim, self.attn_dim)
        
    def forward(self, x):
        '''
        Forward pass for the RouterPFSelfAttention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, attn_dim) representing the attended features.
        '''
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Embedding dimension must match"
        
        Q = x
        K = x
        V = x
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        output = self.out(attn_output)
        
        return output

class RouterPFMultiHeadAttention(nn.Module):
    '''
    Implements multi-head attention mechanism for the MMA architecture.

    Args:
        num_heads (int): Number of attention heads.
        embed_size (int): Dimension of the input embeddings.

    Output:
        A tensor representing the attended features across multiple heads.
    '''

    def __init__(self, num_heads=12, embed_size=768):
        super(RouterPFMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

    def forward(self, queries, mask=None):
        '''
        Forward pass for the RouterPFMultiHeadAttention.

        Args:
            queries (Tensor): Input tensor of shape (batch_size, query_len, embed_size).
            mask (Tensor, optional): Mask tensor for attention scores.

        Returns:
            Tensor: Output tensor of shape (batch_size, query_len, embed_size) representing the attended features.
        '''
        N = queries.shape[0]
        values = queries
        keys = queries
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
        out = self.out(out)

        return out
    

class TemporalAwareRouter(nn.Module):
    """
    Temporal-Aware Router that incorporates both spatial and temporal context.

    Args:
        embed_dim (int): Dimension of input embeddings (d_t in paper)
        num_experts (int): Total number of experts (N_total = 14 with N=2 per modality type)
        kernel_size (int): Kernel size for local temporal convolution
    """
    def __init__(self, embed_dim=768, num_experts=6, kernel_size=3):
        super(TemporalAwareRouter, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        # local temporal context
        self.local_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding="same",
            groups=1
        )

        # global temporal context projection
        self.global_proj = nn.Linear(embed_dim, embed_dim)

        # maps spatio-temporal features to expert scores
        self.route_proj = nn.Linear(5 * embed_dim, num_experts)

    def forward(self, x_text, x_au, x_vis):
        """
        Forward pass for temporal-aware routing.

        Args:
            x_text (Tensor): Text features X_t^l
                            Shape: (batch_size, seq_length, embed_dim)
            x_au (Tensor): Text-Audio features X_a^l
                            Shape: (batch_size, seq_length, embed_dim)
            x_vis (Tensor): Text-Visual features X_v^l
                            Shape: (batch_size, seq_length, embed_dim)

        Returns:
            Tensor: Gating logits g_i^(l) of shape (batch_size*seq_length, num_experts)
        """
        batch_size = x_text.shape[0]
        seq_length = x_text.shape[1]

        # B x d_model x L_t
        x_text_t = x_text.transpose(2, 1)
        local_context = self.local_conv(x_text_t)
        # B x L_t x d_model
        local_context = local_context.transpose(1, 2)

        # B x 1 x d_model
        global_context = torch.mean(x_text, dim=1, keepdim=True)
        # project and broadcast to all positions)
        global_context_proj = self.global_proj(global_context) 
        global_context = global_context_proj.expand(batch_size, seq_length, self.embed_dim)

        # temporal and spatial features (bs x seq_length x 5*embed_dim)
        M = torch.cat([x_text, x_au, x_vis, local_context, global_context], dim=-1)  # 

        # param-free self attn
        Q, K, V = M, M, M

        attn_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        M_tilde = torch.matmul(attn_weights, V) 

        gate_logits = self.route_proj(M_tilde)
        # B * L_t x num_experts
        gate_logits = gate_logits.view(-1, self.num_experts)

        return gate_logits

class TemporalStatisticalRouter(nn.Module):
    """
    Temporal-Aware Router using lightweight statistical temporal indicators.
    
    Theory: Router detects PRESENCE of temporal patterns via change metrics,
    without actually processing temporal dependencies (that's the expert's job).
    
    Args:
        embed_dim (int): Dimension of input embeddings (d_t)
        num_experts (int): Total number of experts
        stat_dim (int): Dimension for statistical features (default: 32)
    """
    def __init__(self, embed_dim=768, num_experts=6, stat_dim=32):
        super(TemporalStatisticalRouter, self).__init__()  # Changed class name here
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.stat_dim = stat_dim
        
        # Lightweight projections for temporal statistics
        # Each modality gets its own change detector
        self.text_stat_proj = nn.Linear(embed_dim * 2, stat_dim)  # 768*2 → 32 = 49K
        self.audio_stat_proj = nn.Linear(embed_dim * 2, stat_dim)
        self.video_stat_proj = nn.Linear(embed_dim * 2, stat_dim)
        
        # Cross-modal alignment detector
        self.alignment_proj = nn.Linear(embed_dim * 2, stat_dim)
        
        # Final routing projection
        # Input: 3*embed_dim (spatial) + 4*stat_dim (temporal stats)
        self.route_proj = nn.Linear(3 * embed_dim + 4 * stat_dim, num_experts)
        
        # Total params: 4*(768*2*32) + (3*768 + 4*32)*14 ≈ 230K

    def compute_temporal_statistics(self, x):
        """
        Compute temporal change statistics without actual temporal modeling.
        
        Args:
            x: (batch_size, seq_length, embed_dim)
            
        Returns:
            stats: Dict with temporal indicators
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # 1. First-order difference (local change indicator)
        # Δ_i = x_i - x_{i-1}
        x_shifted = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)  # Prepend first token
        delta = x - x_shifted  # (bs, seq_len, embed_dim)
        
        # 2. Concatenate change and current value
        # This captures: "What is the token AND how much did it change?"
        change_features = torch.cat([x, delta], dim=-1)  # (bs, seq_len, 2*embed_dim)
        
        return {
            'change_features': change_features,
            'delta': delta
        }

    def forward(self, x_text, x_au, x_vis):
        """
        Args:
            x_text: (batch_size, seq_length, embed_dim)
            x_au: (batch_size, seq_length, embed_dim)
            x_vis: (batch_size, seq_length, embed_dim)
            
        Returns:
            gate_logits: (batch_size*seq_length, num_experts)
        """
        batch_size, seq_length, _ = x_text.shape
        
        # ===== Compute Temporal Statistics (Indicators, not Processing) =====
        
        # Per-modality change detection
        text_stats = self.compute_temporal_statistics(x_text)
        audio_stats = self.compute_temporal_statistics(x_au)
        video_stats = self.compute_temporal_statistics(x_vis)
        
        # Project change features to compact statistics
        text_temporal = self.text_stat_proj(text_stats['change_features'])
        audio_temporal = self.audio_stat_proj(audio_stats['change_features'])
        video_temporal = self.video_stat_proj(video_stats['change_features'])
        
        # Cross-modal alignment indicator
        # Compute difference between audio and video to detect misalignment
        av_diff = x_au - x_vis
        av_concat = torch.cat([x_au, av_diff], dim=-1)
        alignment_indicator = self.alignment_proj(av_concat)
        
        # ===== Concatenate Spatial + Temporal Indicators =====
        
        # Spatial features (what content is present)
        spatial = torch.cat([x_text, x_au, x_vis], dim=-1)  # (bs, seq_len, 3*embed_dim)
        
        # Temporal indicators (what patterns are present)
        temporal = torch.cat([
            text_temporal,      # Text change
            audio_temporal,     # Audio change
            video_temporal,     # Video change
            alignment_indicator # Audio-visual alignment
        ], dim=-1)  # (bs, seq_len, 4*stat_dim)
        
        # Combined features
        features = torch.cat([spatial, temporal], dim=-1)
        
        # ===== Route to Experts =====
        gate_logits = self.route_proj(features)  # (bs, seq_len, num_experts)
        gate_logits = gate_logits.view(-1, self.num_experts)
        
        return gate_logits
    

class MMA_Router(nn.Module):
    """
    Temporal-Aware Router that incorporates both spatial and temporal context.

    Args:
        embed_dim (int): Dimension of input embeddings (d_t in paper)
        num_experts (int): Total number of experts (N_total = 14 with N=2 per modality type)
        kernel_size (int): Kernel size for local temporal convolution
    """
    def __init__(self, embed_dim=768, num_experts=6):
        super(TemporalAwareRouter, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        # maps spatio-temporal features to expert scores
        self.route_proj = nn.Linear(3 * embed_dim, num_experts)

    def forward(self, x_text, x_au, x_vis):
        """
        Forward pass for temporal-aware routing.

        Args:
            x_text (Tensor): Text features X_t^l
                            Shape: (batch_size, seq_length, embed_dim)
            x_au (Tensor): Text-Audio features X_a^l
                            Shape: (batch_size, seq_length, embed_dim)
            x_vis (Tensor): Text-Visual features X_v^l
                            Shape: (batch_size, seq_length, embed_dim)

        Returns:
            Tensor: Gating logits g_i^(l) of shape (batch_size*seq_length, num_experts)
        """
        batch_size = x_text.shape[0]
        seq_length = x_text.shape[1]

        # B x d_model x L_t
        x_text_t = x_text.transpose(2, 1)
        M = torch.cat([x_text, x_au, x_vis], dim=-1)  # 

        # param-free self attn
        Q, K, V = M, M, M

        attn_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(768), dim=-1)
        M_tilde = torch.matmul(attn_weights, V) 
        M_tilde.shape 

        gate_logits = self.route_proj(M_tilde)
        # B * L_t x num_experts
        gate_logits = gate_logits.view(-1, self.num_experts)

        return gate_logits


class UltraLightTemporalRouter(nn.Module):
    """
    Ultra-lightweight temporal router using parameter-free statistics.
    
    Theory: Use simple scalar change metrics as temporal indicators,
    then concatenate with spatial features for routing.
    
    Args:
        embed_dim (int): Dimension of input embeddings (d_t)
        num_experts (int): Total number of experts
    """
    def __init__(self, embed_dim=768, num_experts=6):
        super(UltraLightTemporalRouter, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        # Single projection: [3*embed_dim + 7 scalar features] → num_experts
        # 7 features: 3 change magnitudes + 3 change directions + 1 av_alignment
        self.route_proj = nn.Linear(3 * embed_dim + 7, num_experts)
        
        # Total params: (3*768 + 7) * 6 = 13,854 params for 6 experts
        #               (3*768 + 7) * 14 = 32,326 params for 14 experts

    def compute_change_indicators(self, x):
        """
        Compute parameter-free temporal change indicators.
        
        Args:
            x: (batch_size, seq_length, embed_dim)
            
        Returns:
            change_mag: (batch_size, seq_length, 1) - magnitude of change
            change_dir: (batch_size, seq_length, 1) - direction consistency
        """
        # First-order difference: Δ_i = x_i - x_{i-1}
        x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        delta = x - x_prev  # (bs, seq_len, embed_dim)
        
        # Change magnitude (L2 norm)
        change_mag = torch.norm(delta, p=2, dim=-1, keepdim=True)  # (bs, seq_len, 1)
        
        # Change direction consistency (cosine similarity between consecutive changes)
        # High value → smooth change, Low value → erratic change
        delta_prev = torch.cat([delta[:, :1, :], delta[:, :-1, :]], dim=1)
        # Cosine similarity
        cos_sim = F.cosine_similarity(delta, delta_prev, dim=-1, eps=1e-8)
        change_dir = cos_sim.unsqueeze(-1)  # (bs, seq_len, 1)
        
        return change_mag, change_dir

    def forward(self, x_text, x_au, x_vis):
        """
        Args:
            x_text: (batch_size, seq_length, embed_dim)
            x_au: (batch_size, seq_length, embed_dim)
            x_vis: (batch_size, seq_length, embed_dim)
            
        Returns:
            gate_logits: (batch_size*seq_length, num_experts)
        """
        batch_size, seq_length, _ = x_text.shape
        
        # ===== Compute Parameter-Free Temporal Indicators =====
        
        # Per-modality change indicators
        text_mag, text_dir = self.compute_change_indicators(x_text)
        audio_mag, audio_dir = self.compute_change_indicators(x_au)
        video_mag, video_dir = self.compute_change_indicators(x_vis)
        
        # Cross-modal alignment indicator (audio-visual difference magnitude)
        av_diff = x_au - x_vis
        av_alignment = torch.norm(av_diff, p=2, dim=-1, keepdim=True)  # (bs, seq_len, 1)
        
        # ===== Concatenate Everything =====
        
        features = torch.cat([
            # Spatial features (what content is present)
            x_text,      # (bs, seq_len, 768)
            x_au,        # (bs, seq_len, 768)
            x_vis,       # (bs, seq_len, 768)
            
            # Temporal indicators (scalar features)
            text_mag,    # (bs, seq_len, 1) - How much did text change?
            audio_mag,   # (bs, seq_len, 1) - How much did audio change?
            video_mag,   # (bs, seq_len, 1) - How much did video change?
            text_dir,    # (bs, seq_len, 1) - Is text change smooth or erratic?
            audio_dir,   # (bs, seq_len, 1) - Is audio change smooth or erratic?
            video_dir,   # (bs, seq_len, 1) - Is video change smooth or erratic?
            av_alignment # (bs, seq_len, 1) - Are audio and video aligned?
        ], dim=-1)  # (bs, seq_len, 3*768 + 7)
        
        # ===== Route to Experts =====
        gate_logits = self.route_proj(features)  # (bs, seq_len, num_experts)
        gate_logits = gate_logits.view(-1, self.num_experts)
        
        return gate_logits
    
class GlobalInformedSynchronyRouter(nn.Module):
    """
    Implements the Global-Informed Synchrony Router.

    This router makes token-level routing decisions by combining two streams:
    1. Local Vector ("What"): Captures instantaneous content, volatility (change),
       and audio-visual synchrony at the current token.
    2. Global Vector ("Context"): Captures the average content, volatility,
       and synchrony across the entire sequence.

    The fusion of "What" (local) and "Context" (global) provides a rich
    feature set for routing to spatial, temporal, or synchrony experts.

    Args:
        embed_dim (int): Dimension of input embeddings (d_t).
        num_experts (int): Total number of experts to route to.
        local_spatial_dim (int): Proj. dim for local spatial features.
        local_temporal_dim (int): Proj. dim for local temporal features (deltas).
        local_sync_dim (int): Proj. dim for local synchrony features (AV diff).
        global_dim (int): Proj. dim for the final global context vector.
    """
    def __init__(self,
                 embed_dim=768,
                 num_experts=6,
                 local_spatial_dim=128,
                 local_temporal_dim=64,
                 local_sync_dim=32,
                 global_dim=64):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        # --- Component 1: Local Feature Projections ---
        
        # 1a. Local Spatial Content (Text + Audio + Video)
        self.local_spatial_proj = nn.Linear(3 * embed_dim, local_spatial_dim)
        
        # 1b. Local Temporal Volatility (Deltas for Text, Audio, Video)
        self.local_temporal_proj = nn.Linear(3 * embed_dim, local_temporal_dim)
        
        # 1c. Local Synchrony (Audio-Video Difference)
        self.local_sync_proj = nn.Linear(embed_dim, local_sync_dim)
        
        # --- Component 2: Global Summary Projection ---
        
        # The input to the global projector is the concatenation of the
        # mean-pooled local spatial, temporal, and sync vectors.
        local_summary_dim = local_spatial_dim + local_temporal_dim + local_sync_dim
        self.global_proj = nn.Linear(local_summary_dim, global_dim)
        
        # --- Component 3: Final Routing Projection ---
        
        # The final routing layer takes the fusion of the
        # full local vector and the broadcasted global vector.
        final_feature_dim = local_summary_dim + global_dim
        self.route_proj = nn.Linear(final_feature_dim, num_experts)
        

    def _compute_delta(self, x):
        """
        Computes the first-order difference: delta_i = x_i - x_{i-1}.
        
        Args:
            x: (batch_size, seq_length, embed_dim)
            
        Returns:
            delta: (batch_size, seq_length, embed_dim)
        """
        # Prepend the first token to create x_{i-1}
        x_shifted = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        delta = x - x_shifted
        return delta

    def forward(self, x_text, x_au, x_vis):
        """
        Args:
            x_text: (batch_size, seq_length, embed_dim)
            x_au: (batch_size, seq_length, embed_dim)
            x_vis: (batch_size, seq_length, embed_dim)
            
        Returns:
            gate_logits: (batch_size * seq_length, num_experts)
        """
        batch_size, seq_length, _ = x_text.shape
        
        # ===== Component 1: Local Feature Extractor (The "What") =====
        
        # 1a. Local Spatial Content
        local_spatial_in = torch.cat([x_text, x_au, x_vis], dim=-1)
        local_spatial_vec = self.local_spatial_proj(local_spatial_in) # (B, S, 128)
        
        # 1b. Local Temporal Indicators (Volatility)
        delta_text = self._compute_delta(x_text)
        delta_audio = self._compute_delta(x_au)
        delta_video = self._compute_delta(x_vis)
        
        local_temporal_in = torch.cat([delta_text, delta_audio, delta_video], dim=-1)
        local_temporal_vec = self.local_temporal_proj(local_temporal_in) # (B, S, 64)
        
        # 1c. Local Synchrony Indicator
        local_sync_in = x_au - x_vis
        local_sync_vec = self.local_sync_proj(local_sync_in) # (B, S, 32)
        
        # 1d. Final Local Vector
        # This is the full "What" vector for each token
        local_vector = torch.cat([
            local_spatial_vec, 
            local_temporal_vec, 
            local_sync_vec
        ], dim=-1) # (B, S, 224)
        
        # ===== Component 2: Global Summary Generator (The "Context") =====
        
        # 2a. Pool all three local streams over the time dimension
        global_spatial_summary = torch.mean(local_spatial_vec, dim=1) # (B, 128)
        global_temp_summary = torch.mean(local_temporal_vec, dim=1)   # (B, 64)
        global_sync_summary = torch.mean(local_sync_vec, dim=1)    # (B, 32)
        
        # 2b. Concatenate summaries and project to the final "Context" vector
        global_summary = torch.cat([
            global_spatial_summary, 
            global_temp_summary, 
            global_sync_summary
        ], dim=-1) # (B, 224)
        
        global_vector = self.global_proj(global_summary) # (B, 64)
        
        # ===== Component 3: Fusion and Routing Logic (The "Decision") =====
        
        # 3a. Broadcast the "Context" vector to match the sequence length
        # (B, 64) -> (B, 1, 64) -> (B, S, 64)
        global_vector_expanded = global_vector.unsqueeze(1).expand(-1, seq_length, -1)
        
        # 3b. Fuse the "What" (local) and "Context" (global)
        final_feature = torch.cat([
            local_vector,             # (B, S, 224)
            global_vector_expanded    # (B, S, 64)
        ], dim=-1) # (B, S, 288)
        
        # 3c. Final Routing
        gate_logits = self.route_proj(final_feature) # (B, S, num_experts)
        
        # Reshape for MoE loss/gating
        gate_logits = gate_logits.view(-1, self.num_experts) # (B*S, num_experts)
        
        return gate_logits