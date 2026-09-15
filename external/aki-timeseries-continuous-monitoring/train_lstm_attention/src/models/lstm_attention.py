# %%
"""
LSTM with Attention Model for Time Series Classification

This module implements an LSTM encoder with multi-head attention mechanism
for healthcare time series prediction (e.g., AKI prediction).

The model uses:
- Irregular Time Encoding (ITE) for handling irregularly sampled time series
- Bidirectional LSTM for sequence encoding
- Multi-head self-attention for capturing long-range dependencies
- Classification head for final prediction

Memory Optimizations:
- Gradient checkpointing for LSTM
- Efficient attention pooling (single query)
- Optional unidirectional mode
- Sequence length limiting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Optional, Tuple


class IrregularTimeEmbedding(nn.Module):
    """
    Irregular Time Embedding (ITE) for handling irregularly sampled time series.
    
    Input format: [feature_id, time, value]
    - feature_id: categorical variable indicating the type of measurement
    - time: continuous variable indicating when the measurement was taken
    - value: the actual measurement value
    """
    
    def __init__(self, config: dict):
        super().__init__()
        d_model = config["d_model"]
        
        # Type embedding for feature categories
        self.type_emb = nn.Embedding(len(config["type_dict"]), d_model)
        
        # Value embedding with non-linear projection
        self.value_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model, bias=False)
        )
        
        # Time embedding with non-linear projection
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, 3]
               where the last dimension is [feature_id, time, value]
        
        Returns:
            Embedded tensor of shape [batch, seq_len, d_model]
        """
        # x: [batch, seq, 3] -> [type, time, value]
        type_emb = self.type_emb(x[..., 0].long())
        time_emb = self.time_emb(x[..., 1:2])
        value_emb = self.value_emb(x[..., 2:3])
        
        # Combine embeddings
        return type_emb + time_emb + value_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for capturing dependencies in sequences.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_q, d_model]
            key: [batch, seq_k, d_model]
            value: [batch, seq_v, d_model]
            mask: [batch, seq_q, seq_k] or [batch, 1, seq_k] - True for valid, False for masked
        
        Returns:
            output: [batch, seq_q, d_model]
            attention_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            if mask.dim() == 2:
                # [batch, seq_k] -> [batch, 1, 1, seq_k]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [batch, seq_q, seq_k] -> [batch, 1, seq_q, seq_k]
                mask = mask.unsqueeze(1)
            
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate sequence representations.
    Uses a learnable query to attend over the entire sequence.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] - True for valid, False for masked
        
        Returns:
            Pooled representation: [batch, d_model]
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Apply attention (query attends to the entire sequence)
        pooled, _ = self.attention(query, x, x, mask)
        
        # Remove the sequence dimension and apply layer norm
        pooled = pooled.squeeze(1)
        pooled = self.layer_norm(pooled)
        
        return pooled


class LSTMAttentionModel(nn.Module):
    """
    LSTM with Attention model for irregularly sampled time series classification.
    
    Architecture:
    1. Irregular Time Embedding (ITE) for input encoding
    2. Bidirectional LSTM for sequence modeling
    3. Multi-head self-attention for capturing dependencies
    4. Attention-based pooling for sequence aggregation
    5. Classification head for final prediction
    
    Memory Optimizations:
    - Gradient checkpointing (optional)
    - Reduced FFN expansion factor (2x instead of 4x)
    - Max sequence length limiting
    """
    
    def __init__(self, config: dict, device: str = None):
        super().__init__()
        
        # Validate config
        required = ["d_model", "num_layers", "num_heads", "dropout", "num_classes"]
        if not all(k in config for k in required):
            raise ValueError(f"Missing keys: {set(required) - set(config.keys())}")
        
        d_model = config["d_model"]
        num_layers = config.get("num_lstm_layers", config.get("num_layers", 2))
        num_heads = config["num_heads"]
        dropout = config["dropout"]
        num_classes = config["num_classes"]
        bidirectional = config.get("bidirectional", True)
        
        # Memory optimization settings
        self.use_gradient_checkpointing = config.get("gradient_checkpointing", False)
        self.max_seq_len = config.get("max_seq_len", 2048)  # Limit sequence length
        ffn_expansion = config.get("ffn_expansion", 2)  # Reduced from 4
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by heads {num_heads}")
        
        self.bidirectional = bidirectional
        self.d_model = d_model
        
        # Input embedding
        self.embedding = IrregularTimeEmbedding(config)
        self.input_dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2 if bidirectional else d_model,  # Halve if bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Project LSTM output back to d_model
        lstm_output_dim = d_model  # bidirectional halves hidden, so output is still d_model
        self.lstm_projection = nn.Linear(lstm_output_dim, d_model)
        
        # Self-attention layer
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Feed-forward layer after attention (reduced expansion)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )
        self.ff_layer_norm = nn.LayerNorm(d_model)
        
        # Attention-based pooling
        self.pooling = AttentionPooling(d_model, num_heads, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _truncate_sequence(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Truncate sequence to max_seq_len if needed."""
        if x.size(1) > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            if mask is not None:
                mask = mask[:, :self.max_seq_len]
        return x, mask
    
    def _lstm_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """LSTM forward pass (can be checkpointed)."""
        if mask is not None:
            # Pack padded sequence for efficient LSTM processing
            lengths = mask.sum(dim=1).cpu()
            
            # Sort by length for packing (required by pack_padded_sequence)
            sorted_lengths, sorted_idx = lengths.sort(descending=True)
            sorted_x = x[sorted_idx]
            
            # Clamp lengths to avoid zero-length sequences
            sorted_lengths = sorted_lengths.clamp(min=1)
            
            packed = nn.utils.rnn.pack_padded_sequence(
                sorted_x, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.size(1)
            )
            
            # Unsort to restore original order
            _, unsorted_idx = sorted_idx.sort()
            lstm_output = lstm_output[unsorted_idx]
        else:
            lstm_output, (h_n, c_n) = self.lstm(x)
        
        return lstm_output
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the LSTM with Attention model.
        
        Args:
            x: Input tensor of shape [batch, seq_len, 3]
               where the last dimension is [feature_id, time, value]
            mask: Boolean mask of shape [batch, seq_len]
                  True for valid tokens, False for padding
            return_features: If True, return (logits, features)
        
        Returns:
            logits: Classification logits of shape [batch, num_classes]
            features (optional): Sequence features of shape [batch, d_model]
        """
        # Truncate long sequences to save memory
        x, mask = self._truncate_sequence(x, mask)
        
        batch_size = x.size(0)
        
        # 1. Embed input
        x = self.embedding(x)
        x = self.input_dropout(x)
        
        # 2. LSTM encoding (with optional gradient checkpointing)
        if self.use_gradient_checkpointing and self.training:
            # Note: checkpoint doesn't work well with pack_padded_sequence
            # So we use it only when mask is None
            if mask is None:
                lstm_output = checkpoint(lambda inp: self.lstm(inp)[0], x, use_reentrant=False)
            else:
                lstm_output = self._lstm_forward(x, mask)
        else:
            lstm_output = self._lstm_forward(x, mask)
        
        # Project LSTM output
        lstm_output = self.lstm_projection(lstm_output)
        
        # 3. Self-attention with residual connection
        attn_output, _ = self.self_attention(lstm_output, lstm_output, lstm_output, mask)
        lstm_output = self.attention_layer_norm(lstm_output + self.attention_dropout(attn_output))
        
        # 4. Feed-forward with residual connection
        if self.use_gradient_checkpointing and self.training:
            ff_output = checkpoint(self.feed_forward, lstm_output, use_reentrant=False)
        else:
            ff_output = self.feed_forward(lstm_output)
        lstm_output = self.ff_layer_norm(lstm_output + ff_output)
        
        # 5. Attention pooling
        pooled = self.pooling(lstm_output, mask)
        
        # 6. Classification
        logits = self.classifier(pooled)
        
        if return_features:
            return logits, pooled
        
        return logits


# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    config = {
        "d_model": 128,
        "num_layers": 2,
        "num_lstm_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "num_classes": 2,
        "bidirectional": True,
        "type_dict": {
            "age": 0,
            "sex": 1,
            "bmi": 2,
            "sbp": 3,
            "dbp": 4,
            "pulse": 5,
            "resp": 6,
            "spo2": 7,
            "temp": 8,
            "creatinine": 9,
        },
    }
    
    model = LSTMAttentionModel(config).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test data: [type, time, value]
    x = torch.cat([
        torch.randint(0, 10, (4, 50, 1)),  # type
        torch.randn(4, 50, 1),  # time
        torch.randn(4, 50, 1),  # value
    ], dim=-1).float().to(device)
    
    # Create mask (some padding at the end)
    mask = torch.zeros(4, 50, dtype=torch.bool).to(device)
    for i in range(4):
        seq_len = torch.randint(20, 51, (1,)).item()
        mask[i, :seq_len] = True
    
    with torch.no_grad():
        logits = model(x, mask)
        probs = torch.softmax(logits, dim=-1)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Sample probabilities: {probs[0]}")
# %%

