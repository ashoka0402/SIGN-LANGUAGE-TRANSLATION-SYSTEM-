"""
GRU Temporal Model for Railway Sign Language Recognition
Processes temporal sequences of CNN features to capture gesture dynamics
"""

import torch
import torch.nn as nn


class GRUTemporalModel(nn.Module):
    """
    2-layer GRU for temporal modeling of sign language gestures.
    Takes sequence of CNN features and outputs final hidden state.
    """
    
    def __init__(
        self, 
        input_dim=512, 
        hidden_dim=256, 
        num_layers=2, 
        dropout=0.3,
        bidirectional=False
    ):
        """
        Args:
            input_dim (int): Dimension of input features (from CNN)
            hidden_dim (int): GRU hidden state dimension
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability between layers
            bidirectional (bool): Use bidirectional GRU
        """
        super(GRUTemporalModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output dimension accounting for bidirectional
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
    def forward(self, x, lengths=None):
        """
        Forward pass through GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional tensor of actual sequence lengths for packing
        
        Returns:
            output: GRU outputs of shape (batch_size, seq_len, hidden_dim)
            hidden: Final hidden state of shape (batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Pack sequences if lengths provided (for variable-length sequences)
        if lengths is not None:
            # Sort sequences by length (required for pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = x[sorted_idx]
            
            # Pack padded sequence
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True
            )
            
            # Run through GRU
            packed_output, hidden = self.gru(packed_input)
            
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, 
                batch_first=True
            )
            
            # Restore original order
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
            
        else:
            # Standard forward pass
            output, hidden = self.gru(x)
        
        # Extract final hidden state
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            # Take final layer's hidden state
            final_hidden = hidden[-1, :, :]
        
        return output, final_hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size (int): Batch size
            device: Device to create tensor on
        
        Returns:
            Hidden state tensor
        """
        num_directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * num_directions, 
            batch_size, 
            self.hidden_dim,
            device=device
        )


class LSTMTemporalModel(nn.Module):
    """
    Alternative LSTM-based temporal model.
    Can be used as drop-in replacement for GRU.
    """
    
    def __init__(
        self, 
        input_dim=512, 
        hidden_dim=256, 
        num_layers=2, 
        dropout=0.3,
        bidirectional=False
    ):
        """
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): LSTM hidden state dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            bidirectional (bool): Use bidirectional LSTM
        """
        super(LSTMTemporalModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
    def forward(self, x, lengths=None):
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths
        
        Returns:
            output: LSTM outputs
            hidden: Final hidden state (concatenated h and c)
        """
        if lengths is not None:
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = x[sorted_idx]
            
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True
            )
            
            packed_output, (h, c) = self.lstm(packed_input)
            
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, 
                batch_first=True
            )
            
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            h = h[:, unsorted_idx, :]
            c = c[:, unsorted_idx, :]
        else:
            output, (h, c) = self.lstm(x)
        
        # Extract final hidden state
        if self.bidirectional:
            h_forward = h[-2, :, :]
            h_backward = h[-1, :, :]
            final_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            final_hidden = h[-1, :, :]
        
        return output, final_hidden


class AttentionGRU(nn.Module):
    """
    GRU with attention mechanism for improved temporal modeling.
    """
    
    def __init__(
        self, 
        input_dim=512, 
        hidden_dim=256, 
        num_layers=2, 
        dropout=0.3
    ):
        """
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): GRU hidden dimension
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
        """
        super(AttentionGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, x, lengths=None):
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            lengths: Optional sequence lengths
        
        Returns:
            output: GRU outputs
            context: Attention-weighted context vector
        """
        # Run GRU
        gru_output, _ = self.gru(x)
        
        # Compute attention scores
        attention_scores = self.attention(gru_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention to get context vector
        context = torch.sum(attention_weights * gru_output, dim=1)  # (batch, hidden_dim)
        
        return gru_output, context


if __name__ == "__main__":
    # Test GRU temporal model
    print("Testing GRU Temporal Model...")
    
    # Create model
    model = GRUTemporalModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=False
    )
    
    # Test input: batch of 4 videos, 10 frames each, 512-dim features
    dummy_input = torch.randn(4, 10, 512)
    
    output, hidden = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (4, 10, 256)
    print(f"Hidden shape: {hidden.shape}")  # Should be (4, 256)
    
    # Test with variable lengths
    print("\nTesting with variable sequence lengths...")
    lengths = torch.tensor([10, 8, 6, 5])
    output, hidden = model(dummy_input, lengths=lengths)
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test bidirectional
    print("\nTesting Bidirectional GRU...")
    bi_model = GRUTemporalModel(
        input_dim=512,
        hidden_dim=256,
        bidirectional=True
    )
    output, hidden = bi_model(dummy_input)
    print(f"Bidirectional output shape: {output.shape}")  # (4, 10, 512)
    print(f"Bidirectional hidden shape: {hidden.shape}")  # (4, 512)
    
    # Test attention GRU
    print("\nTesting Attention GRU...")
    attn_model = AttentionGRU(input_dim=512, hidden_dim=256)
    output, context = attn_model(dummy_input)
    print(f"Attention output shape: {output.shape}")
    print(f"Context vector shape: {context.shape}")  # (4, 256)
    
    print("\nAll temporal model tests passed!")