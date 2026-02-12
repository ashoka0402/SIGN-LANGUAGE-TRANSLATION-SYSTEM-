"""
Word Classifier for Railway Sign Language Recognition
End-to-end model combining CNN feature extraction and GRU temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordClassifier(nn.Module):
    """
    Complete sign language word classifier.
    Combines CNN feature extraction, GRU temporal modeling, and classification head.
    """
    
    def __init__(
        self,
        num_classes,
        cnn_feature_dim=512,
        gru_hidden_dim=256,
        gru_num_layers=2,
        dropout=0.3,
        use_attention=False,
        bidirectional=False
    ):
        """
        Args:
            num_classes (int): Number of sign word classes (vocabulary size)
            cnn_feature_dim (int): Dimension of CNN features
            gru_hidden_dim (int): GRU hidden state dimension
            gru_num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            use_attention (bool): Use attention mechanism
            bidirectional (bool): Use bidirectional GRU
        """
        super(WordClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # GRU temporal model
        self.gru = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate GRU output dimension
        gru_output_dim = gru_hidden_dim * 2 if bidirectional else gru_hidden_dim
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(gru_output_dim, gru_output_dim),
                nn.Tanh(),
                nn.Linear(gru_output_dim, 1)
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc = nn.Linear(gru_output_dim, num_classes)
        
    def forward(self, features, lengths=None):
        """
        Forward pass through complete classifier.
        
        Args:
            features: CNN features of shape (batch_size, seq_len, feature_dim)
            lengths: Optional sequence lengths for packing
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
            probs: Class probabilities after softmax
        """
        # Run through GRU
        if lengths is not None:
            # Pack sequences for efficiency
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            features_sorted = features[sorted_idx]
            
            packed_input = nn.utils.rnn.pack_padded_sequence(
                features_sorted,
                lengths_sorted.cpu(),
                batch_first=True
            )
            
            packed_output, hidden = self.gru(packed_input)
            
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
            
            # Restore original order
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            output, hidden = self.gru(features)
        
        # Get representation for classification
        if self.use_attention:
            # Use attention-weighted representation
            attention_scores = self.attention(output)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)
            representation = torch.sum(attention_weights * output, dim=1)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate forward and backward final states
                hidden_forward = hidden[-2, :, :]
                hidden_backward = hidden[-1, :, :]
                representation = torch.cat([hidden_forward, hidden_backward], dim=1)
            else:
                representation = hidden[-1, :, :]
        
        # Apply dropout
        representation = self.dropout(representation)
        
        # Classification
        logits = self.fc(representation)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs
    
    def predict(self, features, lengths=None):
        """
        Predict class and confidence for inference.
        
        Args:
            features: CNN features
            lengths: Optional sequence lengths
        
        Returns:
            predicted_class: Predicted class index
            confidence: Confidence score
        """
        self.eval()
        with torch.no_grad():
            logits, probs = self.forward(features, lengths)
            confidence, predicted_class = torch.max(probs, dim=1)
        
        return predicted_class, confidence


class CompleteSignClassifier(nn.Module):
    """
    End-to-end sign language classifier including CNN feature extraction.
    This is the complete pipeline from raw frames to predictions.
    """
    
    def __init__(
        self,
        num_classes,
        cnn_type='resnet18',
        cnn_pretrained=True,
        cnn_feature_dim=512,
        gru_hidden_dim=256,
        gru_num_layers=2,
        dropout=0.3,
        use_attention=False,
        bidirectional=False,
        freeze_cnn=False
    ):
        """
        Args:
            num_classes (int): Number of sign word classes
            cnn_type (str): Type of CNN ('resnet18', 'vgg16')
            cnn_pretrained (bool): Use pretrained CNN weights
            cnn_feature_dim (int): CNN output feature dimension
            gru_hidden_dim (int): GRU hidden dimension
            gru_num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            use_attention (bool): Use attention mechanism
            bidirectional (bool): Bidirectional GRU
            freeze_cnn (bool): Freeze CNN weights during training
        """
        super(CompleteSignClassifier, self).__init__()
        
        # Import feature extractors
        from torchvision import models
        
        # CNN Feature Extractor
        if cnn_type == 'resnet18':
            resnet = models.resnet18(pretrained=cnn_pretrained)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            actual_feature_dim = 512
        elif cnn_type == 'vgg16':
            vgg = models.vgg16(pretrained=cnn_pretrained)
            self.cnn = nn.Sequential(
                vgg.features,
                vgg.avgpool,
                nn.Flatten(),
                *list(vgg.classifier.children())[:-1]
            )
            actual_feature_dim = 4096
        else:
            raise ValueError(f"Unknown CNN type: {cnn_type}")
        
        # Freeze CNN if requested
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Feature dimension adjustment if needed
        if actual_feature_dim != cnn_feature_dim:
            self.feature_projection = nn.Linear(actual_feature_dim, cnn_feature_dim)
        else:
            self.feature_projection = None
        
        # Word Classifier (GRU + FC)
        self.classifier = WordClassifier(
            num_classes=num_classes,
            cnn_feature_dim=cnn_feature_dim,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            dropout=dropout,
            use_attention=use_attention,
            bidirectional=bidirectional
        )
        
    def forward(self, frames, lengths=None):
        """
        End-to-end forward pass from frames to predictions.
        
        Args:
            frames: Input frames (batch, seq_len, channels, height, width)
            lengths: Optional sequence lengths
        
        Returns:
            logits: Class logits
            probs: Class probabilities
        """
        batch_size, seq_len, c, h, w = frames.shape
        
        # Extract features from all frames
        frames_flat = frames.view(batch_size * seq_len, c, h, w)
        features_flat = self.cnn(frames_flat)
        features_flat = features_flat.view(features_flat.size(0), -1)
        
        # Project features if needed
        if self.feature_projection is not None:
            features_flat = self.feature_projection(features_flat)
        
        # Reshape back to sequence
        features = features_flat.view(batch_size, seq_len, -1)
        
        # Classify
        logits, probs = self.classifier(features, lengths)
        
        return logits, probs
    
    def predict(self, frames, lengths=None):
        """
        Predict class and confidence.
        
        Args:
            frames: Input frames
            lengths: Optional sequence lengths
        
        Returns:
            predicted_class: Class index
            confidence: Confidence score
        """
        self.eval()
        with torch.no_grad():
            logits, probs = self.forward(frames, lengths)
            confidence, predicted_class = torch.max(probs, dim=1)
        
        return predicted_class, confidence


if __name__ == "__main__":
    # Test word classifier
    print("Testing Word Classifier...")
    
    # Define vocabulary size
    num_classes = 20  # 20 railway-related words
    
    # Create classifier
    classifier = WordClassifier(
        num_classes=num_classes,
        cnn_feature_dim=512,
        gru_hidden_dim=256,
        gru_num_layers=2,
        dropout=0.3,
        use_attention=False
    )
    
    # Test with dummy features
    # Batch of 4 videos, 10 frames each, 512-dim features
    dummy_features = torch.randn(4, 10, 512)
    
    logits, probs = classifier(dummy_features)
    print(f"Logits shape: {logits.shape}")  # (4, 20)
    print(f"Probabilities shape: {probs.shape}")  # (4, 20)
    print(f"Probabilities sum to 1: {probs.sum(dim=1)}")
    
    # Test prediction
    predicted_class, confidence = classifier.predict(dummy_features)
    print(f"\nPredicted classes: {predicted_class}")
    print(f"Confidence scores: {confidence}")
    
    # Test with attention
    print("\n\nTesting with Attention...")
    attn_classifier = WordClassifier(
        num_classes=num_classes,
        use_attention=True
    )
    logits, probs = attn_classifier(dummy_features)
    print(f"Attention classifier output shape: {logits.shape}")
    
    # Test complete end-to-end classifier
    print("\n\nTesting Complete Sign Classifier...")
    complete_model = CompleteSignClassifier(
        num_classes=num_classes,
        cnn_type='resnet18',
        cnn_pretrained=False,  # Set False for testing
        freeze_cnn=False
    )
    
    # Test with dummy frames
    # Batch of 2 videos, 10 frames each, 3 channels, 224x224
    dummy_frames = torch.randn(2, 10, 3, 224, 224)
    
    logits, probs = complete_model(dummy_frames)
    print(f"Complete model logits shape: {logits.shape}")  # (2, 20)
    print(f"Complete model probs shape: {probs.shape}")  # (2, 20)
    
    predicted_class, confidence = complete_model.predict(dummy_frames)
    print(f"\nPredicted classes: {predicted_class}")
    print(f"Confidence scores: {confidence}")
    
    # Count parameters
    total_params = sum(p.numel() for p in complete_model.parameters())
    trainable_params = sum(p.numel() for p in complete_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nAll classifier tests passed!")