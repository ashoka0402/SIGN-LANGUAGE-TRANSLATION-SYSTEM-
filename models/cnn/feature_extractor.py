"""
CNN Feature Extractor for Railway Sign Language Recognition
Extracts spatial features from video frames using pretrained ResNet-18
"""

import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for sign language frames.
    Uses pretrained ResNet-18 with final classification layer removed.
    """
    
    def __init__(self, pretrained=True, feature_dim=512):
        """
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            feature_dim (int): Dimension of output feature vector
        """
        super(FeatureExtractor, self).__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet-18 outputs 512-dim features before FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_dim = feature_dim
        
        # Optional: Add adaptive pooling if needed
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Forward pass through feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, seq_len, channels, height, width) for video
        
        Returns:
            features: Tensor of shape (batch_size, feature_dim) 
                     or (batch_size, seq_len, feature_dim) for video
        """
        # Handle both single frames and video sequences
        if len(x.shape) == 5:  # Video: (batch, seq_len, C, H, W)
            batch_size, seq_len, c, h, w = x.shape
            
            # Reshape to process all frames at once
            x = x.view(batch_size * seq_len, c, h, w)
            
            # Extract features
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            
            # Reshape back to sequence format
            features = features.view(batch_size, seq_len, self.feature_dim)
            
        else:  # Single frame: (batch, C, H, W)
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        
        return features
    
    def freeze_backbone(self):
        """Freeze feature extractor weights for fine-tuning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze feature extractor weights"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


class VGGFeatureExtractor(nn.Module):
    """
    Alternative VGG-16 based feature extractor.
    Can be used as drop-in replacement for ResNet.
    """
    
    def __init__(self, pretrained=True, feature_dim=4096):
        """
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            feature_dim (int): Dimension of output feature vector (4096 for VGG)
        """
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pretrained VGG-16
        vgg = models.vgg16(pretrained=pretrained)
        
        # Use features + classifier up to second-to-last layer
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        
        # VGG classifier without final layer
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        Forward pass through VGG feature extractor.
        
        Args:
            x: Input tensor (batch, C, H, W) or (batch, seq_len, C, H, W)
        
        Returns:
            features: (batch, 4096) or (batch, seq_len, 4096)
        """
        if len(x.shape) == 5:  # Video sequence
            batch_size, seq_len, c, h, w = x.shape
            x = x.view(batch_size * seq_len, c, h, w)
            
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            features = self.classifier(x)
            
            features = features.view(batch_size, seq_len, self.feature_dim)
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            features = self.classifier(x)
        
        return features
    
    def freeze_backbone(self):
        """Freeze feature extractor weights"""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze feature extractor weights"""
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test feature extractor
    print("Testing ResNet Feature Extractor...")
    
    # Create model
    extractor = FeatureExtractor(pretrained=False)
    
    # Test single frame
    dummy_frame = torch.randn(4, 3, 224, 224)  # Batch of 4 frames
    features = extractor(dummy_frame)
    print(f"Single frame output shape: {features.shape}")  # Should be (4, 512)
    
    # Test video sequence
    dummy_video = torch.randn(2, 10, 3, 224, 224)  # 2 videos, 10 frames each
    features = extractor(dummy_video)
    print(f"Video sequence output shape: {features.shape}")  # Should be (2, 10, 512)
    
    print("\nTesting VGG Feature Extractor...")
    vgg_extractor = VGGFeatureExtractor(pretrained=False)
    
    features_vgg = vgg_extractor(dummy_video)
    print(f"VGG output shape: {features_vgg.shape}")  # Should be (2, 10, 4096)
    
    print("\nFeature extraction tests passed!")