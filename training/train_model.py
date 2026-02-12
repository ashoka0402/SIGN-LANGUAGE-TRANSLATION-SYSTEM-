"""
Training Script for Railway Sign Language Recognition
Trains the complete sign language word classifier with CNN + GRU architecture
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier.word_classifier import CompleteSignClassifier


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class Trainer:
    """Trainer class for sign language model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        checkpoint_dir='checkpoints',
        log_dir='logs',
        num_classes=20,
        class_names=None
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            num_classes: Number of classes
            class_names: List of class names
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.checkpoint_dir = checkpoint_dir
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (frames, labels, lengths) in enumerate(pbar):
            # Move to device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, probs = self.model(frames, lengths)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(probs, 1)
            running_loss += loss.item() * frames.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += frames.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_corrects.double().item() / total_samples
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # For per-class accuracy
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for frames, labels, lengths in pbar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, probs = self.model(frames, lengths)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Statistics
                _, preds = torch.max(probs, 1)
                running_loss += loss.item() * frames.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += frames.size(0)
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (preds[i] == labels[i]).item()
                    class_total[label] += 1
                
                pbar.set_postfix({
                    'loss': running_loss / total_samples,
                    'acc': running_corrects.double().item() / total_samples
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {self.class_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, num_epochs, early_stopping_patience=10):
        """
        Main training loop
        
        Args:
            num_epochs (int): Number of epochs to train
            early_stopping_patience (int): Patience for early stopping
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        best_val_acc = 0.0
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best validation accuracy: {best_val_acc:.4f}")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping check
            if early_stopping(val_acc, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {early_stopping.best_epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save final checkpoint and history
        self.save_checkpoint(epoch, is_best=False, is_final=True)
        self.save_history()
        
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'class_names': self.class_names
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"  Saved best model to {path}")
        
        if is_final:
            path = os.path.join(self.checkpoint_dir, 'final_model.pth')
            torch.save(checkpoint, path)
            print(f"Saved final model to {path}")
        
        if not is_best and not is_final:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Saved training history to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Railway Sign Language Recognition Model')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=20, help='Number of sign classes')
    parser.add_argument('--cnn-type', type=str, default='resnet18', choices=['resnet18', 'vgg16'])
    parser.add_argument('--gru-hidden', type=int, default=256, help='GRU hidden dimension')
    parser.add_argument('--gru-layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional GRU')
    parser.add_argument('--attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--freeze-cnn', action='store_true', help='Freeze CNN backbone')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    # Output parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Tensorboard log directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Load actual dataset
    # For now, this is a placeholder
    print("NOTE: You need to implement the dataset loader!")
    print("Expected: train_loader and val_loader with (frames, labels, lengths)")
    
    # Create model
    print("\nCreating model...")
    model = CompleteSignClassifier(
        num_classes=args.num_classes,
        cnn_type=args.cnn_type,
        cnn_pretrained=True,
        gru_hidden_dim=args.gru_hidden,
        gru_num_layers=args.gru_layers,
        dropout=args.dropout,
        use_attention=args.attention,
        bidirectional=args.bidirectional,
        freeze_cnn=args.freeze_cnn
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # TODO: Create actual data loaders
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nDataset loaders need to be implemented!")
    print("See preprocessing/video_loader.py for dataset implementation")
    
    # Create trainer
    # trainer = Trainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     device=device,
    #     checkpoint_dir=args.checkpoint_dir,
    #     log_dir=args.log_dir,
    #     num_classes=args.num_classes
    # )
    
    # Train
    # history = trainer.train(num_epochs=args.epochs, early_stopping_patience=args.patience)
    
    print("\nTraining script setup complete!")
    print("Next steps:")
    print("1. Implement dataset in preprocessing/video_loader.py")
    print("2. Uncomment and run the training code")
    print("3. Monitor with: tensorboard --logdir logs")


if __name__ == "__main__":
    main()