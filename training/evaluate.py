"""
Evaluation Script for Railway Sign Language Recognition
Evaluates trained model performance and generates detailed metrics
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier.word_classifier import CompleteSignClassifier


class ModelEvaluator:
    """Evaluator for sign language recognition model"""
    
    def __init__(self, model, test_loader, class_names, device='cuda'):
        """
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            class_names: List of class names
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
        
        self.model.eval()
        
    def evaluate(self):
        """
        Run complete evaluation and return metrics
        
        Returns:
            results: Dictionary containing all evaluation metrics
        """
        print("Running evaluation...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_confidences = []
        
        with torch.no_grad():
            for frames, labels, lengths in tqdm(self.test_loader, desc='Evaluating'):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                logits, probs = self.model(frames, lengths)
                
                # Get predicted class and confidence
                confidences, preds = torch.max(probs, 1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_labels, all_preds, all_probs, all_confidences
        )
        
        return results
    
    def _calculate_metrics(self, labels, preds, probs, confidences):
        """Calculate comprehensive evaluation metrics"""
        
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Average confidence per class
        confidence_per_class = []
        for i in range(self.num_classes):
            class_mask = labels == i
            if class_mask.sum() > 0:
                avg_conf = confidences[class_mask].mean()
                confidence_per_class.append(avg_conf)
            else:
                confidence_per_class.append(0.0)
        
        # Organize results
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'avg_confidence': float(confidences.mean())
            },
            'per_class': {},
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'labels': labels.tolist(),
                'predictions': preds.tolist(),
                'confidences': confidences.tolist()
            }
        }
        
        # Per-class results
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i]),
                'avg_confidence': float(confidence_per_class[i])
            }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results in readable format"""
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy:           {results['overall']['accuracy']:.4f}")
        print(f"  Precision (macro):  {results['overall']['precision_macro']:.4f}")
        print(f"  Recall (macro):     {results['overall']['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {results['overall']['f1_macro']:.4f}")
        print(f"  Avg Confidence:     {results['overall']['avg_confidence']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10} {'Confidence':<12}")
        print("-" * 90)
        
        for class_name in self.class_names:
            metrics = results['per_class'][class_name]
            print(f"{class_name:<20} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} "
                  f"{metrics['support']:<10} "
                  f"{metrics['avg_confidence']:<12.4f}")
        
        print("="*70 + "\n")
    
    def plot_confusion_matrix(self, results, save_path=None):
        """Plot and save confusion matrix"""
        
        cm = np.array(results['confusion_matrix'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_per_class_metrics(self, results, save_path=None):
        """Plot per-class precision, recall, and F1-score"""
        
        classes = self.class_names
        precision = [results['per_class'][c]['precision'] for c in classes]
        recall = [results['per_class'][c]['recall'] for c in classes]
        f1 = [results['per_class'][c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confidence_distribution(self, results, save_path=None):
        """Plot distribution of prediction confidences"""
        
        confidences = np.array(results['predictions']['confidences'])
        labels = np.array(results['predictions']['labels'])
        preds = np.array(results['predictions']['predictions'])
        
        # Separate correct and incorrect predictions
        correct_mask = labels == preds
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall confidence distribution
        ax1.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(confidences.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidences.mean():.3f}')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Overall Confidence Distribution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Correct vs Incorrect confidence
        ax2.hist(correct_conf, bins=30, alpha=0.6, color='green', 
                label=f'Correct (μ={correct_conf.mean():.3f})', edgecolor='black')
        ax2.hist(incorrect_conf, bins=30, alpha=0.6, color='red', 
                label=f'Incorrect (μ={incorrect_conf.mean():.3f})', edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Correct vs Incorrect Predictions', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, results, output_dir):
        """Save all evaluation results and plots"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
        
        # Save plots
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(results, save_path=cm_path)
        
        metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
        self.plot_per_class_metrics(results, save_path=metrics_path)
        
        conf_path = os.path.join(output_dir, 'confidence_distribution.png')
        self.plot_confidence_distribution(results, save_path=conf_path)
        
        # Save detailed classification report
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("RAILWAY SIGN LANGUAGE RECOGNITION - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:           {results['overall']['accuracy']:.4f}\n")
            f.write(f"  Precision (macro):  {results['overall']['precision_macro']:.4f}\n")
            f.write(f"  Recall (macro):     {results['overall']['recall_macro']:.4f}\n")
            f.write(f"  F1-Score (macro):   {results['overall']['f1_macro']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 70 + "\n")
            
            for class_name in self.class_names:
                metrics = results['per_class'][class_name]
                f.write(f"{class_name:<20} "
                       f"{metrics['precision']:<12.4f} "
                       f"{metrics['recall']:<12.4f} "
                       f"{metrics['f1_score']:<12.4f} "
                       f"{metrics['support']:<10}\n")
        
        print(f"Classification report saved to {report_path}")
        print(f"\nAll results saved to {output_dir}/")


def load_model(checkpoint_path, num_classes, device='cuda'):
    """Load trained model from checkpoint"""
    
    # Create model
    model = CompleteSignClassifier(
        num_classes=num_classes,
        cnn_type='resnet18',  # Should match training config
        cnn_pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, class_names


def main():
    parser = argparse.ArgumentParser(description='Evaluate Railway Sign Language Model')
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num-classes', type=int, default=20,
                       help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, class_names = load_model(args.checkpoint, args.num_classes, device)
    
    # TODO: Load test dataset
    print("\nNOTE: You need to implement the test dataset loader!")
    print("Expected: test_loader with (frames, labels, lengths)")
    
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create evaluator
    # evaluator = ModelEvaluator(model, test_loader, class_names, device)
    
    # Run evaluation
    # results = evaluator.evaluate()
    
    # Print results
    # evaluator.print_results(results)
    
    # Save results
    # evaluator.save_results(results, args.output_dir)
    
    print("\nEvaluation script setup complete!")
    print("Next steps:")
    print("1. Implement test dataset loader")
    print("2. Uncomment and run evaluation code")
    print("3. Check results in:", args.output_dir)


if __name__ == "__main__":
    main()