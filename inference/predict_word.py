"""
Word Prediction Module for Railway Sign Language Recognition
Handles single word prediction from video input
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier.word_classifier import CompleteSignClassifier


class WordPredictor:
    """
    Predicts sign language words from video input.
    Handles preprocessing and inference for single sign videos.
    """
    
    def __init__(
        self,
        model_path,
        class_names,
        device='cuda',
        confidence_threshold=0.5,
        frame_size=(224, 224),
        target_fps=25
    ):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            class_names (list): List of class names in order
            device (str): Device to run inference on
            confidence_threshold (float): Minimum confidence for valid prediction
            frame_size (tuple): Target frame size (height, width)
            target_fps (int): Target frames per second for processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.confidence_threshold = confidence_threshold
        self.frame_size = frame_size
        self.target_fps = target_fps
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"WordPredictor initialized on {self.device}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        
        # Create model
        model = CompleteSignClassifier(
            num_classes=self.num_classes,
            cnn_type='resnet18',
            cnn_pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        
        return model
    
    def preprocess_video(self, video_path):
        """
        Preprocess video file for inference.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            frames_tensor: Preprocessed frames tensor (1, seq_len, 3, H, W)
            num_frames: Number of frames extracted
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        frame_skip = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on target FPS
            if frame_count % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame_resized = cv2.resize(frame_rgb, self.frame_size)
                
                # Normalize to [0, 1]
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame_normalized = (frame_normalized - mean) / std
                
                frames.append(frame_normalized)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        # Convert to tensor: (seq_len, H, W, C) -> (seq_len, C, H, W)
        frames_array = np.array(frames)
        frames_tensor = torch.FloatTensor(frames_array).permute(0, 3, 1, 2)
        
        # Add batch dimension: (1, seq_len, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor, len(frames)
    
    def preprocess_frames(self, frames):
        """
        Preprocess numpy array of frames.
        
        Args:
            frames: Numpy array of shape (seq_len, H, W, C) or list of frames
        
        Returns:
            frames_tensor: Preprocessed tensor (1, seq_len, 3, H, W)
        """
        processed_frames = []
        
        for frame in frames:
            # Ensure RGB
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Resize
            frame_resized = cv2.resize(frame, self.frame_size)
            
            # Normalize
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame_normalized = (frame_normalized - mean) / std
            
            processed_frames.append(frame_normalized)
        
        # Convert to tensor
        frames_array = np.array(processed_frames)
        frames_tensor = torch.FloatTensor(frames_array).permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor
    
    def predict(self, video_path, return_top_k=3):
        """
        Predict sign word from video file.
        
        Args:
            video_path (str): Path to video file
            return_top_k (int): Number of top predictions to return
        
        Returns:
            result: Dictionary containing prediction results
        """
        # Preprocess video
        frames_tensor, num_frames = self.preprocess_video(video_path)
        frames_tensor = frames_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits, probs = self.model(frames_tensor)
        
        # Get predictions
        probs = probs.cpu().numpy()[0]  # Remove batch dimension
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[-return_top_k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            predictions.append({
                'class': self.class_names[idx],
                'confidence': float(probs[idx]),
                'index': int(idx)
            })
        
        # Determine if prediction is valid
        top_confidence = predictions[0]['confidence']
        is_valid = top_confidence >= self.confidence_threshold
        
        result = {
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'is_valid': is_valid,
            'num_frames': num_frames,
            'top_k_predictions': predictions,
            'all_probabilities': probs.tolist()
        }
        
        return result
    
    def predict_from_frames(self, frames, return_top_k=3):
        """
        Predict from numpy array of frames.
        
        Args:
            frames: Numpy array or list of frames
            return_top_k (int): Number of top predictions
        
        Returns:
            result: Prediction results dictionary
        """
        # Preprocess frames
        frames_tensor = self.preprocess_frames(frames)
        frames_tensor = frames_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits, probs = self.model(frames_tensor)
        
        probs = probs.cpu().numpy()[0]
        
        # Get top-k
        top_k_indices = np.argsort(probs)[-return_top_k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            predictions.append({
                'class': self.class_names[idx],
                'confidence': float(probs[idx]),
                'index': int(idx)
            })
        
        top_confidence = predictions[0]['confidence']
        is_valid = top_confidence >= self.confidence_threshold
        
        result = {
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'is_valid': is_valid,
            'num_frames': len(frames),
            'top_k_predictions': predictions,
            'all_probabilities': probs.tolist()
        }
        
        return result
    
    def predict_batch(self, video_paths):
        """
        Predict multiple videos in batch.
        
        Args:
            video_paths (list): List of video file paths
        
        Returns:
            results: List of prediction results
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                result['video_path'] = video_path
                results.append(result)
            except Exception as e:
                results.append({
                    'video_path': video_path,
                    'error': str(e),
                    'is_valid': False
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict sign language word from video')
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                       help='List of class names')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Minimum confidence threshold')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = WordPredictor(
        model_path=args.model,
        class_names=args.classes,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Predict
    print(f"\nProcessing video: {args.video}")
    result = predictor.predict(args.video, return_top_k=args.top_k)
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Valid Prediction: {'Yes' if result['is_valid'] else 'No'}")
    print(f"Frames Processed: {result['num_frames']}")
    
    print(f"\nTop {args.top_k} Predictions:")
    for i, pred in enumerate(result['top_k_predictions'], 1):
        print(f"  {i}. {pred['class']:<15} - {pred['confidence']:.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage without command line
    print("Word Predictor Module")
    print("Usage:")
    print("  python predict_word.py --video path/to/video.mp4 --model path/to/checkpoint.pth --classes train platform delay ...")
    print("\nOr import and use programmatically:")
    print("  from predict_word import WordPredictor")
    print("  predictor = WordPredictor(model_path, class_names)")
    print("  result = predictor.predict('video.mp4')")
    
    # Uncomment to run with arguments
    # main()