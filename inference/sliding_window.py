"""
Sliding Window Module for Multi-Word Sign Language Recognition
Segments continuous video into word-level chunks using sliding window approach
"""

import os
import sys
import argparse
from collections import Counter

import cv2
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SlidingWindowSegmenter:
    """
    Segments continuous sign language video into word-level predictions
    using sliding window approach.
    """
    
    def __init__(
        self,
        predictor,
        window_size=30,  # frames
        stride=15,       # frames
        confidence_threshold=0.5,
        nms_threshold=0.3,
        min_frames=10
    ):
        """
        Args:
            predictor: WordPredictor instance
            window_size (int): Number of frames per window
            stride (int): Number of frames to slide (overlap = window_size - stride)
            confidence_threshold (float): Minimum confidence for valid prediction
            nms_threshold (float): Threshold for non-maximum suppression
            min_frames (int): Minimum frames required for a prediction
        """
        self.predictor = predictor
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_frames = min_frames
    
    def extract_frames_from_video(self, video_path):
        """
        Extract all frames from video file.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            frames: List of frames as numpy arrays
            fps: Video FPS
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        return frames, fps
    
    def create_windows(self, total_frames):
        """
        Create sliding window indices.
        
        Args:
            total_frames (int): Total number of frames
        
        Returns:
            windows: List of (start_idx, end_idx) tuples
        """
        windows = []
        start = 0
        
        while start + self.min_frames <= total_frames:
            end = min(start + self.window_size, total_frames)
            windows.append((start, end))
            
            # Stop if we've reached the end
            if end == total_frames:
                break
            
            # Slide window
            start += self.stride
        
        return windows
    
    def predict_windows(self, frames):
        """
        Run prediction on all windows.
        
        Args:
            frames: List of video frames
        
        Returns:
            predictions: List of prediction dictionaries with window info
        """
        total_frames = len(frames)
        windows = self.create_windows(total_frames)
        
        predictions = []
        
        print(f"Processing {len(windows)} windows...")
        
        for i, (start, end) in enumerate(windows):
            # Extract window frames
            window_frames = frames[start:end]
            
            if len(window_frames) < self.min_frames:
                continue
            
            # Predict
            result = self.predictor.predict_from_frames(window_frames)
            
            # Add window information
            result['window_id'] = i
            result['start_frame'] = start
            result['end_frame'] = end
            result['window_frames'] = len(window_frames)
            
            # Only keep predictions above threshold
            if result['confidence'] >= self.confidence_threshold:
                predictions.append(result)
        
        return predictions
    
    def apply_nms(self, predictions):
        """
        Apply Non-Maximum Suppression to remove duplicate predictions.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            filtered_predictions: List of predictions after NMS
        """
        if len(predictions) == 0:
            return []
        
        # Sort by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        kept_predictions = []
        
        for pred in predictions:
            # Check if this prediction overlaps significantly with any kept prediction
            should_keep = True
            
            for kept in kept_predictions:
                # Calculate overlap
                overlap = self._calculate_overlap(
                    (pred['start_frame'], pred['end_frame']),
                    (kept['start_frame'], kept['end_frame'])
                )
                
                # If high overlap and same class, suppress
                if overlap > self.nms_threshold and pred['predicted_class'] == kept['predicted_class']:
                    should_keep = False
                    break
            
            if should_keep:
                kept_predictions.append(pred)
        
        return kept_predictions
    
    def _calculate_overlap(self, window1, window2):
        """
        Calculate overlap ratio between two windows.
        
        Args:
            window1: (start, end) tuple
            window2: (start, end) tuple
        
        Returns:
            overlap_ratio: Overlap as ratio of smaller window
        """
        start1, end1 = window1
        start2, end2 = window2
        
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start >= intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start
        
        # Calculate union or use smaller window
        window1_size = end1 - start1
        window2_size = end2 - start2
        smaller_window = min(window1_size, window2_size)
        
        overlap_ratio = intersection / smaller_window
        
        return overlap_ratio
    
    def merge_consecutive_predictions(self, predictions):
        """
        Merge consecutive predictions of the same class.
        
        Args:
            predictions: List of predictions sorted by start_frame
        
        Returns:
            merged_predictions: List of merged predictions
        """
        if len(predictions) == 0:
            return []
        
        # Sort by start frame
        predictions = sorted(predictions, key=lambda x: x['start_frame'])
        
        merged = []
        current = predictions[0].copy()
        
        for pred in predictions[1:]:
            # If same class and frames are close
            if (pred['predicted_class'] == current['predicted_class'] and 
                pred['start_frame'] - current['end_frame'] < self.stride):
                
                # Merge
                current['end_frame'] = pred['end_frame']
                current['confidence'] = max(current['confidence'], pred['confidence'])
            else:
                # Save current and start new
                merged.append(current)
                current = pred.copy()
        
        # Add last prediction
        merged.append(current)
        
        return merged
    
    def get_word_sequence(self, predictions):
        """
        Extract clean word sequence from predictions.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            word_sequence: List of predicted words in order
        """
        # Sort by start frame
        predictions = sorted(predictions, key=lambda x: x['start_frame'])
        
        word_sequence = [pred['predicted_class'] for pred in predictions]
        
        return word_sequence
    
    def process_video(self, video_path):
        """
        Complete pipeline: extract frames, predict windows, apply NMS, get sequence.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            results: Dictionary containing word sequence and detailed predictions
        """
        print(f"Processing video: {video_path}")
        
        # Extract frames
        print("Extracting frames...")
        frames, fps = self.extract_frames_from_video(video_path)
        print(f"Extracted {len(frames)} frames at {fps:.2f} FPS")
        
        # Predict on windows
        raw_predictions = self.predict_windows(frames)
        print(f"Generated {len(raw_predictions)} raw predictions")
        
        # Apply NMS
        nms_predictions = self.apply_nms(raw_predictions)
        print(f"After NMS: {len(nms_predictions)} predictions")
        
        # Merge consecutive
        merged_predictions = self.merge_consecutive_predictions(nms_predictions)
        print(f"After merging: {len(merged_predictions)} predictions")
        
        # Get word sequence
        word_sequence = self.get_word_sequence(merged_predictions)
        
        results = {
            'video_path': video_path,
            'total_frames': len(frames),
            'fps': fps,
            'word_sequence': word_sequence,
            'predictions': merged_predictions,
            'raw_predictions_count': len(raw_predictions),
            'final_predictions_count': len(merged_predictions)
        }
        
        return results
    
    def visualize_timeline(self, results, output_path=None):
        """
        Create a simple text-based timeline visualization.
        
        Args:
            results: Results dictionary from process_video
            output_path: Optional path to save visualization
        """
        total_frames = results['total_frames']
        predictions = results['predictions']
        
        # Create timeline
        timeline = ['-'] * total_frames
        
        for pred in predictions:
            start = pred['start_frame']
            end = pred['end_frame']
            label = pred['predicted_class'][:3].upper()  # First 3 chars
            
            # Mark frames with label
            for i in range(start, min(end, total_frames)):
                timeline[i] = label[i % len(label)]
        
        # Print timeline in chunks
        chunk_size = 100
        print("\nTimeline Visualization:")
        print("="*80)
        
        for i in range(0, total_frames, chunk_size):
            chunk = timeline[i:i+chunk_size]
            print(f"Frame {i:5d}-{min(i+chunk_size, total_frames):5d}: {''.join(chunk)}")
        
        print("\nWord Sequence:", ' -> '.join(results['word_sequence']))
        print("="*80)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(f"Video: {results['video_path']}\n")
                f.write(f"Total Frames: {total_frames}\n")
                f.write(f"FPS: {results['fps']:.2f}\n\n")
                f.write("Timeline:\n")
                for i in range(0, total_frames, chunk_size):
                    chunk = timeline[i:i+chunk_size]
                    f.write(f"Frame {i:5d}-{min(i+chunk_size, total_frames):5d}: {''.join(chunk)}\n")
                f.write(f"\nWord Sequence: {' -> '.join(results['word_sequence'])}\n")
            
            print(f"Timeline saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-word sign language recognition with sliding window')
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                       help='List of class names')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size in frames')
    parser.add_argument('--stride', type=int, default=15,
                       help='Stride in frames')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.3,
                       help='NMS threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str,
                       help='Output path for timeline visualization')
    
    args = parser.parse_args()
    
    # Create predictor
    from predict_word import WordPredictor
    
    predictor = WordPredictor(
        model_path=args.model,
        class_names=args.classes,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Create segmenter
    segmenter = SlidingWindowSegmenter(
        predictor=predictor,
        window_size=args.window_size,
        stride=args.stride,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # Process video
    results = segmenter.process_video(args.video)
    
    # Print results
    print("\n" + "="*80)
    print("MULTI-WORD RECOGNITION RESULTS")
    print("="*80)
    print(f"Video: {results['video_path']}")
    print(f"Total Frames: {results['total_frames']}")
    print(f"FPS: {results['fps']:.2f}\n")
    
    print("Detected Word Sequence:")
    print("  " + " → ".join(results['word_sequence']))
    
    print(f"\nDetailed Predictions ({len(results['predictions'])}):")
    for i, pred in enumerate(results['predictions'], 1):
        start_time = pred['start_frame'] / results['fps']
        end_time = pred['end_frame'] / results['fps']
        print(f"  {i}. {pred['predicted_class']:<15} "
              f"[{pred['start_frame']:4d}-{pred['end_frame']:4d}] "
              f"({start_time:.2f}s - {end_time:.2f}s) "
              f"conf: {pred['confidence']:.3f}")
    
    print("="*80)
    
    # Visualize timeline
    segmenter.visualize_timeline(results, output_path=args.output)


if __name__ == "__main__":
    print("Sliding Window Segmenter Module")
    print("Usage:")
    print("  python sliding_window.py --video path/to/video.mp4 --model checkpoint.pth --classes train platform delay ...")
    
    # Uncomment to run with arguments
    # main()