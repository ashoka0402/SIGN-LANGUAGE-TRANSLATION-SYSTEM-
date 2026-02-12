"""
Complete Inference Pipeline for Railway Sign Language Recognition
End-to-end pipeline: Video → Words → LLM → Sentence → Speech
"""

import os
import sys
import argparse
import json
from datetime import datetime

import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict_word import WordPredictor
from sliding_window import SlidingWindowSegmenter


class SignLanguagePipeline:
    """
    Complete pipeline for sign language translation:
    1. Video input (webcam or file)
    2. Word recognition (single or multi-word)
    3. LLM sentence reconstruction (optional)
    4. Text-to-speech (optional)
    """
    
    def __init__(
        self,
        model_path,
        class_names,
        device='cuda',
        confidence_threshold=0.5,
        window_size=30,
        stride=15,
        use_llm=False,
        use_tts=False
    ):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            class_names (list): List of class names
            device (str): Device for inference
            confidence_threshold (float): Confidence threshold
            window_size (int): Window size for multi-word recognition
            stride (int): Stride for sliding window
            use_llm (bool): Use LLM for sentence reconstruction
            use_tts (bool): Use text-to-speech
        """
        self.class_names = class_names
        self.use_llm = use_llm
        self.use_tts = use_tts
        
        # Initialize word predictor
        self.predictor = WordPredictor(
            model_path=model_path,
            class_names=class_names,
            device=device,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize sliding window segmenter
        self.segmenter = SlidingWindowSegmenter(
            predictor=self.predictor,
            window_size=window_size,
            stride=stride,
            confidence_threshold=confidence_threshold
        )
        
        print("Sign Language Pipeline Initialized")
        print(f"  LLM Reconstruction: {'Enabled' if use_llm else 'Disabled'}")
        print(f"  Text-to-Speech: {'Enabled' if use_tts else 'Disabled'}")
    
    def process_single_word_video(self, video_path):
        """
        Process a video containing a single sign word.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            result: Prediction result dictionary
        """
        print(f"\nProcessing single-word video: {video_path}")
        
        result = self.predictor.predict(video_path, return_top_k=5)
        result['video_path'] = video_path
        result['mode'] = 'single_word'
        
        return result
    
    def process_multi_word_video(self, video_path):
        """
        Process a video containing multiple sign words.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            result: Multi-word recognition results
        """
        print(f"\nProcessing multi-word video: {video_path}")
        
        result = self.segmenter.process_video(video_path)
        result['mode'] = 'multi_word'
        
        return result
    
    def reconstruct_sentence_llm(self, word_sequence):
        """
        Use LLM to reconstruct grammatically correct sentence.
        
        Args:
            word_sequence (list): List of detected words
        
        Returns:
            sentence (str): Reconstructed sentence
        """
        if not self.use_llm:
            # Simple concatenation fallback
            return ' '.join(word_sequence)
        
        # TODO: Implement LLM API call
        # This is a placeholder - actual implementation should call LLM API
        
        keywords = ', '.join(word_sequence)
        
        # Placeholder: Simple rule-based reconstruction for railway domain
        sentence = self._rule_based_reconstruction(word_sequence)
        
        print(f"\nLLM Reconstruction:")
        print(f"  Keywords: {keywords}")
        print(f"  Sentence: {sentence}")
        
        return sentence
    
    def _rule_based_reconstruction(self, words):
        """
        Simple rule-based sentence reconstruction (fallback).
        
        Args:
            words (list): List of words
        
        Returns:
            sentence (str): Reconstructed sentence
        """
        # Railway-specific templates
        templates = {
            ('train', 'delay'): "The train is delayed.",
            ('train', 'platform'): "The train is at the platform.",
            ('train', 'arrival'): "The train is arriving.",
            ('train', 'departure'): "The train is departing.",
            ('ticket', 'price'): "What is the ticket price?",
            ('help', 'emergency'): "I need emergency help!",
            ('restroom', 'information'): "Where is the restroom?",
            ('platform', 'information'): "Where is the platform?",
        }
        
        # Check for template match
        word_tuple = tuple(sorted(words))
        for key, template in templates.items():
            if set(key).issubset(set(words)):
                return template
        
        # Default: capitalize first word and add period
        if words:
            sentence = ' '.join(words)
            sentence = sentence[0].upper() + sentence[1:] + '.'
            return sentence
        
        return ""
    
    def generate_speech(self, text, output_path=None):
        """
        Convert text to speech.
        
        Args:
            text (str): Text to convert
            output_path (str): Optional path to save audio file
        
        Returns:
            audio_path (str): Path to generated audio (if saved)
        """
        if not self.use_tts:
            print(f"Text-to-Speech disabled. Text: {text}")
            return None
        
        # TODO: Implement actual TTS
        # This is a placeholder
        print(f"\n[TTS] Would generate speech for: {text}")
        
        if output_path:
            print(f"[TTS] Would save audio to: {output_path}")
            # Actual implementation would use gTTS, pyttsx3, or cloud TTS API
        
        return output_path
    
    def process_pipeline(self, video_path, mode='auto', output_dir=None):
        """
        Run complete end-to-end pipeline.
        
        Args:
            video_path (str): Path to input video
            mode (str): 'single', 'multi', or 'auto'
            output_dir (str): Directory to save outputs
        
        Returns:
            final_result: Complete pipeline results
        """
        print("\n" + "="*80)
        print("SIGN LANGUAGE TRANSLATION PIPELINE")
        print("="*80)
        
        # Determine mode if auto
        if mode == 'auto':
            # Simple heuristic: check video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # Videos under 5 seconds treated as single word
            mode = 'single' if duration < 5 else 'multi'
            print(f"Auto-detected mode: {mode} (duration: {duration:.2f}s)")
        
        # Step 1: Word Recognition
        if mode == 'single':
            recognition_result = self.process_single_word_video(video_path)
            word_sequence = [recognition_result['predicted_class']]
        else:
            recognition_result = self.process_multi_word_video(video_path)
            word_sequence = recognition_result['word_sequence']
        
        print(f"\nDetected Words: {word_sequence}")
        
        # Step 2: LLM Sentence Reconstruction
        if len(word_sequence) > 0:
            sentence = self.reconstruct_sentence_llm(word_sequence)
        else:
            sentence = ""
            print("Warning: No words detected!")
        
        # Step 3: Text-to-Speech
        audio_path = None
        if sentence and self.use_tts and output_dir:
            audio_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            audio_path = self.generate_speech(sentence, audio_path)
        
        # Compile final results
        final_result = {
            'video_path': video_path,
            'mode': mode,
            'word_sequence': word_sequence,
            'reconstructed_sentence': sentence,
            'audio_path': audio_path,
            'recognition_details': recognition_result,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results if output directory provided
        if output_dir:
            self._save_results(final_result, output_dir)
        
        return final_result
    
    def _save_results(self, result, output_dir):
        """Save pipeline results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'result_{timestamp}.json')
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_result(self, result):
        """Print pipeline results in readable format"""
        print("\n" + "="*80)
        print("PIPELINE RESULTS")
        print("="*80)
        print(f"Video: {result['video_path']}")
        print(f"Mode: {result['mode']}")
        print(f"\nDetected Words: {' → '.join(result['word_sequence'])}")
        print(f"\nReconstructed Sentence:")
        print(f"  \"{result['reconstructed_sentence']}\"")
        
        if result['audio_path']:
            print(f"\nAudio Output: {result['audio_path']}")
        
        # Additional details for multi-word
        if result['mode'] == 'multi_word':
            details = result['recognition_details']
            print(f"\nRecognition Details:")
            print(f"  Total Frames: {details['total_frames']}")
            print(f"  FPS: {details['fps']:.2f}")
            print(f"  Predictions: {details['final_predictions_count']}")
        
        print("="*80 + "\n")
    
    def process_webcam_stream(self, buffer_size=60):
        """
        Process real-time webcam stream (interactive mode).
        
        Args:
            buffer_size (int): Number of frames to buffer before prediction
        
        Note: This is a simplified version. Production should use threading.
        """
        print("\nStarting webcam stream...")
        print("Press 'q' to quit, 's' to capture and predict")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        frame_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Buffer: {len(frame_buffer)}/{buffer_size}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 's' to predict, 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Recognition', display_frame)
            
            # Add frame to buffer
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            
            # Maintain buffer size
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and len(frame_buffer) >= 10:
                # Predict on buffered frames
                print(f"\nPredicting on {len(frame_buffer)} frames...")
                result = self.predictor.predict_from_frames(frame_buffer)
                
                print(f"Prediction: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                # Show prediction on frame
                pred_frame = frame.copy()
                cv2.putText(pred_frame, f"Predicted: {result['predicted_class']}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(pred_frame, f"Confidence: {result['confidence']:.3f}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Sign Language Recognition', pred_frame)
                cv2.waitKey(2000)  # Show for 2 seconds
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stream ended")


def main():
    parser = argparse.ArgumentParser(description='Complete Sign Language Translation Pipeline')
    
    parser.add_argument('--video', type=str,
                       help='Path to video file (omit for webcam)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                       help='List of class names')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['single', 'multi', 'auto', 'webcam'],
                       help='Processing mode')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size for multi-word')
    parser.add_argument('--stride', type=int, default=15,
                       help='Stride for sliding window')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM for sentence reconstruction')
    parser.add_argument('--use-tts', action='store_true',
                       help='Generate speech output')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SignLanguagePipeline(
        model_path=args.model,
        class_names=args.classes,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        window_size=args.window_size,
        stride=args.stride,
        use_llm=args.use_llm,
        use_tts=args.use_tts
    )
    
    # Process based on mode
    if args.mode == 'webcam' or args.video is None:
        # Webcam mode
        pipeline.process_webcam_stream()
    else:
        # File mode
        result = pipeline.process_pipeline(
            video_path=args.video,
            mode=args.mode,
            output_dir=args.output_dir
        )
        
        # Print results
        pipeline.print_result(result)


if __name__ == "__main__":
    print("Sign Language Translation Pipeline")
    print("="*80)
    print("Usage Examples:")
    print("  Single word:  python pipeline.py --video sign.mp4 --model checkpoint.pth --classes train platform ...")
    print("  Multi word:   python pipeline.py --video signs.mp4 --model checkpoint.pth --classes ... --mode multi")
    print("  With LLM:     python pipeline.py --video signs.mp4 --model checkpoint.pth --classes ... --use-llm")
    print("  Webcam:       python pipeline.py --model checkpoint.pth --classes ... --mode webcam")
    print("="*80)
    
    # Uncomment to run with arguments
    # main()