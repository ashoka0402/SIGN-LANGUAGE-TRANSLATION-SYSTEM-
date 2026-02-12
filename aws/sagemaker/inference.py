"""
SageMaker Inference Handler
Custom inference script for railway sign language model deployment
"""

import torch
import json
import io
import base64
import numpy as np
from PIL import Image
import cv2


def model_fn(model_dir):
    """
    Load the model from the model_dir
    This function is called once when the endpoint starts
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Loaded model
    """
    import sys
    sys.path.append(model_dir)
    
    # Import model architecture
    from models.classifier.word_classifier import SignLanguageClassifier
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageClassifier(num_classes=20)
    
    model_path = f'{model_dir}/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the input data
    
    Args:
        request_body: The request payload
        request_content_type: Content type of the request
        
    Returns:
        Preprocessed input tensor
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # Handle base64 encoded video frames
        if 'frames' in data:
            frames = []
            for frame_b64 in data['frames']:
                # Decode base64
                frame_bytes = base64.b64decode(frame_b64)
                # Convert to numpy array
                nparr = np.frombuffer(frame_bytes, np.uint8)
                # Decode image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            
            # Convert to tensor
            frames = np.array(frames)
            frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
            
            return frames
        
        # Handle video file path
        elif 'video_path' in data:
            frames = extract_frames_from_video(data['video_path'])
            return frames
        
        else:
            raise ValueError("Input must contain 'frames' or 'video_path'")
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Make predictions using the loaded model
    
    Args:
        input_data: Preprocessed input tensor
        model: Loaded model
        
    Returns:
        Model predictions
    """
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Move input to device
        input_data = input_data.to(device)
        
        # Add batch dimension if needed
        if input_data.dim() == 4:
            input_data = input_data.unsqueeze(0)
        
        # Forward pass
        output = model(input_data)
        
        # Get predictions
        probabilities = torch.softmax(output, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0]
        
        # Get top 5 predictions
        top5_probs, top5_classes = torch.topk(probabilities, k=5, dim=-1)
    
    return {
        'predicted_class': predicted_class.cpu().numpy(),
        'confidence': confidence.cpu().numpy(),
        'top5_classes': top5_classes.cpu().numpy(),
        'top5_probabilities': top5_probs.cpu().numpy()
    }


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output
    
    Args:
        prediction: Model prediction
        response_content_type: Desired content type
        
    Returns:
        Serialized prediction
    """
    # Vocabulary mapping (should match training)
    idx_to_word = {
        0: 'hello', 1: 'help', 2: 'emergency', 3: 'police', 4: 'thank_you',
        5: 'train', 6: 'platform', 7: 'arrival', 8: 'departure', 9: 'delay',
        10: 'late', 11: 'ticket', 12: 'reservation', 13: 'cancel', 14: 'confirm',
        15: 'price', 16: 'restroom', 17: 'water', 18: 'luggage', 19: 'information'
    }
    
    if response_content_type == 'application/json':
        # Convert predictions to words
        predicted_word = idx_to_word[int(prediction['predicted_class'][0])]
        confidence = float(prediction['confidence'][0])
        
        # Top 5 predictions
        top5 = []
        for idx, prob in zip(prediction['top5_classes'][0], 
                            prediction['top5_probabilities'][0]):
            top5.append({
                'word': idx_to_word[int(idx)],
                'confidence': float(prob)
            })
        
        result = {
            'predicted_word': predicted_word,
            'confidence': confidence,
            'top_5_predictions': top5
        }
        
        return json.dumps(result)
    
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")


def extract_frames_from_video(video_path, target_frames=25):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        target_frames: Number of frames to extract
        
    Returns:
        Tensor of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    
    frames = np.array(frames)
    frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
    
    return frames