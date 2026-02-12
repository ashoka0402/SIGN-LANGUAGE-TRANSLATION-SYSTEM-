"""
API Routes for Railway Sign Language Translation System
Defines all REST endpoints for video processing and translation
"""

from flask import request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def register_routes(app):
    """Register all API routes with the Flask app"""
    
    @app.route('/api/v1/translate/video', methods=['POST'])
    def translate_video():
        """
        Translate sign language from uploaded video file
        
        Expected input:
        - video file (multipart/form-data)
        - optional: confidence_threshold (float)
        - optional: enable_tts (boolean)
        
        Returns:
        - detected_words: list of recognized sign words
        - sentence: reconstructed sentence from LLM
        - confidence_scores: confidence for each word
        - audio_file: URL to generated speech (if enabled)
        """
        
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        # Check if filename is empty
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(video_file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({
                'error': 'Invalid file type',
                'allowed_types': list(app.config['ALLOWED_EXTENSIONS'])
            }), 400
        
        # Get optional parameters
        confidence_threshold = float(request.form.get('confidence_threshold', 0.7))
        enable_tts = request.form.get('enable_tts', 'false').lower() == 'true'
        
        try:
            # Save uploaded file
            filename = secure_filename(video_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)
            
            # Import inference pipeline
            from inference.pipeline import process_video
            
            # Process video through pipeline
            result = process_video(
                video_path=filepath,
                confidence_threshold=confidence_threshold,
                enable_tts=enable_tts
            )
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({
                'error': 'Processing failed',
                'message': str(e)
            }), 500
    
    
    @app.route('/api/v1/translate/stream', methods=['POST'])
    def translate_stream():
        """
        Translate sign language from video stream frames
        
        Expected input (JSON):
        - frames: base64 encoded frames or frame URLs
        - confidence_threshold: optional threshold
        
        Returns:
        - Same as translate_video but for streaming context
        """
        
        try:
            data = request.get_json()
            
            if not data or 'frames' not in data:
                return jsonify({'error': 'No frames provided'}), 400
            
            frames = data['frames']
            confidence_threshold = data.get('confidence_threshold', 0.7)
            
            # Import streaming inference
            from inference.pipeline import process_frames
            
            # Process frames
            result = process_frames(
                frames=frames,
                confidence_threshold=confidence_threshold
            )
            
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({
                'error': 'Stream processing failed',
                'message': str(e)
            }), 500
    
    
    @app.route('/api/v1/words', methods=['GET'])
    def get_vocabulary():
        """
        Get the current railway sign vocabulary
        
        Returns:
        - words: list of all supported sign words
        - categories: words grouped by category
        - total_count: number of words in vocabulary
        """
        
        vocabulary = {
            'basic_interaction': [
                'hello', 'help', 'emergency', 'police', 'thank_you'
            ],
            'railway_operations': [
                'train', 'platform', 'arrival', 'departure', 'delay', 'late'
            ],
            'ticketing': [
                'ticket', 'reservation', 'cancel', 'confirm', 'price'
            ],
            'facilities': [
                'restroom', 'water', 'luggage', 'information'
            ]
        }
        
        all_words = []
        for category in vocabulary.values():
            all_words.extend(category)
        
        return jsonify({
            'words': all_words,
            'categories': vocabulary,
            'total_count': len(all_words)
        }), 200
    
    
    @app.route('/api/v1/reconstruct', methods=['POST'])
    def reconstruct_sentence():
        """
        Reconstruct grammatically correct sentence from word list
        Uses LLM for natural language generation
        
        Expected input (JSON):
        - words: list of detected sign words
        
        Returns:
        - sentence: grammatically correct sentence
        - keywords_used: which words were incorporated
        """
        
        try:
            data = request.get_json()
            
            if not data or 'words' not in data:
                return jsonify({'error': 'No words provided'}), 400
            
            words = data['words']
            
            if not isinstance(words, list) or len(words) == 0:
                return jsonify({'error': 'Words must be a non-empty list'}), 400
            
            # Import LLM reconstruction
            from llm.sentence_reconstruction import reconstruct_sentence
            
            # Generate sentence
            sentence = reconstruct_sentence(words)
            
            return jsonify({
                'sentence': sentence,
                'keywords_used': words
            }), 200
            
        except Exception as e:
            return jsonify({
                'error': 'Sentence reconstruction failed',
                'message': str(e)
            }), 500
    
    
    @app.route('/api/v1/tts', methods=['POST'])
    def text_to_speech():
        """
        Convert text to speech audio
        
        Expected input (JSON):
        - text: sentence to convert to speech
        
        Returns:
        - audio file (WAV format)
        """
        
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({'error': 'No text provided'}), 400
            
            text = data['text']
            
            # Import TTS module
            from tts.speech_generator import generate_speech
            
            # Generate speech and save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                audio_path = tmp.name
            
            generate_speech(text, output_path=audio_path)
            
            # Send file and clean up after
            return send_file(
                audio_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='translation.wav'
            ), 200
            
        except Exception as e:
            return jsonify({
                'error': 'TTS generation failed',
                'message': str(e)
            }), 500
    
    
    @app.route('/api/v1/model/info', methods=['GET'])
    def model_info():
        """
        Get information about the loaded model
        
        Returns:
        - model_type: CNN + GRU architecture info
        - vocabulary_size: number of classes
        - input_shape: expected input dimensions
        - version: model version
        """
        
        return jsonify({
            'model_type': 'CNN + GRU',
            'cnn_backbone': 'ResNet-18',
            'temporal_model': 'GRU (2 layers, 256 hidden)',
            'vocabulary_size': 20,
            'input_shape': [224, 224, 3],
            'frame_rate': 25,
            'version': '1.0.0'
        }), 200
    
    
    @app.route('/api/v1/predict/single', methods=['POST'])
    def predict_single_word():
        """
        Predict a single sign word from video
        Used for testing and validation
        
        Expected input:
        - video file containing ONE sign word
        
        Returns:
        - predicted_word: most likely word
        - confidence: prediction confidence
        - top_5_predictions: top 5 candidates with scores
        """
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        try:
            # Save uploaded file
            filename = secure_filename(video_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)
            
            # Import prediction module
            from inference.predict_word import predict_single_word
            
            # Predict word
            result = predict_single_word(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({
                'error': 'Prediction failed',
                'message': str(e)
            }), 500