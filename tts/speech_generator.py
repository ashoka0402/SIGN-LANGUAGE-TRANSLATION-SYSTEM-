"""
Text-to-Speech Module for Railway Sign Language System
Converts reconstructed sentences into spoken audio output
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SpeechGenerator:
    """
    Generates speech from text using various TTS engines:
    - gTTS (Google Text-to-Speech) - cloud-based, free
    - pyttsx3 - offline, platform-independent
    - ElevenLabs - high-quality, cloud-based
    - AWS Polly - cloud-based
    """
    
    def __init__(
        self,
        engine='gtts',  # 'gtts', 'pyttsx3', 'elevenlabs', 'polly'
        language='en',
        voice=None,
        rate=150,
        volume=1.0,
        api_key=None
    ):
        """
        Args:
            engine (str): TTS engine to use
            language (str): Language code (e.g., 'en', 'hi', 'en-IN')
            voice (str): Specific voice name (engine-dependent)
            rate (int): Speech rate (words per minute, for pyttsx3)
            volume (float): Volume (0.0 to 1.0, for pyttsx3)
            api_key (str): API key for cloud services
        """
        self.engine_name = engine
        self.language = language
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.api_key = api_key or os.getenv(f'{engine.upper()}_API_KEY')
        
        self.engine = None
        self._initialize_engine()
        
        print(f"Speech Generator initialized:")
        print(f"  Engine: {self.engine_name}")
        print(f"  Language: {self.language}")
        if self.voice:
            print(f"  Voice: {self.voice}")
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine"""
        
        if self.engine_name == 'gtts':
            try:
                from gtts import gTTS
                self.engine = 'gtts'
                print("  Status: gTTS ready")
            except ImportError:
                print("  Warning: gTTS not installed. Run: pip install gtts")
                self.engine = None
        
        elif self.engine_name == 'pyttsx3':
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                
                # Set properties
                self.engine.setProperty('rate', self.rate)
                self.engine.setProperty('volume', self.volume)
                
                # Set voice if specified
                if self.voice:
                    voices = self.engine.getProperty('voices')
                    for v in voices:
                        if self.voice.lower() in v.name.lower():
                            self.engine.setProperty('voice', v.id)
                            break
                
                print("  Status: pyttsx3 ready")
            except ImportError:
                print("  Warning: pyttsx3 not installed. Run: pip install pyttsx3")
                self.engine = None
            except Exception as e:
                print(f"  Warning: pyttsx3 initialization failed: {e}")
                self.engine = None
        
        elif self.engine_name == 'elevenlabs':
            try:
                from elevenlabs import generate, save
                self.engine = 'elevenlabs'
                print("  Status: ElevenLabs ready")
            except ImportError:
                print("  Warning: elevenlabs not installed. Run: pip install elevenlabs")
                self.engine = None
        
        elif self.engine_name == 'polly':
            try:
                import boto3
                self.engine = boto3.client('polly')
                print("  Status: AWS Polly ready")
            except ImportError:
                print("  Warning: boto3 not installed. Run: pip install boto3")
                self.engine = None
            except Exception as e:
                print(f"  Warning: AWS Polly initialization failed: {e}")
                self.engine = None
    
    def generate_gtts(self, text: str, output_path: str) -> bool:
        """
        Generate speech using Google Text-to-Speech.
        
        Args:
            text (str): Text to convert
            output_path (str): Path to save audio file
        
        Returns:
            success (bool): True if successful
        """
        try:
            from gtts import gTTS
            
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(output_path)
            
            print(f"Audio saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generating speech with gTTS: {e}")
            return False
    
    def generate_pyttsx3(self, text: str, output_path: str) -> bool:
        """
        Generate speech using pyttsx3 (offline).
        
        Args:
            text (str): Text to convert
            output_path (str): Path to save audio file
        
        Returns:
            success (bool): True if successful
        """
        if self.engine is None:
            print("pyttsx3 engine not initialized")
            return False
        
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            
            print(f"Audio saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generating speech with pyttsx3: {e}")
            return False
    
    def generate_elevenlabs(self, text: str, output_path: str) -> bool:
        """
        Generate speech using ElevenLabs API.
        
        Args:
            text (str): Text to convert
            output_path (str): Path to save audio file
        
        Returns:
            success (bool): True if successful
        """
        try:
            from elevenlabs import generate, save, set_api_key
            
            if self.api_key:
                set_api_key(self.api_key)
            
            # Use default voice if not specified
            voice = self.voice or "Rachel"
            
            audio = generate(
                text=text,
                voice=voice,
                model="eleven_monolingual_v1"
            )
            
            save(audio, output_path)
            
            print(f"Audio saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generating speech with ElevenLabs: {e}")
            return False
    
    def generate_polly(self, text: str, output_path: str) -> bool:
        """
        Generate speech using AWS Polly.
        
        Args:
            text (str): Text to convert
            output_path (str): Path to save audio file
        
        Returns:
            success (bool): True if successful
        """
        if self.engine is None:
            print("AWS Polly client not initialized")
            return False
        
        try:
            # Map language to voice
            voice_map = {
                'en': 'Joanna',
                'en-US': 'Joanna',
                'en-GB': 'Emma',
                'en-IN': 'Aditi',
                'hi': 'Aditi'
            }
            
            voice_id = self.voice or voice_map.get(self.language, 'Joanna')
            
            response = self.engine.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_id
            )
            
            # Save audio stream to file
            with open(output_path, 'wb') as f:
                f.write(response['AudioStream'].read())
            
            print(f"Audio saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generating speech with AWS Polly: {e}")
            return False
    
    def generate(self, text: str, output_path: Optional[str] = None, play=False) -> Optional[str]:
        """
        Generate speech from text using the configured engine.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save audio file (auto-generated if None)
            play (bool): Play audio immediately after generation
        
        Returns:
            output_path (str): Path to generated audio file, or None if failed
        """
        if not text:
            print("No text provided for speech generation")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = 'mp3' if self.engine_name in ['gtts', 'polly', 'elevenlabs'] else 'wav'
            output_path = f"speech_output_{timestamp}.{extension}"
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating speech for: \"{text}\"")
        
        # Generate based on engine
        success = False
        
        if self.engine_name == 'gtts':
            success = self.generate_gtts(text, output_path)
        elif self.engine_name == 'pyttsx3':
            success = self.generate_pyttsx3(text, output_path)
        elif self.engine_name == 'elevenlabs':
            success = self.generate_elevenlabs(text, output_path)
        elif self.engine_name == 'polly':
            success = self.generate_polly(text, output_path)
        else:
            print(f"Unknown TTS engine: {self.engine_name}")
            return None
        
        if not success:
            return None
        
        # Play audio if requested
        if play and os.path.exists(output_path):
            self.play_audio(output_path)
        
        return output_path
    
    def play_audio(self, audio_path: str):
        """
        Play audio file using system player.
        
        Args:
            audio_path (str): Path to audio file
        """
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return
        
        try:
            # Try different audio playback methods
            import platform
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                os.system(f'afplay "{audio_path}"')
            elif system == 'Windows':
                os.system(f'start "" "{audio_path}"')
            else:  # Linux
                # Try common players
                for player in ['mpg123', 'ffplay', 'vlc', 'mplayer']:
                    if os.system(f'which {player} > /dev/null 2>&1') == 0:
                        os.system(f'{player} "{audio_path}"')
                        break
        
        except Exception as e:
            print(f"Could not play audio: {e}")
    
    def batch_generate(self, text_list: list, output_dir: str = 'audio_output') -> list:
        """
        Generate speech for multiple texts.
        
        Args:
            text_list (list): List of text strings
            output_dir (str): Directory to save audio files
        
        Returns:
            output_paths (list): List of generated audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        
        for i, text in enumerate(text_list, 1):
            print(f"\nGenerating {i}/{len(text_list)}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = 'mp3' if self.engine_name in ['gtts', 'polly', 'elevenlabs'] else 'wav'
            output_path = os.path.join(output_dir, f"speech_{i}_{timestamp}.{extension}")
            
            result = self.generate(text, output_path)
            output_paths.append(result)
        
        return output_paths
    
    def list_voices(self):
        """List available voices for the current engine"""
        
        if self.engine_name == 'pyttsx3' and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                print("\nAvailable voices:")
                for i, voice in enumerate(voices):
                    print(f"  {i+1}. {voice.name} ({voice.id})")
            except Exception as e:
                print(f"Could not list voices: {e}")
        else:
            print(f"Voice listing not supported for {self.engine_name}")


def main():
    parser = argparse.ArgumentParser(description='Text-to-Speech Generator')
    
    parser.add_argument('--text', type=str, required=True,
                       help='Text to convert to speech')
    parser.add_argument('--engine', type=str, default='gtts',
                       choices=['gtts', 'pyttsx3', 'elevenlabs', 'polly'],
                       help='TTS engine')
    parser.add_argument('--language', type=str, default='en',
                       help='Language code (e.g., en, hi, en-IN)')
    parser.add_argument('--voice', type=str,
                       help='Voice name (engine-specific)')
    parser.add_argument('--output', type=str,
                       help='Output audio file path')
    parser.add_argument('--play', action='store_true',
                       help='Play audio after generation')
    parser.add_argument('--list-voices', action='store_true',
                       help='List available voices')
    parser.add_argument('--rate', type=int, default=150,
                       help='Speech rate for pyttsx3')
    parser.add_argument('--volume', type=float, default=1.0,
                       help='Volume for pyttsx3 (0.0-1.0)')
    parser.add_argument('--api-key', type=str,
                       help='API key for cloud services')
    
    args = parser.parse_args()
    
    # Create speech generator
    generator = SpeechGenerator(
        engine=args.engine,
        language=args.language,
        voice=args.voice,
        rate=args.rate,
        volume=args.volume,
        api_key=args.api_key
    )
    
    # List voices if requested
    if args.list_voices:
        generator.list_voices()
        return
    
    # Generate speech
    output_path = generator.generate(
        text=args.text,
        output_path=args.output,
        play=args.play
    )
    
    if output_path:
        print(f"\n✓ Speech generation successful!")
        print(f"  Output: {output_path}")
    else:
        print("\n✗ Speech generation failed")


if __name__ == "__main__":
    print("Text-to-Speech Generator Module")
    print("="*80)
    print("Usage Examples:")
    print('  python speech_generator.py --text "The train is delayed" --engine gtts')
    print('  python speech_generator.py --text "Platform 5" --engine pyttsx3 --play')
    print('  python speech_generator.py --text "Emergency help needed" --engine elevenlabs')
    print('  python speech_generator.py --list-voices --engine pyttsx3')
    print("="*80)
    
    # Example without command line
    print("\nExample Usage:")
    generator = SpeechGenerator(engine='gtts', language='en')
    
    examples = [
        "The train is delayed.",
        "Please proceed to platform 5.",
        "Emergency assistance required."
    ]
    
    print("\nExample texts (will not actually generate audio):")
    for text in examples:
        print(f"  - \"{text}\"")
    
    print("\nTo generate audio, run with --text argument or uncomment main()")
    
    # Uncomment to run with arguments
    # main()