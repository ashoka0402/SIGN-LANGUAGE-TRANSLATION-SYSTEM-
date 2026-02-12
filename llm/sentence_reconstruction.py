"""
LLM Sentence Reconstruction Module
Converts detected sign language keywords into grammatically correct sentences
using Large Language Models
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SentenceReconstructor:
    """
    Reconstructs grammatically correct sentences from sign language keywords
    using LLM APIs (OpenAI, Anthropic, or local models)
    """
    
    def __init__(
        self,
        api_provider='anthropic',  # 'openai', 'anthropic', 'local'
        api_key=None,
        model_name=None,
        domain='railway',
        temperature=0.3,
        max_tokens=100
    ):
        """
        Args:
            api_provider (str): LLM provider ('openai', 'anthropic', 'local')
            api_key (str): API key for the provider
            model_name (str): Specific model to use
            domain (str): Domain context for reconstruction
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens in response
        """
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv(f'{api_provider.upper()}_API_KEY')
        self.domain = domain
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default model names
        if model_name is None:
            if api_provider == 'openai':
                self.model_name = 'gpt-4'
            elif api_provider == 'anthropic':
                self.model_name = 'claude-3-5-sonnet-20241022'
            else:
                self.model_name = 'local-llm'
        else:
            self.model_name = model_name
        
        # Initialize client
        self.client = None
        self._initialize_client()
        
        print(f"Sentence Reconstructor initialized:")
        print(f"  Provider: {self.api_provider}")
        print(f"  Model: {self.model_name}")
        print(f"  Domain: {self.domain}")
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        
        if self.api_provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed. Run: pip install openai")
        
        elif self.api_provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Warning: anthropic package not installed. Run: pip install anthropic")
        
        elif self.api_provider == 'local':
            # Placeholder for local LLM (e.g., llama.cpp, transformers)
            print("Using local LLM (placeholder - implement your own)")
            self.client = None
    
    def create_prompt(self, keywords: List[str], context: Optional[str] = None) -> str:
        """
        Create prompt for LLM based on keywords and domain.
        
        Args:
            keywords (list): List of detected sign language keywords
            context (str): Optional additional context
        
        Returns:
            prompt (str): Formatted prompt for LLM
        """
        keywords_str = ', '.join(keywords)
        
        # Domain-specific prompts
        domain_instructions = {
            'railway': """You are generating railway station announcements and communications. 
Convert the following sign language keywords into a clear, grammatically correct sentence. 
The sentence should be natural, appropriate for a railway context, and use only the provided keywords.
Do not add extra information or hallucinate details.""",
            
            'general': """Convert the following sign language keywords into a grammatically correct sentence.
Use natural language but stay close to the meaning of the keywords provided.
Do not add information not implied by the keywords."""
        }
        
        instruction = domain_instructions.get(self.domain, domain_instructions['general'])
        
        prompt = f"""{instruction}

Keywords: {keywords_str}

Requirements:
- Use ALL provided keywords
- Create ONE clear sentence
- Keep it concise and natural
- Do not add extra information
- Maintain the railway/transportation domain context

Sentence:"""
        
        if context:
            prompt = f"{prompt}\n\nAdditional context: {context}\n\nSentence:"
        
        return prompt
    
    def reconstruct_openai(self, keywords: List[str], context: Optional[str] = None) -> Dict:
        """
        Reconstruct sentence using OpenAI API.
        
        Args:
            keywords (list): List of keywords
            context (str): Optional context
        
        Returns:
            result (dict): Contains 'sentence', 'model', 'tokens_used'
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized")
        
        prompt = self.create_prompt(keywords, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts sign language keywords into natural sentences."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            sentence = response.choices[0].message.content.strip()
            
            result = {
                'sentence': sentence,
                'model': self.model_name,
                'tokens_used': response.usage.total_tokens,
                'provider': 'openai'
            }
            
            return result
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._fallback_reconstruction(keywords)
    
    def reconstruct_anthropic(self, keywords: List[str], context: Optional[str] = None) -> Dict:
        """
        Reconstruct sentence using Anthropic Claude API.
        
        Args:
            keywords (list): List of keywords
            context (str): Optional context
        
        Returns:
            result (dict): Contains 'sentence', 'model', 'tokens_used'
        """
        if self.client is None:
            raise ValueError("Anthropic client not initialized")
        
        prompt = self.create_prompt(keywords, context)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            sentence = response.content[0].text.strip()
            
            result = {
                'sentence': sentence,
                'model': self.model_name,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'provider': 'anthropic'
            }
            
            return result
        
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return self._fallback_reconstruction(keywords)
    
    def reconstruct_local(self, keywords: List[str], context: Optional[str] = None) -> Dict:
        """
        Reconstruct sentence using local LLM.
        
        Args:
            keywords (list): List of keywords
            context (str): Optional context
        
        Returns:
            result (dict): Contains 'sentence', 'model'
        """
        # Placeholder for local LLM implementation
        # You can use transformers, llama.cpp, or other local inference
        
        print("Local LLM reconstruction not implemented - using fallback")
        return self._fallback_reconstruction(keywords)
    
    def _fallback_reconstruction(self, keywords: List[str]) -> Dict:
        """
        Fallback rule-based reconstruction when LLM is unavailable.
        
        Args:
            keywords (list): List of keywords
        
        Returns:
            result (dict): Contains 'sentence', 'model'
        """
        # Railway-specific templates
        templates = {
            ('train', 'delay'): "The train is delayed.",
            ('train', 'delayed'): "The train is delayed.",
            ('train', 'late'): "The train is late.",
            ('train', 'platform'): "The train is at the platform.",
            ('train', 'arrival'): "The train is arriving.",
            ('train', 'departure'): "The train is departing.",
            ('platform', 'delay'): "There is a delay on the platform.",
            ('ticket', 'price'): "What is the ticket price?",
            ('ticket', 'reservation'): "I need a ticket reservation.",
            ('ticket', 'cancel'): "I want to cancel my ticket.",
            ('ticket', 'confirm'): "Please confirm my ticket.",
            ('help', 'emergency'): "I need emergency help!",
            ('emergency', 'police'): "Emergency! Call the police!",
            ('restroom', 'information'): "Where is the restroom?",
            ('water', 'information'): "Where can I get water?",
            ('luggage', 'information'): "Where is the luggage area?",
            ('platform', 'information'): "Which platform?",
        }
        
        # Normalize keywords
        keywords_set = set([k.lower() for k in keywords])
        
        # Try to find matching template
        for template_keys, template_sentence in templates.items():
            if set(template_keys).issubset(keywords_set):
                return {
                    'sentence': template_sentence,
                    'model': 'rule-based-fallback',
                    'provider': 'fallback'
                }
        
        # Default: simple concatenation with capitalization
        if keywords:
            sentence = ' '.join(keywords)
            sentence = sentence[0].upper() + sentence[1:] + '.'
        else:
            sentence = "No keywords detected."
        
        return {
            'sentence': sentence,
            'model': 'simple-concatenation',
            'provider': 'fallback'
        }
    
    def reconstruct(self, keywords: List[str], context: Optional[str] = None) -> Dict:
        """
        Main reconstruction method - routes to appropriate provider.
        
        Args:
            keywords (list): List of detected sign language keywords
            context (str): Optional additional context
        
        Returns:
            result (dict): Reconstruction result with sentence and metadata
        """
        if not keywords:
            return {
                'sentence': '',
                'model': 'none',
                'provider': 'none',
                'error': 'No keywords provided'
            }
        
        print(f"\nReconstructing sentence from keywords: {keywords}")
        
        try:
            if self.api_provider == 'openai':
                result = self.reconstruct_openai(keywords, context)
            elif self.api_provider == 'anthropic':
                result = self.reconstruct_anthropic(keywords, context)
            elif self.api_provider == 'local':
                result = self.reconstruct_local(keywords, context)
            else:
                result = self._fallback_reconstruction(keywords)
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            result = self._fallback_reconstruction(keywords)
        
        result['input_keywords'] = keywords
        result['context'] = context
        
        print(f"Reconstructed: \"{result['sentence']}\"")
        
        return result
    
    def batch_reconstruct(self, keywords_list: List[List[str]]) -> List[Dict]:
        """
        Reconstruct multiple keyword sequences.
        
        Args:
            keywords_list (list): List of keyword lists
        
        Returns:
            results (list): List of reconstruction results
        """
        results = []
        
        for i, keywords in enumerate(keywords_list):
            print(f"\nProcessing sequence {i+1}/{len(keywords_list)}")
            result = self.reconstruct(keywords)
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='LLM Sentence Reconstruction for Sign Language')
    
    parser.add_argument('--keywords', type=str, nargs='+', required=True,
                       help='Sign language keywords to reconstruct')
    parser.add_argument('--provider', type=str, default='anthropic',
                       choices=['openai', 'anthropic', 'local'],
                       help='LLM provider')
    parser.add_argument('--model', type=str,
                       help='Specific model name')
    parser.add_argument('--api-key', type=str,
                       help='API key (or set via environment variable)')
    parser.add_argument('--domain', type=str, default='railway',
                       help='Domain context')
    parser.add_argument('--context', type=str,
                       help='Additional context for reconstruction')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Sampling temperature')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Create reconstructor
    reconstructor = SentenceReconstructor(
        api_provider=args.provider,
        api_key=args.api_key,
        model_name=args.model,
        domain=args.domain,
        temperature=args.temperature
    )
    
    # Reconstruct
    result = reconstructor.reconstruct(args.keywords, args.context)
    
    # Print result
    print("\n" + "="*80)
    print("SENTENCE RECONSTRUCTION RESULT")
    print("="*80)
    print(f"Keywords: {', '.join(args.keywords)}")
    print(f"Reconstructed Sentence: \"{result['sentence']}\"")
    print(f"Model: {result['model']}")
    print(f"Provider: {result['provider']}")
    if 'tokens_used' in result:
        print(f"Tokens Used: {result['tokens_used']}")
    print("="*80)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    print("LLM Sentence Reconstruction Module")
    print("="*80)
    print("Usage Examples:")
    print("  python sentence_reconstruction.py --keywords train delay platform")
    print("  python sentence_reconstruction.py --keywords help emergency --provider openai")
    print("  python sentence_reconstruction.py --keywords ticket price --context 'to Mumbai'")
    print("="*80)
    
    # Example without command line
    print("\nExample Usage:")
    reconstructor = SentenceReconstructor(api_provider='anthropic', domain='railway')
    
    examples = [
        ['train', 'delay'],
        ['help', 'emergency'],
        ['ticket', 'price', 'confirm'],
        ['platform', 'information']
    ]
    
    print("\nRunning example reconstructions (using fallback):")
    for keywords in examples:
        result = reconstructor._fallback_reconstruction(keywords)
        print(f"  {keywords} → \"{result['sentence']}\"")
    
    # Uncomment to run with arguments
    # main()