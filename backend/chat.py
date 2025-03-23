import boto3
import streamlit as st
import json
import random
from typing import Optional, Dict, Any
from botocore.exceptions import NoCredentialsError, CredentialRetrievalError

# Model configurations
DEFAULT_MODEL_CONFIG = {
    "model_id": "amazon.nova-micro-v1:0",  # Using full Nova model for better processing
    "inference_config": {
        "temperature": 0.7,        # Default temperature
        "topP": 0.9,               # Control response diversity (changed from top_p)
        "maxTokens": 1024,         # Maximum response length (changed from max_tokens)
        "stopSequences": []        # Optional stop sequences (changed from stop_sequences)
    }
}

# Mock responses for development when AWS credentials aren't available
MOCK_RESPONSES = [
    "नमस्ते! मैं आपकी मदद कैसे कर सकता हूँ? (Hello! How can I help you?)",
    "हिंदी एक बहुत ही सुंदर भाषा है। (Hindi is a very beautiful language.)",
    "आपका दिन शुभ हो! (Have a good day!)",
    "हिंदी सीखना बहुत मज़ेदार है। (Learning Hindi is very fun.)",
    "मैं आपके साथ हिंदी अभ्यास कर सकता हूँ। (I can practice Hindi with you.)"
]

class BedrockChat:
    def __init__(self, model_id: str = DEFAULT_MODEL_CONFIG["model_id"], use_mock: bool = False):
        """Initialize Bedrock chat client"""
        self.model_id = model_id
        self.use_mock = use_mock
        
        # Try to initialize the Bedrock client
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
            # We can't easily test credentials with bedrock-runtime without making an actual API call
            # So we'll just check if the client initializes without errors
            print("Bedrock client initialized. Credentials will be tested on first API call.")
        except (NoCredentialsError, CredentialRetrievalError) as e:
            print(f"AWS credentials not found: {str(e)}. Enabling mock mode.")
            self.use_mock = True
        except Exception as e:
            print(f"Error initializing Bedrock client: {str(e)}. Enabling mock mode.")
            self.use_mock = True

    def generate_response(self, message: str, inference_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate a response using Amazon Bedrock or mock response if credentials aren't available"""
        # Use mock mode if enabled or if we encounter credential issues
        if self.use_mock:
            print("Using mock mode - returning predefined response")
            return random.choice(MOCK_RESPONSES)
            
        if inference_config is None:
            inference_config = DEFAULT_MODEL_CONFIG["inference_config"]

        messages = [{
            "role": "user",
            "content": [{"text": message}]
        }]

        try:
            print(f"Sending request to Bedrock with model: {self.model_id}")
            print(f"Message: {message[:50]}...")
            print(f"InferenceConfig: {json.dumps(inference_config)}")
            
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
            
            if 'output' in response and 'message' in response['output'] and 'content' in response['output']['message']:
                return response['output']['message']['content'][0]['text']
            else:
                print(f"Unexpected response structure: {json.dumps(response)[:200]}...")
                return None
            
        except (NoCredentialsError, CredentialRetrievalError):
            print("Credentials error during request. Switching to mock mode.")
            self.use_mock = True
            return self.generate_response(message, inference_config)
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            print(error_message)  # Print to console for debugging
            try:
                st.error(error_message)  # Use streamlit if available
            except:
                pass  # Ignore if not in a streamlit context
            return None


if __name__ == "__main__":
    chat = BedrockChat()
    while True:
        user_input = input("You: ")
        if user_input.lower() == '/exit':
            break
        response = chat.generate_response(user_input)
        print("Bot:", response)
