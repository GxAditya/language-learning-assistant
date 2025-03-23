import os
import re
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from backend.chat import BedrockChat

class TranscriptProcessor:
    def __init__(self, transcript_dir: str = "./transcripts"):
        """Initialize the transcript processor"""
        self.transcript_dir = transcript_dir
        # Initialize Bedrock chat for LLM-based processing
        self.bedrock_chat = BedrockChat()
    
    def load_transcript(self, filename: str) -> Optional[str]:
        """Load a transcript from a file as plain text"""
        try:
            filepath = filename
            if not os.path.exists(filepath) and os.path.exists(os.path.join(self.transcript_dir, filename)):
                filepath = os.path.join(self.transcript_dir, filename)
                
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return None
                
            with open(filepath, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
                
            return transcript_text
        except Exception as e:
            print(f"Error loading transcript: {str(e)}")
            return None
    
    def is_hindi(self, text: str) -> bool:
        """Check if text contains Hindi characters"""
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(devanagari_pattern.search(text))
    
    def generate_language_pairs(self, transcript_text: str) -> List[Dict]:
        """Generate Hindi-English language pairs using Nova Micro"""
        # Create prompt for Nova Micro
        prompt = f"""
        Please analyze this Hindi language transcript and extract language pairs.
        For each significant Hindi phrase, provide its English translation.
        Format each pair as a separate JSON object in an array.
        
        Transcript:
        {transcript_text[:3000]}  # Limit text length to avoid token limits
        
        Format your response as:
        ```json
        [
          {{
            "hindi": "Hindi phrase 1",
            "english": "English translation 1"
          }},
          {{
            "hindi": "Hindi phrase 2",
            "english": "English translation 2" 
          }},
          ...
        ]
        ```
        
        Identify at least 10 important Hindi phrases and their translations.
        Only include Hindi phrases actually present in the transcript.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            # Try to extract JSON array
            try:
                # Find JSON between ```json and ```
                json_match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON array directly
                    json_str = re.search(r'\[\s*\{\s*"hindi"', response)
                    if json_str:
                        # Find the matching end bracket
                        start_idx = json_str.start()
                        # Simple approach to find the matching closing bracket
                        count = 0
                        for i, char in enumerate(response[start_idx:]):
                            if char == '[':
                                count += 1
                            elif char == ']':
                                count -= 1
                                if count == 0:
                                    json_str = response[start_idx:start_idx+i+1]
                                    break
                    else:
                        # Fallback to the entire response
                        json_str = response
                
                # Clean the string to ensure valid JSON
                json_str = json_str.strip()
                
                # Extract [...] if present anywhere in the text
                bracket_match = re.search(r'\[[\s\S]*\]', json_str)
                if bracket_match:
                    json_str = bracket_match.group(0)
                
                # Parse JSON
                pairs = json.loads(json_str)
                
                # Add timestamp (not available in generated data)
                for i, pair in enumerate(pairs):
                    pair['timestamp'] = i * 10  # Arbitrary timestamp
                    pair['is_generated'] = True
                
                return pairs
            except Exception as e:
                print(f"Error parsing language pairs from LLM response: {str(e)}")
                print(f"Response was: {response}")
                return []
        
        return []
    
    def generate_dialogues(self, transcript_text: str) -> List[Dict]:
        """Generate dialogues using Nova Micro"""
        # Create prompt for Nova Micro
        prompt = f"""
        Please create structured dialogues from this Hindi language transcript.
        Extract or create realistic conversations with Hindi phrases and their English translations.
        
        Transcript:
        {transcript_text[:3000]}  # Limit text length to avoid token limits
        
        Format your response as:
        ```json
        [
          {{
            "dialogue": [
              {{"text": "Hindi line 1", "is_hindi": true}},
              {{"text": "English translation 1", "is_hindi": false}},
              {{"text": "Hindi line 2", "is_hindi": true}},
              {{"text": "English translation 2", "is_hindi": false}}
            ]
          }},
          {{
            "dialogue": [
              ...
            ]
          }}
        ]
        ```
        
        Create at least 3 separate dialogues, each with 4-6 lines.
        Each dialogue should represent a natural conversation in Hindi with English translations.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            # Try to extract JSON array
            try:
                # Find JSON between ```json and ```
                json_match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON array directly
                    array_match = re.search(r'\[\s*\{\s*"dialogue"', response)
                    if array_match:
                        # Find the matching end bracket
                        start_idx = array_match.start()
                        # Simple approach to find the matching closing bracket
                        count = 0
                        for i, char in enumerate(response[start_idx:]):
                            if char == '[':
                                count += 1
                            elif char == ']':
                                count -= 1
                                if count == 0:
                                    json_str = response[start_idx:start_idx+i+1]
                                    break
                    else:
                        # Fallback to the entire response
                        json_str = response
                
                # Clean the string to ensure valid JSON
                json_str = json_str.strip()
                
                # Extract [...] if present anywhere in the text
                bracket_match = re.search(r'\[[\s\S]*\]', json_str)
                if bracket_match:
                    json_str = bracket_match.group(0)
                
                # Parse JSON
                dialogues = json.loads(json_str)
                
                # Add additional fields
                for i, dialogue in enumerate(dialogues):
                    dialogue['start_time'] = i * 60  # Arbitrary timestamp
                    dialogue['end_time'] = (i + 1) * 60  # Arbitrary timestamp
                    dialogue['is_generated'] = True
                    
                    # Add timestamps to individual lines if not present
                    for j, line in enumerate(dialogue['dialogue']):
                        if 'timestamp' not in line:
                            line['timestamp'] = i * 60 + j * 10  # Arbitrary timestamp
                
                return dialogues
            except Exception as e:
                print(f"Error parsing dialogues from LLM response: {str(e)}")
                print(f"Response was: {response}")
                return []
        
        return []
    
    def generate_vocabulary(self, transcript_text: str) -> Dict[str, str]:
        """Generate vocabulary using Nova Micro"""
        # Create prompt for Nova Micro
        prompt = f"""
        Please create a vocabulary list from this Hindi language transcript.
        Extract individual Hindi words and provide their English translations.
        
        Transcript:
        {transcript_text[:3000]}  # Limit text length to avoid token limits
        
        Format your response as:
        ```json
        {{
          "नमस्ते": "Hello",
          "धन्यवाद": "Thank you",
          "आप": "You",
          ...
        }}
        ```
        
        Provide at least 20 unique Hindi words with their accurate English translations.
        Focus on common words that would be useful for a Hindi language learner.
        Ensure the words actually appear in the transcript.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            # Try to extract JSON dictionary
            try:
                # Find JSON between ```json and ```
                json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON dictionary directly
                    json_str = re.search(r'\{\s*"[\u0900-\u097F]+"\s*:', response)
                    if json_str:
                        # Find the matching end bracket
                        start_idx = json_str.start()
                        # Simple approach to find the matching closing bracket
                        count = 0
                        for i, char in enumerate(response[start_idx:]):
                            if char == '{':
                                count += 1
                            elif char == '}':
                                count -= 1
                                if count == 0:
                                    json_str = response[start_idx:start_idx+i+1]
                                    break
                    else:
                        # Fallback to the entire response
                        json_str = response
                
                # Clean the string to ensure valid JSON
                json_str = json_str.strip()
                
                # Extract {...} if present anywhere in the text
                bracket_match = re.search(r'\{[\s\S]*\}', json_str)
                if bracket_match:
                    json_str = bracket_match.group(0)
                
                # Parse JSON
                vocabulary = json.loads(json_str)
                return vocabulary
            except Exception as e:
                print(f"Error parsing vocabulary from LLM response: {str(e)}")
                print(f"Response was: {response}")
                
                # Attempt fallback with regex matching for key-value pairs
                try:
                    vocabulary = {}
                    hindi_translation_pairs = re.findall(r'"([\u0900-\u097F]+)":\s*"([^"]+)"', response)
                    for hindi, english in hindi_translation_pairs:
                        vocabulary[hindi] = english
                    
                    if vocabulary:
                        return vocabulary
                except Exception:
                    return {}
        
        return {}
    
    def process_transcript(self, filename: str) -> Dict[str, Any]:
        """Process a transcript file and use Nova Micro to generate structured data"""
        transcript_text = self.load_transcript(filename)
        if not transcript_text:
            return {"error": f"Failed to load transcript: {filename}"}
        
        # Generate all data using Nova Micro
        language_pairs = self.generate_language_pairs(transcript_text)
        dialogues = self.generate_dialogues(transcript_text)
        vocabulary = self.generate_vocabulary(transcript_text)
        
        return {
            "language_pairs": language_pairs,
            "dialogues": dialogues,
            "vocabulary": vocabulary,
            "stats": {
                "transcript_length": len(transcript_text),
                "language_pairs": len(language_pairs),
                "dialogues": len(dialogues),
                "vocabulary_items": len(vocabulary)
            }
        }
    
    def save_structured_data(self, data: Dict[str, Any], output_file: str) -> bool:
        """Save structured data to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving structured data: {str(e)}")
            return False
    
    def get_dataframe(self, data: Dict[str, Any], data_type: str) -> Optional[pd.DataFrame]:
        """Convert structured data to pandas DataFrame"""
        if data_type == "language_pairs" and "language_pairs" in data:
            return pd.DataFrame(data["language_pairs"])
        elif data_type == "vocabulary" and "vocabulary" in data:
            vocab_df = pd.DataFrame({
                "hindi": list(data["vocabulary"].keys()),
                "english": list(data["vocabulary"].values())
            })
            return vocab_df
        return None


if __name__ == "__main__":
    # Test the processor
    processor = TranscriptProcessor()
    
    # Load and process a transcript file
    test_file = "sample_transcript.txt"
    structured_data = processor.process_transcript(test_file)
    
    if "error" not in structured_data:
        # Save the structured data
        processor.save_structured_data(structured_data, "structured_data.json")
        
        # Display some stats
        print(f"Processed transcript: {test_file}")
        print(f"Stats: {structured_data['stats']}")
        
        # Create a DataFrame of language pairs
        pairs_df = processor.get_dataframe(structured_data, "language_pairs")
        if pairs_df is not None:
            print("\nLanguage Pairs Sample:")
            print(pairs_df.head())
