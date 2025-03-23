import random
import json
from typing import List, Dict, Any, Optional, Tuple
import boto3
import re

from backend.rag import HindiRAG
from backend.chat import BedrockChat

class InteractiveLearning:
    def __init__(self, structured_data_path: str = None, rag_system=None):
        """Initialize the interactive learning system"""
        self.structured_data = self._load_data(structured_data_path) if structured_data_path else None
        self.rag_system = rag_system
        self.bedrock_chat = BedrockChat()
        
        # Initialize Bedrock client for potential TTS
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
        except Exception as e:
            print(f"Could not initialize Bedrock client: {str(e)}")
            self.bedrock_client = None
    
    def _load_data(self, path: str) -> Dict[str, Any]:
        """Load structured data from a file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading structured data: {str(e)}")
            return {
                "language_pairs": [],
                "dialogues": [],
                "vocabulary": {}
            }
    
    def generate_dialogue_practice(self, difficulty: str = "beginner") -> Dict[str, Any]:
        """Generate a dialogue practice scenario based on difficulty level"""
        # Define difficulty keywords for filtering appropriate dialogues
        difficulty_keywords = {
            "beginner": ["basic", "simple", "greeting", "introduction", "नमस्ते", "धन्यवाद", "hello", "thank you"],
            "intermediate": ["restaurant", "shopping", "directions", "travel", "खाना", "यात्रा", "food", "journey"],
            "advanced": ["business", "politics", "philosophy", "culture", "व्यापार", "संस्कृति", "business", "culture"]
        }
        
        scenario = {
            "title": f"Hindi Dialogue Practice ({difficulty.capitalize()})",
            "difficulty": difficulty,
            "scenario": "",
            "dialogue": [],
            "cues": [],
            "translations": {}
        }
        
        # Try to find an appropriate dialogue from structured data
        if self.structured_data and "dialogues" in self.structured_data and self.structured_data["dialogues"]:
            # Filter dialogues that might match the difficulty level
            matching_dialogues = []
            for dialogue in self.structured_data["dialogues"]:
                dialogue_text = " ".join([line["text"] for line in dialogue["dialogue"]])
                if any(keyword in dialogue_text.lower() for keyword in difficulty_keywords.get(difficulty, [])):
                    matching_dialogues.append(dialogue)
            
            if matching_dialogues:
                # Use a random matching dialogue
                selected = random.choice(matching_dialogues)
                
                # Extract Hindi lines for cues
                hindi_lines = [line for line in selected["dialogue"] if line["is_hindi"]]
                english_lines = [line for line in selected["dialogue"] if not line["is_hindi"]]
                
                # Create cues from English text
                for i, line in enumerate(hindi_lines[:4]):  # Limit to 4 cues
                    # Find corresponding English line
                    for e_line in english_lines:
                        if abs(e_line.get("timestamp", 0) - line.get("timestamp", 0)) < 5:
                            scenario["cues"].append(e_line["text"])
                            scenario["translations"][e_line["text"]] = line["text"]
                            break
                
                # Set dialogue and scenario
                scenario["dialogue"] = [{"text": line["text"], "is_hindi": line["is_hindi"]} 
                                      for line in selected["dialogue"]]
                scenario["scenario"] = f"Practice this {difficulty} level Hindi dialogue"
                
                return scenario
        
        # If no appropriate dialogue found or no structured data, generate with Bedrock
        return self._generate_dialogue_with_llm(difficulty, difficulty_keywords[difficulty])
    
    def _generate_dialogue_with_llm(self, difficulty: str, keywords: List[str]) -> Dict[str, Any]:
        """Generate a dialogue using Amazon Bedrock Nova Micro"""
        prompt = f"""
        Create a natural Hindi dialogue practice scenario for {difficulty} level learners.
        Include 4-6 conversation exchanges between two people.
        The scenario should incorporate these themes: {', '.join(keywords)}
        
        Format your response as JSON:
        {{
          "title": "Hindi Dialogue Practice ({difficulty.capitalize()})",
          "difficulty": "{difficulty}",
          "scenario": "A brief description of the scenario",
          "dialogue": [
            {{"text": "Hindi line 1", "is_hindi": true}},
            {{"text": "English translation 1", "is_hindi": false}},
            {{"text": "Hindi line 2", "is_hindi": true}},
            {{"text": "English translation 2", "is_hindi": false}}
          ],
          "cues": ["English cue 1", "English cue 2", "English cue 3", "English cue 4"],
          "translations": {{"English cue 1": "Hindi translation 1", "English cue 2": "Hindi translation 2"}}
        }}
        
        Make sure the scenario is appropriate for {difficulty} level Hindi learners.
        For beginner: use simple greetings and basic phrases.
        For intermediate: use everyday situations like shopping or dining.
        For advanced: use complex topics like business or cultural discussions.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Find the first { and last }
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                    else:
                        json_str = response
                
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing dialogue from LLM: {str(e)}")
                # Fallback to a basic structure
                return {
                    "title": f"Hindi Dialogue Practice ({difficulty.capitalize()})",
                    "difficulty": difficulty,
                    "scenario": "Practice basic Hindi conversation",
                    "dialogue": [
                        {"text": "नमस्ते, आप कैसे हैं?", "is_hindi": True},
                        {"text": "Hello, how are you?", "is_hindi": False},
                        {"text": "मैं ठीक हूँ, धन्यवाद।", "is_hindi": True},
                        {"text": "I am fine, thank you.", "is_hindi": False}
                    ],
                    "cues": ["Hello, how are you?", "I am fine, thank you."],
                    "translations": {
                        "Hello, how are you?": "नमस्ते, आप कैसे हैं?",
                        "I am fine, thank you.": "मैं ठीक हूँ, धन्यवाद।"
                    }
                }
        
        # Fallback if LLM fails
        return {
            "title": f"Hindi Dialogue Practice ({difficulty.capitalize()})",
            "difficulty": difficulty,
            "scenario": "Practice basic Hindi conversation",
            "dialogue": [
                {"text": "नमस्ते, आप कैसे हैं?", "is_hindi": True},
                {"text": "Hello, how are you?", "is_hindi": False},
                {"text": "मैं ठीक हूँ, धन्यवाद।", "is_hindi": True},
                {"text": "I am fine, thank you.", "is_hindi": False}
            ],
            "cues": ["Hello, how are you?", "I am fine, thank you."],
            "translations": {
                "Hello, how are you?": "नमस्ते, आप कैसे हैं?",
                "I am fine, thank you.": "मैं ठीक हूँ, धन्यवाद।"
            }
        }
    
    def generate_vocabulary_quiz(self, num_questions: int = 5) -> Dict[str, Any]:
        """Generate a vocabulary quiz with multiple-choice questions"""
        quiz = {
            "title": "Hindi Vocabulary Quiz",
            "questions": [],
            "answers": {}
        }
        
        # Try to use vocabulary from structured data
        vocab_items = []
        if self.structured_data and "vocabulary" in self.structured_data:
            vocab = self.structured_data["vocabulary"]
            vocab_items = [(hindi, english) for hindi, english in vocab.items()]
        
        # If we don't have enough vocabulary, generate questions with LLM
        if len(vocab_items) < num_questions * 2:  # Need extra for distractors
            return self._generate_vocab_quiz_with_llm(num_questions)
        
        # Shuffle vocabulary items
        random.shuffle(vocab_items)
        
        # Create quiz questions
        for i in range(min(num_questions, len(vocab_items))):
            hindi, correct_english = vocab_items[i]
            
            # Get distractors (other English translations)
            distractors = [eng for hnd, eng in vocab_items[num_questions:] if eng != correct_english][:3]
            
            # If we don't have enough distractors, add some generic ones
            while len(distractors) < 3:
                generic = ["greeting", "thank you", "goodbye", "yes", "no", "please", "friend", 
                          "family", "food", "water", "help", "good", "bad", "big", "small"]
                distractor = random.choice(generic)
                if distractor not in distractors and distractor != correct_english:
                    distractors.append(distractor)
            
            # Create options and shuffle them
            options = [correct_english] + distractors
            random.shuffle(options)
            
            # Create question
            question = {
                "question": f"What is the meaning of '{hindi}'?",
                "options": options
            }
            
            quiz["questions"].append(question)
            quiz["answers"][question["question"]] = correct_english
        
        return quiz
    
    def _generate_vocab_quiz_with_llm(self, num_questions: int) -> Dict[str, Any]:
        """Generate a vocabulary quiz using Amazon Bedrock Nova Micro"""
        prompt = f"""
        Create a Hindi vocabulary quiz with {num_questions} multiple-choice questions.
        Each question should ask for the English translation of a Hindi word.
        
        Format your response as JSON:
        {{
          "title": "Hindi Vocabulary Quiz",
          "questions": [
            {{
              "question": "What is the meaning of 'नमस्ते'?",
              "options": ["Hello", "Goodbye", "Thank you", "Please"]
            }},
            ... (more questions)
          ],
          "answers": {{
            "What is the meaning of 'नमस्ते'?": "Hello",
            ... (more answers)
          }}
        }}
        
        Make sure each question has exactly 4 options with 1 correct answer.
        Include a variety of Hindi words covering different topics.
        Ensure the Hindi words used are common and useful for language learners.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Find the first { and last }
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                    else:
                        json_str = response
                
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing quiz from LLM: {str(e)}")
                # Fallback to a basic quiz
                return self._create_basic_quiz()
        
        # Fallback if LLM fails
        return self._create_basic_quiz()
    
    def _create_basic_quiz(self) -> Dict[str, Any]:
        """Create a basic vocabulary quiz as fallback"""
        basic_quiz = {
            "title": "Hindi Vocabulary Quiz",
            "questions": [
                {
                    "question": "What is the meaning of 'नमस्ते'?",
                    "options": ["Hello", "Goodbye", "Thank you", "Please"]
                },
                {
                    "question": "What is the meaning of 'धन्यवाद'?",
                    "options": ["Thank you", "Please", "Welcome", "Sorry"]
                },
                {
                    "question": "What is the meaning of 'हाँ'?",
                    "options": ["Yes", "No", "Maybe", "Hello"]
                },
                {
                    "question": "What is the meaning of 'पानी'?",
                    "options": ["Water", "Food", "Bread", "Tea"]
                },
                {
                    "question": "What is the meaning of 'अच्छा'?",
                    "options": ["Good", "Bad", "Big", "Small"]
                }
            ],
            "answers": {
                "What is the meaning of 'नमस्ते'?": "Hello",
                "What is the meaning of 'धन्यवाद'?": "Thank you",
                "What is the meaning of 'हाँ'?": "Yes",
                "What is the meaning of 'पानी'?": "Water",
                "What is the meaning of 'अच्छा'?": "Good"
            }
        }
        return basic_quiz
    
    def generate_listening_exercise(self, difficulty: str = "beginner") -> Dict[str, Any]:
        """Generate a listening exercise with comprehension questions"""
        exercise = {
            "title": f"Hindi Listening Exercise ({difficulty.capitalize()})",
            "difficulty": difficulty,
            "audio_text": "",
            "translation": "",
            "questions": [],
            "answers": {}
        }
        
        # Try to use language pairs from structured data
        if self.structured_data and "language_pairs" in self.structured_data:
            pairs = self.structured_data["language_pairs"]
            # Filter longer phrases suitable for listening exercises
            suitable_pairs = [pair for pair in pairs if len(pair["hindi"].split()) > 3]
            
            if suitable_pairs:
                # Use a random suitable pair
                selected = random.choice(suitable_pairs)
                exercise["audio_text"] = selected["hindi"]
                exercise["translation"] = selected["english"]
                
                # Generate comprehension questions
                questions = self._get_translations_with_llm(selected["hindi"], selected["english"], difficulty)
                if questions:
                    exercise.update(questions)
                    return exercise
        
        # If no suitable pairs found or questions generation failed, generate with Bedrock
        return self._generate_listening_exercise_with_llm(difficulty)
    
    def _get_translations_with_llm(self, hindi_text: str, english_text: str, difficulty: str) -> Dict[str, Any]:
        """Generate comprehension questions based on a Hindi text using LLM"""
        prompt = f"""
        Create comprehension questions for a Hindi listening exercise.
        
        Hindi text: {hindi_text}
        English translation: {english_text}
        Difficulty level: {difficulty}
        
        Create 3 multiple-choice questions that test understanding of this text.
        For beginner level, focus on basic vocabulary and simple information.
        For intermediate level, include more nuanced understanding.
        For advanced level, include cultural context and subtle meanings.
        
        Format your response as JSON:
        {{
          "questions": [
            {{
              "question": "What is being discussed in this audio?",
              "options": ["Option A", "Option B", "Option C", "Option D"]
            }},
            ... (more questions)
          ],
          "answers": {{
            "What is being discussed in this audio?": "Correct option here",
            ... (more answers)
          }}
        }}
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Find the first { and last }
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                    else:
                        json_str = response
                
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing questions from LLM: {str(e)}")
                return None
        
        return None
    
    def _generate_listening_exercise_with_llm(self, difficulty: str) -> Dict[str, Any]:
        """Generate a complete listening exercise using Amazon Bedrock Nova Micro"""
        prompt = f"""
        Create a Hindi listening exercise for {difficulty} level learners.
        Include a Hindi passage, its English translation, and comprehension questions.
        
        Format your response as JSON:
        {{
          "title": "Hindi Listening Exercise ({difficulty.capitalize()})",
          "difficulty": "{difficulty}",
          "audio_text": "Hindi passage here (5-8 sentences)",
          "translation": "English translation of the Hindi passage",
          "questions": [
            {{
              "question": "What is being discussed in this audio?",
              "options": ["Option A", "Option B", "Option C", "Option D"]
            }},
            ... (2 more questions)
          ],
          "answers": {{
            "What is being discussed in this audio?": "Correct option here",
            ... (answers for other questions)
          }}
        }}
        
        For beginner level: use simple vocabulary and basic sentence structures.
        For intermediate level: use everyday topics with more complex grammar.
        For advanced level: use nuanced vocabulary and cultural references.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Find the first { and last }
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                    else:
                        json_str = response
                
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing listening exercise from LLM: {str(e)}")
                # Fallback to a basic exercise
                return self._create_basic_listening_exercise(difficulty)
        
        # Fallback if LLM fails
        return self._create_basic_listening_exercise(difficulty)
    
    def _create_basic_listening_exercise(self, difficulty: str) -> Dict[str, Any]:
        """Create a basic listening exercise as fallback"""
        basic_exercises = {
            "beginner": {
                "title": "Hindi Listening Exercise (Beginner)",
                "difficulty": "beginner",
                "audio_text": "नमस्ते! मेरा नाम राहुल है। मैं दिल्ली से हूँ। मुझे हिंदी सीखना पसंद है।",
                "translation": "Hello! My name is Rahul. I am from Delhi. I like learning Hindi.",
                "questions": [
                    {
                        "question": "What is the speaker's name?",
                        "options": ["Rahul", "Amit", "Priya", "Neha"]
                    },
                    {
                        "question": "Where is the speaker from?",
                        "options": ["Delhi", "Mumbai", "Kolkata", "Chennai"]
                    },
                    {
                        "question": "What does the speaker like?",
                        "options": ["Learning Hindi", "Eating food", "Playing cricket", "Reading books"]
                    }
                ],
                "answers": {
                    "What is the speaker's name?": "Rahul",
                    "Where is the speaker from?": "Delhi",
                    "What does the speaker like?": "Learning Hindi"
                }
            },
            "intermediate": {
                "title": "Hindi Listening Exercise (Intermediate)",
                "difficulty": "intermediate",
                "audio_text": "मैं पिछले पांच साल से भारत में रह रहा हूँ। मुझे यहाँ का खाना बहुत पसंद है, विशेष रूप से दक्षिण भारतीय व्यंजन। मैंने थोड़ी हिंदी सीखी है, लेकिन अभी भी सीख रहा हूँ।",
                "translation": "I have been living in India for the past five years. I really like the food here, especially South Indian cuisine. I have learned some Hindi, but I am still learning.",
                "questions": [
                    {
                        "question": "How long has the speaker been living in India?",
                        "options": ["5 years", "2 years", "10 years", "6 months"]
                    },
                    {
                        "question": "What kind of food does the speaker prefer?",
                        "options": ["South Indian cuisine", "North Indian cuisine", "Chinese food", "Italian food"]
                    },
                    {
                        "question": "What is the speaker's current status with Hindi?",
                        "options": ["Still learning", "Fluent", "Just started", "Not interested"]
                    }
                ],
                "answers": {
                    "How long has the speaker been living in India?": "5 years",
                    "What kind of food does the speaker prefer?": "South Indian cuisine",
                    "What is the speaker's current status with Hindi?": "Still learning"
                }
            },
            "advanced": {
                "title": "Hindi Listening Exercise (Advanced)",
                "difficulty": "advanced",
                "audio_text": "आधुनिक भारत में भाषाई विविधता एक महत्वपूर्ण मुद्दा है। हालांकि हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है, देश में 22 आधिकारिक भाषाएँ हैं। यह विविधता भारतीय संस्कृति की समृद्धि का प्रतीक है, लेकिन कभी-कभी संचार में चुनौतियां भी पैदा करती है।",
                "translation": "Linguistic diversity is an important issue in modern India. Although Hindi is the most widely spoken language in India, there are 22 official languages in the country. This diversity is a symbol of the richness of Indian culture, but sometimes also creates challenges in communication.",
                "questions": [
                    {
                        "question": "What is described as an important issue in modern India?",
                        "options": ["Linguistic diversity", "Economic growth", "Political stability", "Environmental concerns"]
                    },
                    {
                        "question": "How many official languages are there in India according to the text?",
                        "options": ["22", "10", "15", "30"]
                    },
                    {
                        "question": "According to the text, what challenge can linguistic diversity create?",
                        "options": ["Communication difficulties", "Economic inequality", "Political conflicts", "Cultural homogeneity"]
                    }
                ],
                "answers": {
                    "What is described as an important issue in modern India?": "Linguistic diversity",
                    "How many official languages are there in India according to the text?": "22",
                    "According to the text, what challenge can linguistic diversity create?": "Communication difficulties"
                }
            }
        }
        
        return basic_exercises.get(difficulty, basic_exercises["beginner"])
    
    def check_answer(self, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """Check user answer and provide feedback"""
        is_correct = user_answer == correct_answer
        
        if is_correct:
            feedback = "Correct! Great job!"
        else:
            feedback = f"Not quite. The correct answer is: {correct_answer}"
        
        return is_correct, feedback
    
    def get_hint(self, hindi_text: str) -> str:
        """Generate a hint for learning a Hindi phrase"""
        prompt = f"""
        Provide a helpful hint for learning this Hindi phrase: "{hindi_text}"
        Focus on pronunciation tips, word meanings, or cultural context.
        Keep it brief and beginner-friendly.
        """
        
        response = self.bedrock_chat.generate_response(prompt)
        if response:
            return response
        else:
            return "Focus on pronouncing each syllable clearly and listen for the tone."


if __name__ == "__main__":
    # Test the interactive learning system
    interactive = InteractiveLearning("structured_data.json")
    
    # Generate a dialogue practice
    dialogue_practice = interactive.generate_dialogue_practice("beginner")
    print("\nDIALOGUE PRACTICE:")
    print(f"Title: {dialogue_practice['title']}")
    print(f"Scenario: {dialogue_practice['scenario']}")
    print("Dialogue:")
    for line in dialogue_practice['dialogue'][:4]:  # Show first 4 lines
        print(f"- {line['text']}")
    print("Cues:")
    for cue in dialogue_practice['cues'][:2]:  # Show first 2 cues
        print(f"- {cue}")
    
    # Generate a vocabulary quiz
    vocab_quiz = interactive.generate_vocabulary_quiz(3)  # 3 questions
    print("\nVOCABULARY QUIZ:")
    print(f"Title: {vocab_quiz['title']}")
    for i, q in enumerate(vocab_quiz['questions'][:2]):  # Show first 2 questions
        print(f"Q{i+1}: {q['question']}")
        print(f"Options: {', '.join(q['options'])}")
    
    # Generate a listening exercise
    listening_exercise = interactive.generate_listening_exercise("beginner")
    print("\nLISTENING EXERCISE:")
    print(f"Title: {listening_exercise['title']}")
    print(f"Hindi: {listening_exercise['audio_text'][:50]}...")  # Show first 50 chars
    print(f"English: {listening_exercise['translation'][:50]}...")  # Show first 50 chars
    print("First question:")
    if listening_exercise['questions']:
        q = listening_exercise['questions'][0]
        print(f"- {q['question']}")
        print(f"  Options: {', '.join(q['options'])}")
    
    # Hint
    print("\nHINT: These are common Hindi phrases to know:")
    print("नमस्ते (Namaste) - Hello")
    print("धन्यवाद (Dhanyavaad) - Thank you")
    print("कृपया (Kripaya) - Please")
