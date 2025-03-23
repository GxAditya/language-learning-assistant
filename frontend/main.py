import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from typing import Dict
import json
from collections import Counter
import re

# Now import backend modules
from backend.get_transcript import YouTubeTranscriptDownloader
from backend.chat import BedrockChat


# Page config
st.set_page_config(
    page_title="Hindi Learning Assistant",
    page_icon="🇮🇳",
    layout="wide"
)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def render_header():
    """Render the header section"""
    st.title("🇮🇳 Hindi Learning Assistant")
    st.markdown("""
    Transform YouTube transcripts into interactive Hindi learning experiences.
    
    This tool demonstrates:
    - Base LLM Capabilities
    - RAG (Retrieval Augmented Generation)
    - Amazon Bedrock Integration
    - Agent-based Learning Systems
    """)

def render_sidebar():
    """Render the sidebar with component selection"""
    with st.sidebar:
        st.header("Development Stages")
        
        # Main component selection
        selected_stage = st.radio(
            "Select Stage:",
            [
                "1. Chat with Nova",
                "2. Raw Transcript",
                "3. Structured Data",
                "4. RAG Implementation",
                "5. Interactive Learning"
            ]
        )
        
        # Stage descriptions
        stage_info = {
            "1. Chat with Nova": """
            **Current Focus:**
            - Basic Hindi learning
            - Understanding LLM capabilities
            - Identifying limitations
            """,
            
            "2. Raw Transcript": """
            **Current Focus:**
            - YouTube transcript download
            - Raw text visualization
            - Initial data examination
            """,
            
            "3. Structured Data": """
            **Current Focus:**
            - Text cleaning
            - Dialogue extraction
            - Data structuring
            """,
            
            "4. RAG Implementation": """
            **Current Focus:**
            - Bedrock embeddings
            - Vector storage
            - Context retrieval
            """,
            
            "5. Interactive Learning": """
            **Current Focus:**
            - Scenario generation
            - Audio synthesis
            - Interactive practice
            """
        }
        
        st.markdown("---")
        st.markdown(stage_info[selected_stage])
        
        return selected_stage

def render_chat_stage():
    """Render an improved chat interface"""
    st.header("Chat with Nova")

    # Initialize BedrockChat instance if not in session state
    if 'bedrock_chat' not in st.session_state:
        st.session_state.bedrock_chat = BedrockChat()

    # Introduction text
    st.markdown("""
    Start by exploring Nova's base Hindi language capabilities. Try asking questions about Hindi grammar, 
    vocabulary, or cultural aspects.
    """)

    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input area
    if prompt := st.chat_input("Ask about Hindi language..."):
        # Process the user input
        process_message(prompt)

    # Example questions in sidebar
    with st.sidebar:
        st.markdown("### Try These Examples")
        example_questions = [
            "How do I say 'Where is the train station?' in Hindi?",
            "Explain the difference between में and मैं",
            "What's the polite form of खाना?",
            "How do I count objects in Hindi?",
            "What's the difference between नमस्ते and नमस्कार?",
            "How do I ask for directions politely?"
        ]
        
        for q in example_questions:
            if st.button(q, use_container_width=True, type="secondary"):
                # Process the example question
                process_message(q)
                st.rerun()

    # Add a clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.rerun()

def process_message(message: str):
    """Process a message and generate a response"""
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(message)

    # Generate and display assistant's response
    with st.chat_message("assistant", avatar="🤖"):
        response = st.session_state.bedrock_chat.generate_response(message)
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})



def count_characters(text):
    """Count Hindi and total characters in text"""
    if not text:
        return 0, 0
        
    def is_hindi(char):
        return '\u0900' <= char <= '\u097F'  # Devanagari Unicode range
    
    hindi_chars = sum(1 for char in text if is_hindi(char))
    return hindi_chars, len(text)

def render_transcript_stage():
    """Render the raw transcript stage"""
    st.header("Raw Transcript Processing")
    
    # URL input
    url = st.text_input(
        "YouTube URL",
        placeholder="Enter a Hindi lesson YouTube URL"
    )
    
    # Download button and processing
    if url:
        if st.button("Download Transcript"):
            try:
                downloader = YouTubeTranscriptDownloader()
                transcript = downloader.get_transcript(url)
                if transcript:
                    # Store the raw transcript text in session state
                    transcript_text = "\n".join([entry['text'] for entry in transcript])
                    st.session_state.transcript = transcript_text
                    st.success("Transcript downloaded successfully!")
                else:
                    st.error("No transcript found for this video.")
            except Exception as e:
                st.error(f"Error downloading transcript: {str(e)}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Transcript")
        if st.session_state.transcript:
            st.text_area(
                label="Raw text",
                value=st.session_state.transcript,
                height=400,
                disabled=True
            )
    
        else:
            st.info("No transcript loaded yet")
    
    with col2:
        st.subheader("Transcript Stats")
        if st.session_state.transcript:
            # Calculate stats
            jp_chars, total_chars = count_characters(st.session_state.transcript)
            total_lines = len(st.session_state.transcript.split('\n'))
            
            # Display stats
            st.metric("Total Characters", total_chars)
            st.metric("Hindi Characters", jp_chars)
            st.metric("Total Lines", total_lines)
        else:
            st.info("Load a transcript to see statistics")

def render_structured_stage():
    """Render the structured data stage"""
    st.header("Structured Data Processing")
    
    # Import the TranscriptProcessor
    from backend.structured_data import TranscriptProcessor
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dialogue Extraction")
        
        # Check if transcript exists in session state
        if not st.session_state.transcript:
            st.warning("Please download a transcript first in the Raw Transcript stage")
            return
        
        # Save transcript to a temporary file for processing
        temp_file = "temp_transcript.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(st.session_state.transcript)
        
        # Initialize processor
        processor = TranscriptProcessor()
        
        # Process button
        if st.button("Process Transcript"):
            with st.spinner("Processing transcript..."):
                # Process the transcript
                structured_data = processor.process_transcript(temp_file)
                
                # Store in session state
                st.session_state.structured_data = structured_data
                
                # Success message
                if "error" not in structured_data:
                    st.success("Transcript processed successfully!")
                else:
                    st.error(f"Error processing transcript: {structured_data['error']}")
        
    with col2:
        st.subheader("Data Structure")
        
        # Display structured data if available
        if "structured_data" in st.session_state:
            data = st.session_state.structured_data
            
            # Statistics
            st.metric("Language Pairs", data["stats"]["language_pairs"])
            st.metric("Dialogues", data["stats"]["dialogues"])
            st.metric("Vocabulary Items", data["stats"]["vocabulary_items"])
            
            # Display sample of language pairs
            if data["language_pairs"]:
                st.subheader("Sample Language Pairs")
                pairs_df = processor.get_dataframe(data, "language_pairs")
                if pairs_df is not None and not pairs_df.empty:
                    st.dataframe(pairs_df.head(5))
            
            # Display sample of vocabulary
            if data["vocabulary"]:
                st.subheader("Sample Vocabulary")
                vocab_df = processor.get_dataframe(data, "vocabulary")
                if vocab_df is not None and not vocab_df.empty:
                    st.dataframe(vocab_df.head(5))
        else:
            st.info("Process a transcript to see structured data")

def render_rag_stage():
    """Render the RAG implementation stage"""
    st.header("RAG System")
    
    # Import the RAG system
    from backend.rag import HindiRAG
    
    # Initialize RAG system if not in session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = HindiRAG()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Knowledge Base")
        
        # Add structured data to RAG if available
        if "structured_data" in st.session_state:
            if st.button("Add Structured Data to RAG"):
                with st.spinner("Adding data to knowledge base..."):
                    st.session_state.rag_system.add_structured_data(st.session_state.structured_data)
                    st.success("Data added to knowledge base!")
                    
                    # Store flag for data loaded
                    st.session_state.rag_data_loaded = True
        else:
            st.warning("Process structured data first")
    
    with col2:
        st.subheader("Test Query")
        
        # Query input
        query = st.text_input(
            "Enter a Hindi learning question",
            placeholder="How do I say 'hello' in Hindi?"
        )
        
        # Process query
        if query and st.button("Search"):
            # Check if data is loaded
            if not st.session_state.get('rag_data_loaded', False):
                st.warning("Please add structured data to the knowledge base first")
                return
                
            with st.spinner("Searching..."):
                # Get results
                results = st.session_state.rag_system.query(query, n_results=3)
                
                # Store in session state
                st.session_state.rag_results = results
                st.session_state.rag_query = query
    
    # Results section
    if "rag_results" in st.session_state:
        st.subheader("Retrieved Context")
        
        results = st.session_state.rag_results
        
        if "error" in results:
            st.error(f"Error during retrieval: {results['error']}")
        elif results["documents"] and results["documents"][0]:
            # Display each retrieved document
            for i, doc in enumerate(results["documents"][0]):
                with st.expander(f"Document {i+1}"):
                    st.text(doc)
                    
                    # Show metadata if available
                    if results["metadatas"] and results["metadatas"][0] and i < len(results["metadatas"][0]):
                        metadata = results["metadatas"][0][i]
                        st.json(metadata)
            
            # Generate response with context
            if st.button("Generate Response with Context"):
                # Import the chat module
                from backend.chat import BedrockChat
                
                # Get formatted context
                context = st.session_state.rag_system.get_context_for_bedrock(st.session_state.rag_query)
                
                # Create chat instance
                chat = BedrockChat()
                
                # Generate response with context
                with st.spinner("Generating response..."):
                    prompt = f"""
                    {context}
                    
                    Using the context information provided above, please answer the following question
                    about Hindi language learning:
                    
                    Question: {st.session_state.rag_query}
                    """
                    
                    response = chat.generate_response(prompt)
                    
                    # Display response
                    st.subheader("Generated Response")
                    st.markdown(response)
        else:
            st.info("No results found. Try a different query.")

def render_interactive_stage():
    """Render the interactive learning stage"""
    st.header("Interactive Learning")
    
    # Import the interactive learning system
    from backend.interactive import InteractiveLearning
    
    # Initialize interactive learning if not in session state
    if 'interactive_learning' not in st.session_state:
        st.session_state.interactive_learning = InteractiveLearning()
    
    # Practice type selection
    practice_type = st.selectbox(
        "Select Practice Type",
        ["Dialogue Practice", "Vocabulary Quiz", "Listening Exercise"]
    )
    
    # Difficulty selection
    difficulty = st.select_slider(
        "Select Difficulty",
        options=["beginner", "intermediate", "advanced"],
        value="beginner"
    )
    
    # Generate button
    if st.button("Generate Exercise"):
        with st.spinner("Generating exercise..."):
            if practice_type == "Dialogue Practice":
                exercise = st.session_state.interactive_learning.generate_dialogue_practice(difficulty=difficulty)
                st.session_state.current_exercise = exercise
                st.session_state.exercise_type = "dialogue"
            
            elif practice_type == "Vocabulary Quiz":
                exercise = st.session_state.interactive_learning.generate_vocabulary_quiz(
                    num_questions=4, 
                    difficulty=difficulty
                )
                st.session_state.current_exercise = exercise
                st.session_state.exercise_type = "vocabulary"
            
            elif practice_type == "Listening Exercise":
                exercise = st.session_state.interactive_learning.generate_listening_exercise(difficulty=difficulty)
                st.session_state.current_exercise = exercise
                st.session_state.exercise_type = "listening"
    
    # Display current exercise if available
    if "current_exercise" in st.session_state:
        exercise = st.session_state.current_exercise
        exercise_type = st.session_state.exercise_type
        
        # Common exercise header
        st.subheader(exercise.get("title", practice_type))
        st.markdown(exercise.get("instructions", ""))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Render different exercise types
            if exercise_type == "dialogue":
                st.subheader("Practice Scenario")
                st.markdown(exercise.get("scenario", ""))
                
                # Display dialogue
                st.text_area("Dialogue", exercise.get("dialogue", ""), height=200)
                
                # Display cues for practice
                if "cues" in exercise:
                    st.subheader("Practice Cues")
                    for i, cue in enumerate(exercise["cues"]):
                        if cue.get("is_hindi", False):
                            st.text(f"Fill in: {cue.get('cue', '')}")
                            # Reveal button for each cue
                            if st.button(f"Reveal Line {i+1}"):
                                st.success(cue.get("original", ""))
                        else:
                            st.text(cue.get("original", ""))
            
            elif exercise_type == "vocabulary":
                st.subheader("Vocabulary Quiz")
                
                # Display questions
                if "questions" in exercise:
                    for i, question in enumerate(exercise["questions"]):
                        st.markdown(f"### {i+1}. {question.get('word', '')}")
                        
                        # Radio button for options
                        option_key = f"vocab_q{i}"
                        selected_option = st.radio(
                            f"Select translation for '{question.get('word', '')}'",
                            question.get("options", []),
                            key=option_key
                        )
                        
                        # Check answer button
                        check_key = f"check_vocab_q{i}"
                        if st.button("Check Answer", key=check_key):
                            if selected_option == question.get("correct_answer", ""):
                                st.success("Correct! Great job!")
                            else:
                                st.error(f"Not quite. The correct answer is: {question.get('correct_answer', '')}")
                        
                        st.markdown("---")
            
            elif exercise_type == "listening":
                st.subheader("Listening Exercise")
                
                # Display exercises
                if "exercises" in exercise:
                    for i, ex in enumerate(exercise["exercises"]):
                        st.markdown(f"### Exercise {i+1}")
                        
                        # Display audio text (in a real app, this would be TTS audio)
                        st.text_area(f"Listen to this phrase", ex.get("audio_text", ""), key=f"audio_text_{i}")
                        
                        # Display question
                        st.markdown(ex.get("question", ""))
                        
                        # Radio button for options
                        listen_key = f"listen_q{i}"
                        selected_option = st.radio(
                            "Select your answer",
                            ex.get("options", []),
                            key=listen_key
                        )
                        
                        # Check answer button
                        check_key = f"check_listen_q{i}"
                        if st.button("Check Answer", key=check_key):
                            if selected_option == ex.get("correct_answer", ""):
                                st.success("Correct! Great job!")
                            else:
                                st.error(f"Not quite. The correct answer is: {ex.get('correct_answer', '')}")
                        
                        st.markdown("---")
        
        with col2:
            st.subheader("Help")
            
            # Hint button
            if exercise_type == "dialogue" or exercise_type == "vocabulary":
                if st.button("Get a Hint"):
                    # Get a hindi word to hint about
                    hint_text = ""
                    if exercise_type == "dialogue" and "dialogue" in exercise:
                        # Find first Hindi line in dialogue
                        lines = exercise["dialogue"].split("\n")
                        for line in lines:
                            if any('\u0900' <= c <= '\u097F' for c in line):
                                hint_text = line
                                break
                    elif exercise_type == "vocabulary" and "questions" in exercise and exercise["questions"]:
                        # Use first question word
                        hint_text = exercise["questions"][0].get("word", "")
                    
                    if hint_text:
                        hint = st.session_state.interactive_learning.get_hint(hint_text)
                        st.info(f"Hint: {hint}")
            
            # Progress tracking placeholder
            st.subheader("Your Progress")
            st.markdown("Progress tracking will be implemented here")

def main():
    render_header()
    selected_stage = render_sidebar()
    
    # Render appropriate stage
    if selected_stage == "1. Chat with Nova":
        render_chat_stage()
    elif selected_stage == "2. Raw Transcript":
        render_transcript_stage()
    elif selected_stage == "3. Structured Data":
        render_structured_stage()
    elif selected_stage == "4. RAG Implementation":
        render_rag_stage()
    elif selected_stage == "5. Interactive Learning":
        render_interactive_stage()
    
    # Debug section at the bottom
    with st.expander("Debug Information"):
        st.json({
            "selected_stage": selected_stage,
            "transcript_loaded": st.session_state.transcript is not None,
            "chat_messages": len(st.session_state.messages)
        })

if __name__ == "__main__":
    main()