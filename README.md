# Hindi Language Learning Assistant

An intelligent, progressive learning tool powered by RAG (Retrieval Augmented Generation) that helps users learn Hindi through interactive exercises and contextual guidance.

## Project Overview

**Difficulty Level:** Intermediate (Level 200) 
> *Requires understanding of RAG implementation and AWS service integration*

### Business Goal

This application demonstrates how RAG and intelligent agents enhance language learning by grounding responses in authentic Hindi lesson content. The system showcases the evolution from basic LLM responses to a fully contextual learning assistant, providing both practical language learning value and technical insights into RAG implementation.

### Key Features

- **Progressive Learning Stages**: From basic chat to fully interactive exercises
- **YouTube Transcript Processing**: Converts Hindi lesson videos into learning materials
- **Structured Content Extraction**: Identifies dialogues, vocabulary, and language patterns
- **Vector-Based Retrieval**: Contextual responses based on relevant language examples
- **Interactive Practice**: Dialogue scenarios, vocabulary quizzes, and listening exercises

## Technical Implementation

### Core Technologies

* **Amazon Bedrock**:
   * API integration (converse, guardrails, embeddings, agents) ([AWS Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html))
   * Amazon Nova Micro for text generation ([AWS Nova Documentation](https://aws.amazon.com/ai/generative-ai/nova))
   * Titan for embeddings
* **Frontend**: Streamlit, pandas (data visualization)
* **Vector Storage**: ChromaDB with SQLite backend
* **Content Source**: YouTube transcripts via [YouTubeTranscriptApi](https://pypi.org/project/youtube-transcript-api/)

### System Architecture

The application demonstrates clear progression through five key stages:

1. **Base LLM**: Direct interaction with Amazon Nova Micro
2. **Raw Transcript**: YouTube transcript processing and analysis
3. **Structured Data**: Extraction of language pairs, dialogues, and vocabulary
4. **RAG Implementation**: Vector storage and contextual retrieval
5. **Interactive Features**: Generated learning exercises based on content

## Technical Challenges

1. Processing and structuring bilingual (Hindi/English) content for RAG
2. Optimal chunking and embedding of Hindi language content
3. Demonstrating progression from base LLM to RAG to students
4. Maintaining context accuracy when retrieving Hindi language examples
5. Balancing between direct answers and providing learning guidance
6. Structuring effective multiple-choice questions from retrieved content

## Installation and Usage

1. Clone the repository
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Set up AWS credentials for Bedrock access
4. Run the application: `streamlit run frontend/main.py`

## Development Guidelines

* Maintain clear separation between components for teaching purposes
* Include proper error handling for Hindi text processing
* Provide clear visualization of the RAG process
* Design within AWS free tier limits where possible

## Resources

* [ChromaDB Documentation](https://github.com/chroma-core/chroma)