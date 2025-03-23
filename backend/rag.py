import os
import json
import boto3
import chromadb
from typing import List, Dict, Any, Optional
from chromadb.utils import embedding_functions

class BedrockEmbeddingFunction:
    def __init__(self, bedrock_client):
        """Initialize the Bedrock embedding function."""
        self.bedrock_client = bedrock_client
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Get embeddings from Amazon Bedrock Titan model."""
        embeddings = []
        for text in input:
            try:
                response = self.bedrock_client.invoke_model(
                    modelId="amazon.titan-embed-text-v1",
                    body=json.dumps({
                        "inputText": text
                    })
                )
                embedding = json.loads(response.get('body').read())['embedding']
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding: {str(e)}")
                # Return a zero embedding as fallback
                embeddings.append([0.0] * 1536)  # Titan embeddings are 1536 dimensions
        return embeddings

class HindiRAG:
    def __init__(self, collection_name: str = "hindi-learning-content"):
        """Initialize the RAG system with ChromaDB"""
        self.client = chromadb.Client()
        
        # Use Amazon Bedrock for embeddings if available, otherwise use default
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
            self.embedding_function = BedrockEmbeddingFunction(self.bedrock_client)
        except Exception as e:
            print(f"Could not initialize Bedrock client: {str(e)}")
            # Fallback to Chroma's default embedding function
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
        # Create or get collection
        try:
            # Check if collection exists
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
        except Exception as e:
            print(f"Error creating/getting collection: {str(e)}")
            # Create a temporary collection with a unique name as fallback
            import uuid
            temp_name = f"{collection_name}-{uuid.uuid4().hex[:8]}"
            self.collection = self.client.create_collection(
                name=temp_name,
                embedding_function=self.embedding_function
            )
    
    def add_language_pairs(self, language_pairs: List[Dict]) -> None:
        """Add language pairs to the vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for i, pair in enumerate(language_pairs):
            # Create a combined document with hindi and english
            document = f"Hindi: {pair['hindi']}\nEnglish: {pair['english']}"
            documents.append(document)
            
            # Add metadata
            metadatas.append({
                "hindi": pair['hindi'],
                "english": pair['english'],
                "timestamp": pair.get('timestamp', 0),
                "type": "language_pair"
            })
            
            # Create a unique ID
            ids.append(f"pair_{i}")
        
        # Add to collection in batches to avoid overwhelming the system
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
    
    def add_dialogues(self, dialogues: List[Dict]) -> None:
        """Add dialogues to the vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for i, dialogue in enumerate(dialogues):
            # Create a document from the dialogue
            dialogue_text = "\n".join([line['text'] for line in dialogue['dialogue']])
            documents.append(dialogue_text)
            
            # Add metadata
            metadatas.append({
                "start_time": dialogue.get('start_time', 0),
                "end_time": dialogue.get('end_time', 0),
                "type": "dialogue"
            })
            
            # Create a unique ID
            ids.append(f"dialogue_{i}")
        
        # Add to collection in batches
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
    
    def add_vocabulary(self, vocabulary: Dict[str, str]) -> None:
        """Add vocabulary to the vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for i, (hindi, english) in enumerate(vocabulary.items()):
            # Create a document for the vocabulary item
            document = f"Hindi word: {hindi}\nEnglish meaning: {english}"
            documents.append(document)
            
            # Add metadata
            metadatas.append({
                "hindi": hindi,
                "english": english,
                "type": "vocabulary"
            })
            
            # Create a unique ID
            ids.append(f"vocab_{i}")
        
        # Add to collection in batches
        batch_size = 200
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
    
    def add_structured_data(self, data: Dict[str, Any]) -> None:
        """Add all types of structured data to the vector database"""
        if "language_pairs" in data:
            self.add_language_pairs(data["language_pairs"])
        
        if "dialogues" in data:
            self.add_dialogues(data["dialogues"])
        
        if "vocabulary" in data:
            self.add_vocabulary(data["vocabulary"])
    
    def load_from_json(self, json_file: str) -> bool:
        """Load structured data from a JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.add_structured_data(data)
            return True
        except Exception as e:
            print(f"Error loading data from JSON: {str(e)}")
            return False
    
    def query(self, query_text: str, n_results: int = 3, filter_type: Optional[str] = None) -> Dict[str, Any]:
        """Query the vector database for relevant context"""
        try:
            # Apply filter if specified
            where_filter = None
            if filter_type:
                where_filter = {"type": filter_type}
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
            
            # Format results
            formatted_results = {
                "documents": results.get("documents", [[]]),
                "metadatas": results.get("metadatas", [[]]),
                "distances": results.get("distances", [[]]),
                "ids": results.get("ids", [[]])
            }
            
            return formatted_results
        except Exception as e:
            print(f"Error querying the database: {str(e)}")
            return {"error": str(e)}
    
    def get_context_for_bedrock(self, query: str, n_results: int = 3) -> str:
        """Get formatted context for Bedrock prompt"""
        results = self.query(query, n_results=n_results)
        
        if "error" in results:
            return ""
        
        context = "CONTEXT INFORMATION:\n\n"
        
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            doc_type = metadata.get("type", "unknown")
            
            context += f"--- Document {i+1} ({doc_type}) ---\n"
            context += doc + "\n\n"
        
        return context


if __name__ == "__main__":
    # Test the RAG system
    rag = HindiRAG()
    
    # Test adding data
    test_data = {
        "language_pairs": [
            {"hindi": "नमस्ते", "english": "Hello", "timestamp": 0},
            {"hindi": "आप कैसे हैं?", "english": "How are you?", "timestamp": 5}
        ],
        "dialogues": [
            {
                "dialogue": [
                    {"text": "नमस्ते", "is_hindi": True, "timestamp": 0},
                    {"text": "Hello", "is_hindi": False, "timestamp": 2},
                    {"text": "आप कैसे हैं?", "is_hindi": True, "timestamp": 5},
                    {"text": "I am fine, thank you.", "is_hindi": False, "timestamp": 7}
                ],
                "start_time": 0,
                "end_time": 7
            }
        ],
        "vocabulary": {
            "नमस्ते": "Hello",
            "आप": "You",
            "कैसे": "How",
            "हैं": "Are"
        }
    }
    
    rag.add_structured_data(test_data)
    
    # Test querying
    query_result = rag.query("How to say hello in Hindi?", n_results=2)
    print("\nQuery Results:")
    if "documents" in query_result and query_result["documents"]:
        for i, doc in enumerate(query_result["documents"][0]):
            print(f"\nDocument {i+1}:")
            print(doc)
    
    # Test getting context for Bedrock
    context = rag.get_context_for_bedrock("How to greet someone in Hindi?")
    print("\nContext for Bedrock:")
    print(context)