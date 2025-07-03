import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager

import redis
import pymongo
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
import openai
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models for our Service Desk Bot
@dataclass
class DocumentChunk:
    """Represents a chunk of processed document content"""
    id: str
    content: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class ChatMessage:
    """Individual message in a conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None

@dataclass
class ChatSession:
    """Complete chat session with history"""
    session_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_activity: datetime
    context_summary: Optional[str] = None

class ServiceDeskRAG:
    """
    Core RAG pipeline for service desk operations with memory optimization,
    caching, and session management capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize embedding model - using a lightweight but effective model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector database (FAISS for efficiency)
        self.vector_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.document_store = {}  # Maps vector indices to document chunks
        
        # Initialize caching layer (Redis for fast retrieval)
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Initialize session storage (MongoDB for flexible document storage)
        self.mongo_client = pymongo.MongoClient(config.get('mongo_url', 'mongodb://localhost:27017/'))
        self.chat_db = self.mongo_client.service_desk_bot.chats
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Optimal size for our embedding model
            chunk_overlap=50,  # Maintain context between chunks
            separators=["\n\n", "\n", ".", " "]  # Semantic boundaries
        )
        
        # LLM client (OpenAI GPT)
        openai.api_key = config.get('openai_api_key')
        
        # Cache configuration for memory efficiency
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self.max_memory_chunks = config.get('max_memory_chunks', 10000)
        
        logger.info("ServiceDeskRAG initialized successfully")

    async def process_documents(self, file_paths: List[str]) -> None:
        """
        Process and index PDF and text documents into our RAG system.
        This method handles the entire document ingestion pipeline.
        """
        logger.info(f"Processing {len(file_paths)} documents")
        
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # Load document based on file type
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.lower().endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                # Load and split documents
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    chunk_id = self._generate_chunk_id(file_path, i)
                    
                    document_chunk = DocumentChunk(
                        id=chunk_id,
                        content=chunk.page_content,
                        source_file=Path(file_path).name,
                        chunk_index=i,
                        metadata={
                            'file_path': file_path,
                            'page_number': chunk.metadata.get('page', 0),
                            'processed_at': datetime.now().isoformat()
                        }
                    )
                    
                    all_chunks.append(document_chunk)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Generate embeddings in batches for efficiency
        await self._generate_and_store_embeddings(all_chunks)
        
        logger.info(f"Successfully processed {len(all_chunks)} chunks")

    async def _generate_and_store_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """
        Generate embeddings for document chunks and store them efficiently.
        Uses batch processing to optimize memory usage.
        """
        batch_size = 32  # Process embeddings in batches to manage memory
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract text content for embedding
            texts = [chunk.content for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Store in vector index and document store
            start_idx = self.vector_index.ntotal
            self.vector_index.add(embeddings)
            
            for j, chunk in enumerate(batch):
                chunk.embedding = embeddings[j]
                self.document_store[start_idx + j] = chunk
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """Generate unique ID for document chunks"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 5,
        session_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve most relevant document chunks for a given query.
        Implements caching and session-aware retrieval.
        """
        
        # Check cache first for efficiency
        cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            logger.info("Retrieved results from cache")
            chunk_indices = json.loads(cached_result)
            return [self.document_store[idx] for idx in chunk_indices if idx in self.document_store]
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search vector index
        scores, indices = self.vector_index.search(query_embedding, top_k)
        
        # Retrieve corresponding chunks
        relevant_chunks = []
        for idx, score in zip(indices[0], scores[0]):
            if idx in self.document_store and score > 0.3:  # Relevance threshold
                chunk = self.document_store[idx]
                relevant_chunks.append(chunk)
        
        # Cache results for future queries
        chunk_indices = [self.vector_index.ntotal - len(self.document_store) + 
                        list(self.document_store.keys()).index(chunk.id) 
                        for chunk in relevant_chunks if chunk.id in [c.id for c in self.document_store.values()]]
        
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(indices[0].tolist()))
        
        return relevant_chunks

    async def generate_response(
        self, 
        query: str, 
        session_id: str,
        user_id: str
    ) -> Tuple[str, List[str]]:
        """
        Generate response using RAG pipeline with session context.
        Returns both the response and source citations.
        """
        
        # Retrieve chat history for context
        session = await self.get_chat_session(session_id, user_id)
        
        # Get relevant chunks
        relevant_chunks = await self.retrieve_relevant_chunks(query, session_id=session_id)
        
        if not relevant_chunks:
            return "I apologize, but I couldn't find relevant information to answer your question. Could you please rephrase or provide more details?", []
        
        # Prepare context from retrieved chunks
        context = self._prepare_context(relevant_chunks, session)
        
        # Generate response using LLM
        response = await self._generate_llm_response(query, context, session)
        
        # Extract source citations
        sources = [chunk.source_file for chunk in relevant_chunks]
        
        # Update chat session
        await self._update_chat_session(session_id, user_id, query, response)
        
        return response, list(set(sources))  # Remove duplicate sources

    def _prepare_context(self, chunks: List[DocumentChunk], session: ChatSession) -> str:
        """
        Prepare context string from retrieved chunks and chat history.
        Implements smart context management to stay within token limits.
        """
        
        # Start with retrieved document context
        context_parts = ["RELEVANT DOCUMENTATION:"]
        
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}: {chunk.source_file}]")
            context_parts.append(chunk.content)
            context_parts.append("")  # Empty line for readability
        
        # Add recent chat history for continuity
        if session and session.messages:
            context_parts.append("RECENT CONVERSATION:")
            # Include last 3 exchanges to maintain context without overwhelming the prompt
            recent_messages = session.messages[-6:]  # Last 3 user-assistant pairs
            
            for msg in recent_messages:
                context_parts.append(f"{msg.role.upper()}: {msg.content}")
        
        return "\n".join(context_parts)

    async def _generate_llm_response(
        self, 
        query: str, 
        context: str, 
        session: ChatSession
    ) -> str:
        """
        Generate response using the language model with proper prompting.
        """
        
        system_prompt = """You are a helpful service desk assistant. Your role is to:

1. Provide accurate, helpful responses based on the documentation provided
2. Acknowledge when information is not available in the provided context
3. Maintain a professional but friendly tone
4. Reference specific sources when possible
5. Ask clarifying questions when the user's request is ambiguous

Always base your responses on the provided documentation context. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing and suggest how the user might get help."""

        user_prompt = f"""Context:
{context}

User Question: {query}

Please provide a helpful response based on the context above."""

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent, factual responses
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment or contact support for immediate assistance."

    async def get_chat_session(self, session_id: str, user_id: str) -> ChatSession:
        """
        Retrieve or create chat session with efficient caching.
        """
        
        # Check Redis cache first
        cache_key = f"session:{session_id}"
        cached_session = self.redis_client.get(cache_key)
        
        if cached_session:
            session_data = json.loads(cached_session)
            messages = [
                ChatMessage(
                    role=msg['role'],
                    content=msg['content'],
                    timestamp=datetime.fromisoformat(msg['timestamp']),
                    metadata=msg.get('metadata')
                ) for msg in session_data['messages']
            ]
            
            return ChatSession(
                session_id=session_data['session_id'],
                user_id=session_data['user_id'],
                messages=messages,
                created_at=datetime.fromisoformat(session_data['created_at']),
                last_activity=datetime.fromisoformat(session_data['last_activity']),
                context_summary=session_data.get('context_summary')
            )
        
        # Check MongoDB for persistent storage
        session_doc = self.chat_db.find_one({'session_id': session_id, 'user_id': user_id})
        
        if session_doc:
            messages = [
                ChatMessage(
                    role=msg['role'],
                    content=msg['content'],
                    timestamp=datetime.fromisoformat(msg['timestamp']),
                    metadata=msg.get('metadata')
                ) for msg in session_doc['messages']
            ]
            
            session = ChatSession(
                session_id=session_doc['session_id'],
                user_id=session_doc['user_id'],
                messages=messages,
                created_at=datetime.fromisoformat(session_doc['created_at']),
                last_activity=datetime.fromisoformat(session_doc['last_activity']),
                context_summary=session_doc.get('context_summary')
            )
        else:
            # Create new session
            session = ChatSession(
                session_id=session_id,
                user_id=user_id,
                messages=[],
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
        
        # Cache in Redis for fast access
        session_dict = asdict(session)
        session_dict['messages'] = [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'metadata': msg.metadata
            } for msg in session.messages
        ]
        session_dict['created_at'] = session.created_at.isoformat()
        session_dict['last_activity'] = session.last_activity.isoformat()
        
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(session_dict))
        
        return session

    async def _update_chat_session(
        self, 
        session_id: str, 
        user_id: str, 
        user_message: str, 
        assistant_response: str
    ) -> None:
        """
        Update chat session with new messages and manage memory efficiently.
        """
        
        session = await self.get_chat_session(session_id, user_id)
        
        # Add new messages
        now = datetime.now()
        session.messages.extend([
            ChatMessage(role='user', content=user_message, timestamp=now),
            ChatMessage(role='assistant', content=assistant_response, timestamp=now)
        ])
        
        # Implement message history trimming to manage memory
        max_messages = 20  # Keep last 10 exchanges
        if len(session.messages) > max_messages:
            # Summarize older messages if needed
            if not session.context_summary:
                session.context_summary = await self._summarize_conversation(session.messages[:-max_messages])
            
            # Keep only recent messages
            session.messages = session.messages[-max_messages:]
        
        session.last_activity = now
        
        # Update both cache and persistent storage
        await self._save_session(session)

    async def _summarize_conversation(self, messages: List[ChatMessage]) -> str:
        """
        Create a summary of older conversation messages to maintain context
        while reducing memory usage.
        """
        
        conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "Summarize the following conversation, focusing on key issues discussed and any solutions provided. Keep it concise but comprehensive."
                    },
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {str(e)}")
            return "Previous conversation covered various service desk topics."

    async def _save_session(self, session: ChatSession) -> None:
        """
        Save session to both cache and persistent storage.
        """
        
        # Prepare session document for MongoDB
        session_doc = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'metadata': msg.metadata
                } for msg in session.messages
            ],
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'context_summary': session.context_summary
        }
        
        # Update MongoDB
        self.chat_db.replace_one(
            {'session_id': session.session_id, 'user_id': session.user_id},
            session_doc,
            upsert=True
        )
        
        # Update Redis cache
        cache_key = f"session:{session.session_id}"
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(session_doc))

    async def cleanup_old_sessions(self, days_old: int = 30) -> None:
        """
        Background task to clean up old chat sessions and manage storage.
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Remove old sessions from MongoDB
        result = self.chat_db.delete_many({
            'last_activity': {'$lt': cutoff_date.isoformat()}
        })
        
        logger.info(f"Cleaned up {result.deleted_count} old chat sessions")

# FastAPI Application
app = FastAPI(title="Service Desk Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'mongo_url': 'mongodb://localhost:27017/',
        'openai_api_key': 'your-openai-api-key-here',
        'cache_ttl': 3600,
        'max_memory_chunks': 10000
    }
    
    rag_system = ServiceDeskRAG(config)
    
    # Process initial documents (you would specify your document paths here)
    # await rag_system.process_documents(['path/to/your/docs.pdf', 'path/to/manual.txt'])

class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for the service desk bot.
    """
    try:
        response, sources = await rag_system.generate_response(
            query=request.message,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        return ChatResponse(
            response=response,
            sources=sources,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload-documents")
async def upload_documents(file_paths: List[str], background_tasks: BackgroundTasks):
    """
    Endpoint to upload and process new documents.
    """
    background_tasks.add_task(rag_system.process_documents, file_paths)
    return {"message": f"Processing {len(file_paths)} documents in background"}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """
    Delete a specific chat session.
    """
    # Remove from cache
    rag_system.redis_client.delete(f"session:{session_id}")
    
    # Remove from MongoDB
    result = rag_system.chat_db.delete_one({
        'session_id': session_id, 
        'user_id': user_id
    })
    
    if result.deleted_count > 0:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)