# ================================================================
# 📂 Project   : GPT Custom Modelling
# 📜 File      : app_with_logging.py
# 🧑‍💻 Author    : Abhinaba
# 🕒 Created   : 2026-05-23
# 🔄 Revision  : v1.0.0
# ✨ Purpose   : App with logging for Small Language model
# ================================================================
# 📝 Change Log:
#   v1.0.0 | 2026-05-23 | Initial version
# ================================================================

import torch
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from langchain.callbacks import BaseCallbackHandler
from langchain.schema import LLMResult
import sqlite3

# ============================================================================
# CUSTOM LANGCHAIN CALLBACK FOR LOGGING
# ============================================================================

class TokenUsageLogger(BaseCallbackHandler):
    """Custom LangChain callback for logging token usage and chat info."""
    
    def __init__(self, db_path: str = "chat_logs.db"):
        self.db_path = db_path
        self.chat_id = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for logging."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                total_prompt_tokens INTEGER DEFAULT 0,
                total_completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                messages_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                timestamp TIMESTAMP,
                prompt TEXT,
                generated_text TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                temperature REAL,
                max_length INTEGER,
                top_p REAL,
                FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def set_chat_id(self, chat_id: str):
        """Set the current chat ID."""
        self.chat_id = chat_id
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if chat already exists
        cursor.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO chats (chat_id, created_at)
                VALUES (?, ?)
            """, (chat_id, datetime.now()))
            conn.commit()
        
        conn.close()
    
    def log_message(
        self,
        chat_id: str,
        prompt: str,
        generated_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        temperature: float,
        max_length: int,
        top_p: float
    ):
        """Log a chat message with token counts."""
        total_tokens = prompt_tokens + completion_tokens
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute("""
            INSERT INTO messages
            (chat_id, timestamp, prompt, generated_text, prompt_tokens,
             completion_tokens, total_tokens, temperature, max_length, top_p)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chat_id,
            datetime.now(),
            prompt,
            generated_text,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            temperature,
            max_length,
            top_p
        ))
        
        # Update chat totals
        cursor.execute("""
            UPDATE chats
            SET total_prompt_tokens = total_prompt_tokens + ?,
                total_completion_tokens = total_completion_tokens + ?,
                total_tokens = total_tokens + ?,
                messages_count = messages_count + 1
            WHERE chat_id = ?
        """, (prompt_tokens, completion_tokens, total_tokens, chat_id))
        
        conn.commit()
        conn.close()
    
    def get_chat_stats(self, chat_id: str) -> Dict[str, Any]:
        """Get statistics for a chat session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chats WHERE chat_id = ?
        """, (chat_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return {
            "chat_id": result[0],
            "created_at": result[1],
            "total_prompt_tokens": result[2],
            "total_completion_tokens": result[3],
            "total_tokens": result[4],
            "messages_count": result[5]
        }
    
    def get_all_chats(self) -> List[Dict[str, Any]]:
        """Get all chat sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "chat_id": row[0],
                "created_at": row[1],
                "total_prompt_tokens": row[2],
                "total_completion_tokens": row[3],
                "total_tokens": row[4],
                "messages_count": row[5]
            }
            for row in results
        ]
    
    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get message history for a chat session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, prompt, generated_text, prompt_tokens,
                   completion_tokens, total_tokens, temperature, max_length, top_p
            FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp
        """, (chat_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "prompt": row[2],
                "generated_text": row[3],
                "prompt_tokens": row[4],
                "completion_tokens": row[5],
                "total_tokens": row[6],
                "temperature": row[7],
                "max_length": row[8],
                "top_p": row[9]
            }
            for row in results
        ]
    
    def export_chat_logs(self, output_file: str = "chat_logs.json"):
        """Export all chat logs to JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all chats
        cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
        chats = cursor.fetchall()
        
        # Get all messages
        cursor.execute("""
            SELECT * FROM messages ORDER BY timestamp
        """)
        messages = cursor.fetchall()
        conn.close()
        
        # Organize data
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_chats": len(chats),
            "chats": [
                {
                    "chat_id": chat[0],
                    "created_at": chat[1],
                    "total_prompt_tokens": chat[2],
                    "total_completion_tokens": chat[3],
                    "total_tokens": chat[4],
                    "messages_count": chat[5]
                }
                for chat in chats
            ],
            "messages": [
                {
                    "id": msg[0],
                    "chat_id": msg[1],
                    "timestamp": msg[2],
                    "prompt": msg[3],
                    "generated_text": msg[4],
                    "prompt_tokens": msg[5],
                    "completion_tokens": msg[6],
                    "total_tokens": msg[7],
                    "temperature": msg[8],
                    "max_length": msg[9],
                    "top_p": msg[10]
                }
                for msg in messages
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_file


# ============================================================================
# INITIALIZE FASTAPI APP AND LOGGER
# ============================================================================

app = FastAPI(
    title="SLM Chat API with Logging",
    description="API for chatting with a Small Language Model (with token tracking)",
    version="2.0.0"
)

# Initialize token logger
token_logger = TokenUsageLogger(db_path="chat_logs.db")

# Load model globally
MODEL_DIR = "./my_slm"
model = None
tokenizer = None
device = None


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup."""
    global model, tokenizer, device
    
    try:
        print("Loading model and tokenizer...")
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded on {device}")
        print(f"✓ Vocab size: {len(tokenizer)}")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    prompt: str
    chat_id: Optional[str] = None
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_sequences: Optional[int] = 1


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    chat_id: str
    prompt: str
    generated_texts: List[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    device: str
    timestamp: str


class ChatStatsResponse(BaseModel):
    """Response model for chat statistics."""
    chat_id: str
    created_at: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    messages_count: int


class ChatHistoryItem(BaseModel):
    """Model for individual chat history item."""
    id: int
    timestamp: str
    prompt: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    temperature: float
    max_length: int
    top_p: float


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SLM Chat API with Token Logging",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/chat": "Chat with the model (POST)",
            "/chat/stats/{chat_id}": "Get chat statistics",
            "/chat/history/{chat_id}": "Get chat message history",
            "/chats": "List all chat sessions",
            "/logs/export": "Export all logs to JSON"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the model and log token usage.
    
    Parameters:
    - prompt: Input text
    - chat_id: Optional chat session ID (auto-generated if not provided)
    - max_length: Maximum tokens to generate (10-512)
    - temperature: Randomness control (0.0-2.0)
    - top_p: Nucleus sampling (0-1)
    - num_sequences: Number of outputs (1-5)
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if request.max_length < 10 or request.max_length > 512:
        raise HTTPException(status_code=400, detail="max_length must be between 10 and 512")
    
    # Generate or use provided chat_id
    chat_id = request.chat_id or str(uuid.uuid4())
    token_logger.set_chat_id(chat_id)
    
    try:
        # Tokenize input
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
        prompt_tokens = input_ids.shape[1]
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Calculate token counts
        completion_tokens = output_ids.shape[1] - prompt_tokens
        total_tokens = prompt_tokens + completion_tokens
        
        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Log the message
        token_logger.log_message(
            chat_id=chat_id,
            prompt=request.prompt,
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            temperature=request.temperature,
            max_length=request.max_length,
            top_p=request.top_p
        )
        
        return ChatResponse(
            chat_id=chat_id,
            prompt=request.prompt,
            generated_texts=[generated_text],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            device=device,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/chat/stats/{chat_id}", response_model=ChatStatsResponse, tags=["Logging"])
async def get_chat_stats(chat_id: str):
    """Get statistics for a specific chat session."""
    stats = token_logger.get_chat_stats(chat_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    
    return ChatStatsResponse(**stats)


@app.get("/chat/history/{chat_id}", tags=["Logging"])
async def get_chat_history(chat_id: str):
    """Get message history for a specific chat session."""
    # Verify chat exists
    if not token_logger.get_chat_stats(chat_id):
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    
    history = token_logger.get_chat_history(chat_id)
    
    return {
        "chat_id": chat_id,
        "message_count": len(history),
        "messages": history
    }


@app.get("/chats", tags=["Logging"])
async def list_all_chats():
    """List all chat sessions with summary statistics."""
    chats = token_logger.get_all_chats()
    
    total_stats = {
        "total_chats": len(chats),
        "total_prompt_tokens": sum(chat["total_prompt_tokens"] for chat in chats),
        "total_completion_tokens": sum(chat["total_completion_tokens"] for chat in chats),
        "total_tokens": sum(chat["total_tokens"] for chat in chats),
        "total_messages": sum(chat["messages_count"] for chat in chats)
    }
    
    return {
        "summary": total_stats,
        "chats": chats
    }


@app.get("/logs/export", tags=["Logging"])
async def export_logs(output_file: str = "chat_logs.json"):
    """Export all chat logs to JSON file."""
    file_path = token_logger.export_chat_logs(output_file)
    
    return {
        "message": "Logs exported successfully",
        "file": file_path,
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/chat/{chat_id}", tags=["Logging"])
async def delete_chat(chat_id: str):
    """Delete a chat session and its logs."""
    conn = sqlite3.connect("chat_logs.db")
    cursor = conn.cursor()
    
    # Check if chat exists
    cursor.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    
    # Delete messages
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    
    # Delete chat
    cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    
    conn.commit()
    conn.close()
    
    return {
        "message": f"Chat {chat_id} deleted successfully",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/logs/stats", tags=["Logging"])
async def get_overall_stats():
    """Get overall statistics across all chats."""
    chats = token_logger.get_all_chats()
    
    if not chats:
        return {
            "total_chats": 0,
            "total_messages": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0
        }
    
    return {
        "total_chats": len(chats),
        "total_messages": sum(chat["messages_count"] for chat in chats),
        "total_prompt_tokens": sum(chat["total_prompt_tokens"] for chat in chats),
        "total_completion_tokens": sum(chat["total_completion_tokens"] for chat in chats),
        "total_tokens": sum(chat["total_tokens"] for chat in chats),
        "average_tokens_per_message": sum(chat["total_tokens"] for chat in chats) / sum(chat["messages_count"] for chat in chats) if sum(chat["messages_count"] for chat in chats) > 0 else 0
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting SLM Chat API with Token Logging")
    print("=" * 70)
    print("\nAPI Documentation:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("\nKey Endpoints:")
    print("  - POST   /chat : Chat and log tokens")
    print("  - GET    /chat/stats/{chat_id} : Get chat statistics")
    print("  - GET    /chat/history/{chat_id} : Get message history")
    print("  - GET    /chats : List all chats")
    print("  - GET    /logs/stats : Get overall statistics")
    print("  - GET    /logs/export : Export logs to JSON")
    print("  - DELETE /chat/{chat_id} : Delete a chat")
    print("\nDatabase: chat_logs.db")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
