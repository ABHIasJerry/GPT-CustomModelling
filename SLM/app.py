# ================================================================
# 📂 Project   : GPT Custom Modelling
# 📜 File      : app.py
# 🧑‍💻 Author    : Abhinaba
# 🕒 Created   : 2026-05-23
# 🔄 Revision  : v1.0.0
# ✨ Purpose   : App for chatting with a Small Language Model
# ================================================================
# 📝 Change Log:
#   v1.0.0 | 2026-05-23 | Initial version
# ================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import List, Optional

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="SLM Chat API",
    description="API for chatting with a Small Language Model",
    version="1.0.0"
)

# ============================================================================
# LOAD MODEL GLOBALLY (ON STARTUP)
# ============================================================================

MODEL_DIR = "./my_slm"
model = None
tokenizer = None
device = None


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer when the server starts."""
    global model, tokenizer, device
    
    try:
        print("Loading model and tokenizer...")
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Vocab size: {len(tokenizer)}")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise


# ============================================================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_sequences: Optional[int] = 1


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    prompt: str
    generated_texts: List[str]
    device: str
    num_tokens_generated: int


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    model_name: str
    vocab_size: int
    device: str
    max_sequence_length: int
    status: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SLM Chat API",
        "endpoints": {
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)",
            "/model-info": "Get model information",
            "/chat": "Chat with the model (POST)"
        }
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="GPT-2 Small Language Model",
        vocab_size=len(tokenizer),
        device=device,
        max_sequence_length=512,
        status="ready"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the model.
    
    Parameters:
    - prompt: Input text to generate from
    - max_length: Maximum length of generated text (default: 100)
    - temperature: Controls randomness (0.0-2.0, default: 0.7)
    - top_p: Nucleus sampling parameter (default: 0.9)
    - num_sequences: Number of text sequences to generate (default: 1)
    
    Returns:
    - generated_texts: List of generated text sequences
    - num_tokens_generated: Total tokens generated
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate inputs
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.max_length < 10 or request.max_length > 512:
            raise HTTPException(status_code=400, detail="max_length must be between 10 and 512")
        
        if request.temperature < 0 or request.temperature > 2:
            raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
        
        if request.top_p < 0 or request.top_p > 1:
            raise HTTPException(status_code=400, detail="top_p must be between 0 and 1")
        
        if request.num_sequences < 1 or request.num_sequences > 5:
            raise HTTPException(status_code=400, detail="num_sequences must be between 1 and 5")
        
        # Tokenize input
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                num_return_sequences=request.num_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        # Calculate total tokens generated
        num_tokens = output_ids.shape[1] - input_ids.shape[1]
        
        return ChatResponse(
            prompt=request.prompt,
            generated_texts=generated_texts,
            device=device,
            num_tokens_generated=num_tokens
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


@app.post("/chat-simple", tags=["Chat"])
async def chat_simple(prompt: str):
    """
    Simplified chat endpoint (single string prompt).
    
    Parameters:
    - prompt: Input text (query parameter)
    
    Returns:
    - generated_text: Generated text
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting SLM Chat API Server")
    print("=" * 60)
    print("\nAPI Documentation:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("\nEndpoints:")
    print("  - GET  / : Root endpoint")
    print("  - GET  /health : Health check")
    print("  - GET  /model-info : Model information")
    print("  - POST /chat : Advanced chat (JSON body)")
    print("  - POST /chat-simple : Simple chat (query parameter)")
    print("\n" + "=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
