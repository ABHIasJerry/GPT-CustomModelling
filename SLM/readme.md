Quick Start [simple api]
bash# 1. Install dependencies
pip install fastapi uvicorn

# 2. Run the server
python app.py

# 3. Open Postman and make requests to http://localhost:8000/chat
📊 API Endpoints
MethodEndpointPurposeGET/Welcome messageGET/healthHealth checkGET/model-infoModel informationPOST/chatAdvanced chat (JSON body)POST/chat-simpleSimple chat (query param)
📮 Postman Setup
Method 1: Simple Chat
POST http://localhost:8000/chat-simple?prompt=Hello
Method 2: Advanced Chat (Recommended)
POST http://localhost:8000/chat

Body (JSON):
{
  "prompt": "<parse your prompt>",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "num_sequences": 1
}
🎯 Key Features
✅ Auto-documentation - Swagger UI at http://localhost:8000/docs
✅ GPU support - Automatically uses CUDA if available
✅ Input validation - Checks all parameters
✅ Error handling - Detailed error messages
✅ Multiple outputs - Generate 1-5 text variations
✅ Temperature control - Control randomness of generation

---------------------------------------------------------------------------------------

Quick Start [with Langchain & Token logger]
bash# Install dependencies
pip install fastapi uvicorn transformers torch langchain

# Run the server
python app_with_logging.py

📮 Key Endpoints
MethodEndpointPurposePOST/chatChat and auto-log tokensGET/chat/stats/{chat_id}Get token statisticsGET/chat/history/{chat_id}View all messagesGET/chatsList all chat sessionsGET/logs/statsOverall usage statsGET/logs/exportExport to JSONDELETE/chat/{chat_id}Delete a chat

📝 Postman Example
Start a Chat:
jsonPOST http://localhost:8000/chat

Body:
{
  "prompt": "Machine learning",
  "max_length": 100,
  "temperature": 0.7
}

Response:
{
  "chat_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "Machine learning",
  "generated_texts": [...],
  "prompt_tokens": 3,
  "completion_tokens": 95,
  "total_tokens": 98,
  "timestamp": "2024-01-15T10:30:05"
}
Get Chat Stats:
jsonGET http://localhost:8000/chat/stats/550e8400-e29b-41d4-a716-446655440000

Response:
{
  "chat_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00",
  "total_prompt_tokens": 45,
  "total_completion_tokens": 285,
  "total_tokens": 330,
  "messages_count": 3
}
View Message History:
jsonGET http://localhost:8000/chat/history/550e8400-e29b-41d4-a716-446655440000

Shows all messages with individual token counts

📊 Database Schema
Table: chats
chat_id → created_at → total_prompt_tokens → total_completion_tokens → total_tokens → messages_count
Table: messages
id → chat_id → timestamp → prompt → generated_text → prompt_tokens → completion_tokens → total_tokens → temperature → max_length → top_p

🔍 Analytics Examples
Overall Statistics:
bashGET /logs/stats
List All Chats:
bashGET /chats
Export All Logs:
bashGET /logs/export?output_file=logs_backup.json

✨ Key Advantages
✅ LangChain Compatible - Uses standard callback patterns
✅ Persistent Storage - SQLite database
✅ Automatic Logging - No manual tracking needed
✅ Complete History - View past conversations anytime
✅ Analytics Ready - Export for analysis
✅ Easy Integration - Just use the endpoints!