from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uvicorn
from agentic_rag_bot import AgenticRAG, load_llm, MemoryManager
from profile_manager import ProfileManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = load_llm()
memory_manager = MemoryManager()
profile_manager = ProfileManager(llm)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SessionRequest(BaseModel):
    session_id: str

def ensure_session_files(session_id: str):
    full_path = os.path.join("./memory", f"session_{session_id}_full.json")
    clean_path = os.path.join("./memory", f"session_{session_id}_clean.json")
    if not os.path.exists(full_path):
        with open(full_path, 'w') as f: json.dump([], f)
    if not os.path.exists(clean_path):
        with open(clean_path, 'w') as f: json.dump([], f)

@app.post("/create-session")
async def create_session(req: SessionRequest):
    ensure_session_files(req.session_id)
    return {"status": "Session initialized", "session_id": req.session_id}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Deletes the corresponding session files from the memory folder."""
    full_path = os.path.join("./memory", f"session_{session_id}_full.json")
    clean_path = os.path.join("./memory", f"session_{session_id}_clean.json")
    
    try:
        if os.path.exists(full_path): os.remove(full_path)
        if os.path.exists(clean_path): os.remove(clean_path)
        return {"status": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    full, _ = memory_manager.load_history(session_id)
    return {"full_history": full}

@app.post("/chat")
async def chat(req: ChatRequest):
    ensure_session_files(req.session_id)
    agent = AgenticRAG(llm, session_id=req.session_id)
    
    # Trigger summarization if needed
    if agent.summarize_history():
        memory_manager.save_history(req.session_id, agent.interface_history, agent.clean_history)
    
    profile = memory_manager.load_profile()
    related, hyp_passage = agent.analyze_intent_and_generate_hp(req.message, profile)
    
    agent.interface_history.append({"role": "user", "content": req.message})
    
    if not related:
        declaration = "I am a domain-specific tool specializing in entrepreneurship legal knowledge."
        agent.interface_history.append({"role": "bot", "content": declaration})
        memory_manager.save_history(req.session_id, agent.interface_history, agent.clean_history)
        return {"answer": declaration, "related": False}

    agent.clean_history.append({"role": "user", "content": req.message})
    chunks = agent.run_retrieval_loop(hyp_passage, req.message)
    final_answer = agent.generate_final_response(chunks, profile)
    
    agent.interface_history.append({"role": "bot", "content": final_answer})
    agent.clean_history.append({"role": "bot", "content": final_answer})
    memory_manager.save_history(req.session_id, agent.interface_history, agent.clean_history)
    
    return {"answer": final_answer, "related": True}

@app.post("/sync-profile")
async def sync_profile(req: SessionRequest):
    _, clean_history = memory_manager.load_history(req.session_id)
    current_profile = memory_manager.load_profile() or {}
    updated_profile = profile_manager.update_profile(clean_history, current_profile)
    memory_manager.save_profile(updated_profile)
    return {"status": "Profile updated successfully", "profile": updated_profile}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
