import json
import os
import torch
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import CrossEncoder

from tools import LegalRetrievalTools

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Using Ollama for better KV cache and resource management
OLLAMA_MODEL = "gemma4:31b-cloud"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MEMORY_DIR = "./memory"
USER_PROFILE_PATH = os.path.join(MEMORY_DIR, "user_profile.json")
TEMPLATE_PATH = "profile_template.json"

os.makedirs(MEMORY_DIR, exist_ok=True)

def load_llm():
    print(f"Initializing Ollama connection to {OLLAMA_MODEL}...")
    # Ollama handles the KV cache and VRAM management on the server side
    return Ollama(model=OLLAMA_MODEL, temperature=0.1)

# ==========================================
# MEMORY MANAGEMENT
# ==========================================
class MemoryManager:
    @staticmethod
    def load_profile() -> Optional[Dict[str, Any]]:
        if os.path.exists(USER_PROFILE_PATH):
            with open(USER_PROFILE_PATH, 'r') as f: return json.load(f)
        return None

    @staticmethod
    def save_profile(profile: Dict[str, Any]):
        with open(USER_PROFILE_PATH, 'w') as f: json.dump(profile, f, indent=4)

    @staticmethod
    def save_history(session_id: str, interface_history: List[Dict], clean_history: List[Dict]):
        with open(os.path.join(MEMORY_DIR, f"session_{session_id}_full.json"), 'w') as f:
            json.dump(interface_history, f, indent=4)
        with open(os.path.join(MEMORY_DIR, f"session_{session_id}_clean.json"), 'w') as f:
            json.dump(clean_history, f, indent=4)

    @staticmethod
    def load_history(session_id: str) -> Tuple[List[Dict], List[Dict]]:
        full_path = os.path.join(MEMORY_DIR, f"session_{session_id}_full.json")
        clean_path = os.path.join(MEMORY_DIR, f"session_{session_id}_clean.json")
        full = []
        if os.path.exists(full_path):
            with open(full_path, 'r') as f: full = json.load(f)
        clean = []
        if os.path.exists(clean_path):
            with open(clean_path, 'r') as f: clean = json.load(f)
        return full, clean

# ==========================================
# PURE RETRIEVAL AGENT
# ==========================================
class AgenticRAG:
    def __init__(self, llm, session_id="default"):
        self.llm = llm
        self.tools = LegalRetrievalTools()
        self.session_id = session_id
        self.memory = MemoryManager()
        self.interface_history, self.clean_history = self.memory.load_history(session_id)
        
        print(f"Loading Cross-Encoder: {CROSS_ENCODER_MODEL}...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    def call_llm(self, system_prompt: str, messages: List[Any]) -> str:
        # Ollama usually takes a single string prompt
        # We format the system prompt and messages into a coherent conversation
        prompt = f"System: {system_prompt}\n\n"
        for m in messages:
            role = m.type if hasattr(m, 'type') else m['role']
            content = m.content if hasattr(m, 'content') else m['content']
            prompt += f"{role}: {content}\n"
        
        return self.llm.invoke(prompt)

    def summarize_history(self, threshold=20, summarize_count=10):
        """
        Compresses clean history by summarizing the oldest part.
        """
        if len(self.clean_history) < threshold:
            return False
        
        print("Summarizing conversation history to save tokens...")
        to_summarize = self.clean_history[:summarize_count]
        remaining = self.clean_history[summarize_count:]
        
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        
        system_prompt = (
            "You are a Legal Secretary. Summarize the following conversation history into a "
            "concise 'Contextual Summary' that preserves all legal facts, entities, and user goals. "
            "This summary will be used as the starting context for future responses."
        )
        
        summary = self.call_llm(system_prompt, [HumanMessage(content=history_str)])
        
        # Replace the oldest part with a single summary message
        self.clean_history = [{"role": "system", "content": f"CONTEXT SUMMARY: {summary}"}] + remaining
        return True

    def analyze_intent_and_generate_hp(self, user_input: str, profile: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        system_prompt = (
            "You are a Legal Intent Analyzer and Knowledge Architect. "
            "Step 1: Analyze if the last user query is related to entrepreneurship or legal matters, "
            "considering the provided conversation history.\n"
            "Step 2: If related, create a 'Hypothetical Passage' (a factual-sounding legal excerpt) "
            "Keep it neutrally descriptive, NO conclusions or advice.\n\n"
            "OUTPUT FORMAT: Return STRICT JSON only:\n"
            "{\n"
            "  \"related\": boolean,\n"
            "  \"hypothetical_passage\": \"The generated passage here or null if not related\"\n"
            "}"
        )
        
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.clean_history])
        
        context = (
            f"CLEAN CONVERSATION HISTORY:\n{history_str}\n\n"
            f"LAST USER QUERY: {user_input}"
        )
        
        res_text = self.call_llm(system_prompt, [HumanMessage(content=context)])
        
        try:
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if match:
                res = json.loads(match.group())
                return res.get("related", False), res.get("hypothetical_passage")
            return False, None
        except Exception as e:
            print(f"Intent Analysis Error: {e}")
            return False, None

    def rerank_and_filter(self, query: str, chunks: List[str], threshold: float = -5.0) -> List[str]:
        if not chunks: return []
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)
        scored_chunks = [(score, chunk) for score, chunk in zip(scores, chunks) if score > threshold]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks]

    def run_retrieval_loop(self, hyp_passage: str, original_query: str) -> List[str]:
        system_prompt = (
            "You are a Legal Retrieval Expert. Your goal is to find real legal excerpts that validate "
            "the claims made in a 'Hypothetical Passage'.\n\n"
            "### THINKING PROCESS:\n"
            "1. **Gap Analysis**: Compare the 'Hypothetical Passage' with the 'Retrieved Chunks' in the attempts log.\n"
            "2. **Fact Extraction**: Identify missing legal rules, ordinance names, or section numbers.\n"
            "3. **Tool Selection**: `keyword_search` for names/IDs, `semantic_search` for concepts, `hybrid_search` for both.\n"
            "4. **Termination**: If the evidence is sufficient, set `is_complete` to true.\n\n"
            "### OUTPUT FORMAT:\n"
            "Valid JSON only. Example: {\"thought\": \"...\", \"action\": \"...\", \"action_input\": \"...\", \"is_complete\": false}"
        )
        
        retrieved_chunks = []
        attempts_log = []
        
        for attempt in range(1, 4):
            history_str = "\n".join([
                f"Attempt {i+1} [{log['tool']}]:\n{log['results']}" 
                for i, log in enumerate(attempts_log)
            ]) if attempts_log else "No chunks retrieved yet."

            retrieval_context = (
                f"HYPOTHETICAL PASSAGE:\n{hyp_passage}\n\n"
                f"RETRIEVED CHUNKS LOG:\n{history_str}"
            )
            
            response_text = self.call_llm(system_prompt, [HumanMessage(content=retrieval_context)])
            
            try:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not match: continue
                res = json.loads(match.group())
                if res.get("is_complete"): return retrieved_chunks
                
                tool_name, query = res.get("action"), res.get("action_input")
                if not tool_name or not query: continue

                if tool_name == "semantic_search": raw_obs = self.tools.semantic_search(query)
                elif tool_name == "keyword_search": raw_obs = self.tools.keyword_search(query)
                elif tool_name == "hybrid_search": raw_obs = self.tools.hybrid_search(query)
                else: raw_obs = ["Error: Unknown tool"]
                
                filtered_obs = self.rerank_and_filter(original_query, raw_obs)
                obs_text = "\n".join(filtered_obs)
                retrieved_chunks.extend(filtered_obs)
                attempts_log.append({"tool": tool_name, "results": obs_text})
                
            except Exception:
                if attempt == 3: break
                continue

        return retrieved_chunks

    def generate_final_response(self, chunks: List[str], profile: Optional[Dict[str, Any]]) -> str:
        system_prompt = (
            "You are an expert Legal Consultant for startups. Use the provided legal chunks to answer the user's query. "
            "Refer to the User Profile (startup stage/industry) to personalize the advice and suggest a follow-up question or legal terms to learn. "
            "CRITICAL: Rely ONLY on the provided legal chunks. If the chunks don't contain the answer, state it clearly. "
            "Do not invent legal rules. "
        )
        
        profile_str = json.dumps(profile) if profile else "No user profile available."
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.clean_history])
        
        context = (
            f"USER PROFILE: {profile_str}\n\n"
            f"CONVERSATION HISTORY:\n{history_str}\n\n"
            f"LEGAL CHUNKS:\n{os.linesep.join(chunks)}\n\n"
            f"Please provide a detailed and professional response based on the above."
        )
        return self.call_llm(system_prompt, [HumanMessage(content=context)])

    def run(self):
        profile = self.memory.load_profile()
        print(f"\n--- Agentic Legal RAG [Ollama Optimized] ---")
        
        while True:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() == ":q": break

            # Trigger summarization if history is too long
            self.summarize_history()

            related, hyp_passage = self.analyze_intent_and_generate_hp(user_input, profile)
            self.interface_history.append({"role": "user", "content": user_input})
            
            if not related:
                declaration = "I am a domain-specific tool specializing in entrepreneurship legal knowledge."
                print(f"Bot: {declaration}")
                self.interface_history.append({"role": "bot", "content": declaration})
                self.memory.save_history(self.session_id, self.interface_history, self.clean_history)
                continue

            self.clean_history.append({"role": "user", "content": user_input})

            print("Retrieving...", end="", flush=True)
            chunks = self.run_retrieval_loop(hyp_passage, user_input)
            
            print("Generating Response...", end="", flush=True)
            final_answer = self.generate_final_response(chunks, profile)
            
            print("\r" + " " * 30 + "\r", end="")
            print(f"Bot: {final_answer}\n")

            self.interface_history.append({"role": "bot", "content": final_answer})
            self.clean_history.append({"role": "bot", "content": final_answer})
            self.memory.save_history(self.session_id, self.interface_history, self.clean_history)
