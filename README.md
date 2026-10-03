# LegalRAG Agent ⚖️

A professional, agentic RAG (Retrieval-Augmented Generation) system designed for entrepreneurship and legal consultation. This system moves beyond simple retrieval by using an iterative reasoning loop (ReAct) and a dual-context memory system to provide grounded, personalized legal advice.

## 🚀 Core Architecture

The system transforms a user query into a professional legal response through the following pipeline:

### 1. Intent Analysis & Contextual HP Generation
To avoid the "vocabulary gap" between casual user queries and formal legal texts, the agent uses **Hypothetical Document Embeddings (HyDE)**:
- **Combined Analysis**: In a single LLM call, the agent analyzes the full `clean_history` and `user_profile` to judge if the query is domain-related.
- **HP Generation**: If related, it generates a "Hypothetical Passage"—a factual-sounding legal excerpt that represents the ideal answer. This is used as the search query for the retrieval stage.

### 2. Agentic Retrieval Loop (ReAct)
The agent operates as a reasoning entity that can iteratively refine its search:
- **Hybrid Search**: Combines **Semantic Search** (Vector DB) and **Keyword Search** (BM25) using **Reciprocal Rank Fusion (RRF)**.
- **Cross-Encoder Re-ranking**: All retrieved chunks are filtered through a `cross-encoder` to remove noise and rank the most legally relevant texts.
- **Iterative Refinement**: If the agent determines that the retrieved chunks are insufficient to validate the Hypothetical Passage, it will autonomously adjust its query and search again (up to 3 attempts).

### 3. Memory & Personalization
- **Dual-Context History**: 
    - `interface_history`: Stores every message (including out-of-domain) for UI display.
    - `clean_history`: Stores only domain-related interactions, used as the source for LLM reasoning.
- **Context Compression**: To maintain efficiency, the system automatically summarizes the oldest part of the `clean_history` once it exceeds a threshold, preserving key legal facts while saving tokens.
- **User Profiling**: A dedicated `ProfileManager` allows the agent to extract business details (industry, stage, goals) into a `user_profile.json` via a manual sync process, enabling highly personalized advice.

---

## 🛠️ Tech Stack

- **LLM**: `gemma4:31b-cloud` via **Ollama** (enabling optimized KV caching).
- **Vector Store**: `ChromaDB`.
- **Embeddings**: `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
- **Re-ranking**: `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`).
- **Backend**: `FastAPI` + `Uvicorn`.
- **Frontend**: `React` + `Tailwind CSS` + `Vite`.

---

## 📂 Project Structure

- `server.py`: FastAPI server managing multi-window session IDs and API routing.
- `agentic_rag_bot.py`: The core agent logic (Intent $\rightarrow$ HP $\rightarrow$ ReAct Loop $\rightarrow$ Response).
- `profile_manager.py`: Logic for summarizing conversations into a persistent user profile.
- `tools.py`: Implementation of Hybrid Search (BM25 + Vector + RRF).
- `frontend/`: Full React application for interacting with the agent.
- `/memory`: Storage for session files (`_full.json`, `_clean.json`) and the global `user_profile.json`.
- `launch.sh`: One-click script to start the backend and frontend.

---

## 🚦 Getting Started

### Prerequisites
1. Install [Ollama](https://ollama.ai/) and pull the model:
   ```bash
   ollama pull gemma4:31b-cloud
   ```
2. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn langchain_community langchain_huggingface sentence-transformers rank_bm25 chromadb
   ```

### Launching the System
Run the provided shell script from the root directory:
```bash
chmod +x launch.sh
./launch.sh
```
- **Backend**: Accessible at `http://localhost:8000`
- **Frontend**: Accessible at `http://localhost:5173`

---

## 🧪 Evaluation Strategy
The system is designed to be tested against a "Gold Dataset" focusing on:
- **Faithfulness**: Zero-tolerance for hallucinations (responses must be grounded in chunks).
- **Multi-hop Reasoning**: Ability to use multiple tools to answer complex legal questions.
- **Profile Accuracy**: Correct extraction of user business stages into the JSON profile.
