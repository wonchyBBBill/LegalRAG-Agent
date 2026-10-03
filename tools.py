import os
import json
from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

class LegalRetrievalTools:
    """
    Standalone Tool Provider for Legal Retrieval.
    Includes Semantic Search (Vector), Keyword Search (BM25), and Hybrid Search (RRF).
    """
    def __init__(self, db_path="./law_md_db", md_dir="./law_md", embedding_model="all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.md_dir = md_dir
        
        # Initialize Corpus and BM25
        self.corpus = self._load_corpus()
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_corpus(self):
        """Loads all markdown files from law_md to support keyword search."""
        corpus = []
        if os.path.exists(self.md_dir):
            for file in os.listdir(self.md_dir):
                if file.endswith(".md"):
                    with open(os.path.join(self.md_dir, file), 'r', encoding='utf-8') as f:
                        # Splitting by double newline to treat paragraphs as documents
                        chunks = f.read().split("\n\n")
                        corpus.extend([c.strip() for c in chunks if c.strip()])
        return corpus

    def semantic_search(self, query: str, k=3) -> List[str]:
        """Vector-based search for conceptual matching."""
        docs = self.vector_db.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def keyword_search(self, query: str, k=3) -> List[str]:
        """BM25-based keyword search for specific terms and technical phrasing."""
        tokenized_query = query.lower().split()
        # get_top_n returns the docs based on BM25 score
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return results

    def hybrid_search(self, query: str, k=3) -> List[str]:
        """
        Combines semantic and keyword search results using Reciprocal Rank Fusion (RRF).
        RRF helps normalize scores from different retrieval methods.
        """
        # Increase k during retrieval to ensure we have enough overlap for RRF
        search_k = k * 2
        sem_res = self.semantic_search(query, k=search_k)
        key_res = self.keyword_search(query, k=search_k)

        # RRF constant (standard value is 60)
        K = 60
        scores = {}

        # Score semantic results
        for rank, doc in enumerate(sem_res, 1):
            scores[doc] = scores.get(doc, 0) + 1.0 / (K + rank)

        # Score keyword results
        for rank, doc in enumerate(key_res, 1):
            scores[doc] = scores.get(doc, 0) + 1.0 / (K + rank)

        # Sort by fused score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:k]]

# For testing the tool independently
if __name__ == "__main__":
    tools = LegalRetrievalTools()
    print("\n--- Testing Semantic Search ---")
    print(tools.semantic_search("company registration")[:1])
    
    print("\n--- Testing Keyword Search (BM25) ---")
    print(tools.keyword_search("Registration Ordinance")[:1])
    
    print("\n--- Testing Hybrid Search (RRF) ---")
    print(tools.hybrid_search("Registration Ordinance")[:1])
