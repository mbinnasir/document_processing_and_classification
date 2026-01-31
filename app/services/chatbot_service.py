import ollama
import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import Document
from app.services.search_engine import SearchEngine

class ChatbotService:
    def __init__(self, model_name: str = "ministral-3:latest"):
        self.model_name = model_name
        self.search_engine = SearchEngine()

    def chat(self, query: str, db: Session) -> str:
        # 1. Search for relevant context using embeddings
        query_embedding = self.search_engine.model.encode(query).tolist()
        
        # In SQLite with sqlite-vec, we'll fetch all docs and their embeddings, 
        # then find similarities. For a really large dataset, we'd use a virtual table.
        # But for this simple implementation, we'll fetch docs that have embeddings.
        docs_with_embeddings = db.query(Document).filter(Document.vector_embeddings.isnot(None)).all()
        
        # Simple manual similarity if no virtual table is set up yet
        # We'll sort by cosine similarity in memory
        from sentence_transformers import util
        import torch
        
        relevant_docs = []
        if docs_with_embeddings:
            embeddings = [doc.vector_embeddings for doc in docs_with_embeddings]
            cos_scores = util.cos_sim(torch.tensor(query_embedding), torch.tensor(embeddings))[0]
            top_results = torch.topk(cos_scores, k=min(3, len(docs_with_embeddings)))
            
            for score, idx in zip(top_results[0], top_results[1]):
                relevant_docs.append(docs_with_embeddings[idx.item()])

        context = ""
        for doc in relevant_docs:
            if doc.processed_output:
                context += f"Document: {doc.document_name}\nData: {json.dumps(doc.processed_output)}\n\n"
            else:
                context += f"Document: {doc.document_name}\nContent: {doc.content[:500]}...\n\n"

        # 2. Build prompt for the LLM
        prompt = f"""
        You are a helpful document assistant. Use the following context from processed documents to answer the user's question.
        If the user asks for specific JSON data, provide it from the context.
        If the answer is not in the context, say you don't know based on the current documents.

        Context:
        {context}

        User Question: {query}
        """

        try:
            print(f"DEBUG: Chatting with LLM using context from {len(relevant_docs)} docs")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.5,
                }
            )
            return response['response']
        except Exception as e:
            print(f"DEBUG ERROR: Chatbot error: {e}")
            return f"I encountered an error while processing your request: {e}"
