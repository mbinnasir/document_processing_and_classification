import ollama
import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import Document
from app.services.search_engine import SearchEngine
from sentence_transformers import util
import torch


class ChatbotService:
    def __init__(self, model_name: str = "ministral-3:latest"):
        self.model_name = model_name
        self.search_engine = SearchEngine()

    def chat(self, query: str, db: Session) -> str:
        query_embedding = self.search_engine.model.encode(query).tolist()
        docs_with_embeddings = db.query(Document).filter(Document.vector_embeddings.isnot(None)).all()
        
        # Filter out documents that might have None embeddings despite the query filter
        # This handles cases where JSON null might be returned or other anomalies
        valid_docs = [doc for doc in docs_with_embeddings if doc.vector_embeddings is not None and isinstance(doc.vector_embeddings, list)]
        
        relevant_docs = []
        if valid_docs:
            embeddings = [doc.vector_embeddings for doc in valid_docs]
            # Ensure query_embedding is also a tensor
            query_tensor = torch.tensor(query_embedding)
            embeddings_tensor = torch.tensor(embeddings)
            
            cos_scores = util.cos_sim(query_tensor, embeddings_tensor)[0]
            top_results = torch.topk(cos_scores, k=min(3, len(valid_docs)))
            
            for score, idx in zip(top_results[0], top_results[1]):
                relevant_docs.append(valid_docs[idx.item()])

        context = ""
        for doc in relevant_docs:
            if doc.processed_output:
                context += f"Document: {doc.document_name}\nData: {json.dumps(doc.processed_output)}\n\n"
            else:
                context += f"Document: {doc.document_name}\nContent: {doc.content[:500]}...\n\n"

        print("Content:",context)
        prompt = f"""
        You are a helpful document assistant. Use the following context from processed documents to answer the user's question.
        If the user asks for specific JSON data, provide it from the context.
        If the answer is not in the context, say you don't know based on the current documents.
        also only return the answer in the format of json which is given no extra context

        Context:
        {context}

        User Question: {query}
        """

        try:
            # print(f"DEBUG: Chatting with LLM using context from {len(relevant_docs)} docs")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.5,
                }
            )
            print(f"DEBUG: LLM response: {response['response']}")
            content = response['response'].strip()
            if content.startswith("```"):
                content = content.strip("`").replace("json\n", "", 1).strip()
            
            try:
                return content
            except json.JSONDecodeError:
                pass
        except Exception as e:
            print(f"DEBUG ERROR: Chatbot error: {e}")
            return f"I encountered an error while processing your request: {e}"
