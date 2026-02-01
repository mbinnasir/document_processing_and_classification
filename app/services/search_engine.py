from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Dict, Any

class SearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            self.model = None
            raise HTTPException(status_code=500, detail=str(e))
        self.documents = []
        self.embeddings = None

    def index_documents(self, docs: List[Dict[str, Any]]):
        if not self.model or not docs:
            return

        self.documents = docs
        texts = [doc['text'] for doc in docs]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.model or self.embeddings is None:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        top_results = torch.topk(cos_scores, k=min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            doc_idx = idx.item()
            results.append({
                "score": float(score),
                "filename": self.documents[doc_idx]['filename'],
                "metadata": self.documents[doc_idx].get('metadata', {}),
                "snippet": self.documents[doc_idx]['text'][:300] + "..."
            })
        
        return results
