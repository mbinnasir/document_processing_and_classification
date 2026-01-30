from transformers import pipeline
from typing import List, Dict, Any
import re

class DocumentClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.candidate_labels = ["Invoice", "Resume", "Utility Bill", "Other"]
        self.threshold = 0.5
        try:
            self.classifier = pipeline("zero-shot-classification", model=model_name)
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            self.classifier = None

    def classify(self, text: str) -> str:
        if not text:
            return "Unclassifiable"

        # Keyword-based check first (faster)
        keywords_scores = self._keyword_check(text)
        top_keyword = max(keywords_scores, key=keywords_scores.get)
        if keywords_scores[top_keyword] > 0.8: # High confidence from keywords
            return top_keyword

        # ML-based classification
        if self.classifier:
            try:
                # Truncate text to fit model context window (typically 512 or 1024 tokens)
                # Using a rough character limit
                truncated_text = text[:1000] 
                result = self.classifier(truncated_text, self.candidate_labels)
                
                top_label = result['labels'][0]
                top_score = result['scores'][0]

                if top_score >= self.threshold:
                    return top_label
                else:
                    return "Unclassifiable"
            except Exception as e:
                print(f"ML Classification error: {e}")
                return self._best_from_keywords(keywords_scores)
        
        return self._best_from_keywords(keywords_scores)

    def _keyword_check(self, text: str) -> Dict[str, float]:
        text = text.lower()
        scores = {label: 0.0 for label in self.candidate_labels}
        
        # Invoice markers
        if any(word in text for word in ["invoice", "bill to", "total amount", "tax invoice", "subtotal"]):
            scores["Invoice"] += 0.6
        if re.search(r'inv-\d+|invoice #', text):
            scores["Invoice"] += 0.3

        # Resume markers
        if any(word in text for word in ["resume", "experience", "education", "skills", "projects", "work history"]):
            scores["Resume"] += 0.6
        if any(word in text for word in ["curriculum vitae", "summary", "languages"]):
            scores["Resume"] += 0.2

        # Utility Bill markers
        if any(word in text for word in ["utility bill", "kwh", "electricity", "water bill", "gas bill", "account number"]):
            scores["Utility Bill"] += 0.7
        if any(word in text for word in ["usage", "meter reading", "service period"]):
            scores["Utility Bill"] += 0.3

        return scores

    def _best_from_keywords(self, scores: Dict[str, float]) -> str:
        top_label = max(scores, key=scores.get)
        if scores[top_label] > 0.4:
            return top_label
        return "Unclassifiable"
