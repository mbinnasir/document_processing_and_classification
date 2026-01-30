import ollama
import json
import logging
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMExtractor:
    def __init__(self, model_name: str = "ministral-3:latest"):
        self.model_name = model_name
        self.timeout = 60 # seconds

    def extract(self, text: str, doc_class: str) -> Dict[str, Any]:
        """
        Extract structured data from text using a local LLM.
        """
        prompt = self._build_prompt(text, doc_class)
        
        try:
            print(f"DEBUG: Extracting data using model: {self.model_name}")
            # print(f"DEBUG: Prompt sent to LLM: {prompt}") # Commented out to avoid cluttering terminal with large text
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 512,
                }
            )
            
            print("DEBUG: Received response from Ollama")
            print(f"DEBUG: Raw LLM response: {response['response']}")
            
            # Parse the JSON response
            extracted_data = json.loads(response['response'])
            print("DEBUG: Successfully parsed LLM response JSON")
            return extracted_data
            
        except Exception as e:
            print(f"DEBUG ERROR: Error during LLM extraction: {e}")
            return {"error": str(e), "fallback": "True"}

    def _build_prompt(self, text: str, doc_class: str) -> str:
        """
        Construct a prompt for the LLM to both classify and extract data.
        """
        prompt = f"""
        Analyze the text below. 
        1. Identify what type of document it is (e.g., Invoice, Resume, Utility Bill, or Other).
        2. Extract relevant structured information based on that type.
        
        Expected fields for common types:
        - Invoice: invoice_number, date, total_amount, currency, vendor_name
        - Resume: name, email, phone, experience_years, latest_job_title
        - Utility Bill: account_number, date, amount_due, usage_kwh
        
        Return the result strictly as a JSON object with a "document_type" field and all extracted data.
        
        Text to process:
        ---
        {text[:2500]}
        ---
        
        Return ONLY the JSON object.
        """
        return prompt

    def _get_schema_for_class(self, doc_class: str) -> Dict[str, str]:
        # This method is now legacy as the LLM determines schemas dynamically in the new prompt
        return {}
