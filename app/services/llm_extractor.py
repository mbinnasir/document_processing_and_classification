import ollama
import json
import logging
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMExtractor:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL") or "qwen3-vl:latest"
        self.timeout = 60

    def extract(self, text: str, doc_class: str) -> Dict[str, Any]:
        prompt = self._build_prompt(text, doc_class)

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 512,
                },
            )

            raw_data = json.loads(response["response"])

            if "extracted_data" in raw_data and isinstance(
                raw_data["extracted_data"], dict
            ):
                extracted_data = raw_data["extracted_data"]
                extracted_data["document_type"] = raw_data.get("document_type", "Other")
                return extracted_data

            return raw_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _build_prompt(self, text: str, doc_class: str) -> str:
        prompt = f"""
        Analyze the text below and extract structured data.
        
        STRICT CLASSIFICATION RULES:
        Classify the document into exactly ONE of these types:
        - "Invoice"
        - "Resume"
        - "Utility Bill"
        - "Other" (if it doesn't clearly fit the above)

        REQUIRED EXTRACTION SCHEMAS:
        
        1. IF "Invoice":
           - invoice_number (string)
           - date (string, YYYY-MM-DD format if possible)
           - company (string, vendor name)
           - total_amount (string or number)
           
        2. IF "Resume":
           - name (string)
           - email (string)
           - phone (string)
           - experience_years (number, estimate if needed)
           
        3. IF "Utility Bill":
           - account_number (string)
           - date (string)
           - usage_kwh (string or number)
           - amount_due (string or number)
           
        4. IF "Other":
           - summary (string, brief summary of content)

        OUTPUT FORMAT:
        Return ONLY a raw JSON object. Do not include markdown formatting (```json).
        The JSON must have this exact structure:
        {{
            "document_type": "The Class Name",
            "extracted_data": {{ ... fields based on schema ... }}
        }}
        
        Text to process:
        ---
        {text[:2500]}
        ---
        
        Return ONLY the valid JSON object.
        """
        return prompt

    def _get_schema_for_class(self, doc_class: str) -> Dict[str, str]:
        return {}
