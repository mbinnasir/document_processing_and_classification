import ollama
import json
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import Document
from app.services.search_engine import SearchEngine
from sentence_transformers import util
import torch
from dotenv import load_dotenv
import os
from fastapi import HTTPException

load_dotenv()

print("Model_name", os.getenv("OLLAMA_MODEL"))


class ChatbotService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL") or "qwen3-vl:latest"
        self.search_engine = SearchEngine()

    def chat(self, query: str, db: Session) -> str:
        # Fetch all documents to provide full context as requested
        all_docs = db.query(Document).all()

        context = ""
        for doc in all_docs:
            if doc.processed_output:
                context += f"Document: {doc.document_name}\nData: {json.dumps(doc.processed_output)}\n\n"
            elif doc.content:
                context += f"Document: {doc.document_name}\nContent: {doc.content[:500]}...\n\n"

        print("Content:", context)
        # prompt = f"""
        # You are a highly capable Document AI Assistant. Your goal is to provide precise, structured answers based on the provided document context.

        # ### GUIDELINES:
        # 1. **RELEVANCE IS CRITICAL**: Only include documents and data points that are directly relevant to the user's query. If a document does not contain the specific information requested, EXCLUDE it from your response.
        # 2. **STRICT JSON**: Your response must be valid JSON. No conversational text, no comments (//), no markdown titles. Just the JSON object or array.
        # 3. **FIELD PRECISION**: Pay attention to field names. For example, if a user asks for "due dates", look for fields like 'due_date' or 'amount_due' in Utility Bills. Do not confuse generic 'date' fields in Invoices with 'due dates' unless they are explicitly labeled as such.
        # 4. **FILTERING**: If the user's query implies a category (e.g., "bills"), do not include other categories (e.g., "invoices").
        # 5. **NO EXTRA FIELDS**: Only return the fields requested or the most important identifiers (like document_name, date, total) if a summary is asked.

        # ### CONTEXT:
        # {context}

        # ### USER QUESTION:
        # {query}
        # """
        prompt = f"""
        You are a strict Document Extraction Engine.

        Your task is to FILTER documents and RETURN structured data that EXACTLY matches the user's request.

        ### ABSOLUTE RULES (DO NOT VIOLATE):
        1. OUTPUT MUST BE VALID JSON ONLY.
        - No explanations
        - No markdown
        - No comments
        2. INCLUDE A DOCUMENT ONLY IF IT FULLY MATCHES ALL FILTER CONDITIONS.
        3. IF A DOCUMENT DOES NOT CONTAIN THE EXACT FIELD REQUESTED, EXCLUDE IT.
        4. WHEN A TERM HAS A DOMAIN-SPECIFIC MEANING, USE THE CORRECT DOCUMENT TYPE:
        - "due amount" → Utility Bill → field: amount_due
        - Invoices use total_amount and MUST NOT be treated as due amounts.
        5. DATE FILTERS ARE STRICT:
        - "June" means month == 06 only.
        - Any other month MUST be excluded.
        6. DO NOT INFER OR SUBSTITUTE FIELDS.
        7. DO NOT MIX DOCUMENT TYPES UNLESS EXPLICITLY REQUESTED.

        ### FIELD MAPPING REFERENCE:
        - Utility Bill:
        - date
        - amount_due
        - account_number
        - Invoice:
        - date
        - total_amount (NOT a due amount)

        ### CONTEXT DOCUMENTS:
        {context}

        ### USER QUERY:
        {query}

        ### REQUIRED OUTPUT FORMAT:
        {{
        "response": [ <filtered objects only> ]
        }}
        """
        # print("Model_name":self.model_name)
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Lower temperature for better JSON consistency
                },
            )
            content = response["response"].strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip("`").strip()

            # Attempt to strip C-style comments if the LLM included them
            import re

            content = re.sub(r"//.*", "", content)

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
