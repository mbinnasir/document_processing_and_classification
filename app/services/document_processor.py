import pdfplumber
import os
from typing import Optional

class DocumentProcessor:
    def __init__(self):
        pass

    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from a PDF or text file."""
        if file_path.lower().endswith('.pdf'):
            return self._extract_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            return self._extract_from_txt(file_path)
        return None

    def _extract_from_pdf(self, file_path: str) -> Optional[str]:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip() if text else None
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return None

    def _extract_from_txt(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        # Remove extra whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
