import re
from typing import Dict, Any, Optional
from dateutil import parser as date_parser

class DataExtractor:
    def __init__(self):
        # Regular expressions for extraction
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        self.amount_pattern = r'\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})'
        self.invoice_num_pattern = r'(?:Invoice|INV|#)\s?[:#]?\s?([A-Za-z0-9-]+)'
        self.account_num_pattern = r'(?:Account|Acc)\s?[:#]?\s?([A-Za-z0-9-]+)'
        self.usage_kwh_pattern = r'(\d+\.?\d*)\s?kwh'

    def extract(self, text: str, doc_class: str) -> Dict[str, Any]:
        """Extract fields based on document class."""
        if doc_class == "Invoice":
            return self._extract_invoice(text)
        elif doc_class == "Resume":
            return self._extract_resume(text)
        elif doc_class == "Utility Bill":
            return self._extract_utility_bill(text)
        return {}

    def _extract_invoice(self, text: str) -> Dict[str, Any]:
        invoice_num = self._find_first(self.invoice_num_pattern, text)
        date = self._find_date(text)
        total_amount = self._find_amount(text, ["total", "amount", "due", "balance"])
        company = self._find_company(text)

        return {
            "invoice_number": invoice_num,
            "date": date,
            "company": company,
            "total_amount": float(total_amount.replace('$', '').replace(',', '')) if total_amount else None
        }

    def _extract_resume(self, text: str) -> Dict[str, Any]:
        name = self._find_name(text)
        email = self._find_first(self.email_pattern, text)
        phone = self._find_first(self.phone_pattern, text)
        experience = self._find_experience(text)

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "experience_years": experience
        }

    def _extract_utility_bill(self, text: str) -> Dict[str, Any]:
        account_num = self._find_first(self.account_num_pattern, text)
        date = self._find_date(text)
        usage_kwh = self._find_first(self.usage_kwh_pattern, text, ignore_case=True)
        amount_due = self._find_amount(text, ["amount due", "total due", "payable"])

        return {
            "account_number": account_num,
            "date": date,
            "usage_kwh": float(usage_kwh) if usage_kwh else None,
            "amount_due": float(amount_due.replace('$', '').replace(',', '')) if amount_due else None
        }

    def _find_first(self, pattern: str, text: str, ignore_case: bool = False) -> Optional[str]:
        flags = re.IGNORECASE if ignore_case else 0
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        return None

    def _find_date(self, text: str) -> Optional[str]:
        # Simple date finder
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', # 01/01/2023
            r'\w+\s\d{1,2},?\s\d{4}', # January 1, 2023
            r'\d{4}[-]\d{2}[-]\d{2}'  # 2023-01-01
        ]
        for p in date_patterns:
            match = re.search(p, text)
            if match:
                try:
                    dt = date_parser.parse(match.group(0))
                    return dt.strftime("%Y-%m-%d")
                except:
                    continue
        return None

    def _find_amount(self, text: str, keywords: list) -> Optional[str]:
        for kw in keywords:
            # Look for keyword followed by an amount
            pattern = rf'{kw}.*?({self.amount_pattern})'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        
        # Fallback: find any amount
        return self._find_first(self.amount_pattern, text)

    def _find_company(self, text: str) -> Optional[str]:
        # Basic approach: first line often has company name
        lines = text.splitlines()
        if lines:
            return lines[0].strip()
        return None

    def _find_name(self, text: str) -> Optional[str]:
        # Basic approach: first line of a resume often has the name
        lines = text.splitlines()
        for line in lines[:3]: # check first 3 lines
            if line.strip() and len(line.strip().split()) >= 2:
                return line.strip()
        return None

    def _find_experience(self, text: str) -> int:
        match = re.search(r'(\d+)\+?\s?years?', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
