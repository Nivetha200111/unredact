"""
AI Classification Module - Uses GPT-4 to classify entities as PEP or Victim
"""
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

import config
from pdf_processor import PDFDocument
from unredactor import UnredactionResult


class EntityType(Enum):
    """Classification categories for entities"""
    PEP = "PEP"  # Politically Exposed Person
    VICTIM = "VICTIM"
    UNKNOWN = "UNKNOWN"
    OTHER = "OTHER"


@dataclass
class ClassifiedEntity:
    """Represents a classified entity from the documents"""
    name: str
    email: Optional[str]
    entity_type: EntityType
    confidence: float
    reasoning: str
    source_document: str
    source_page: Optional[int]
    was_redacted: bool
    context: str


class AIClassifier:
    """
    Uses GPT-4 to analyze text and classify mentioned entities
    as either PEPs (Politically Exposed Persons) or victims
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_MODEL
    
    def _extract_entities_prompt(self, text: str, context: str = "") -> str:
        """Create prompt for entity extraction"""
        return f"""Analyze the following text and extract all person names and email addresses mentioned.
For each person/entity found, determine if they are:
1. PEP (Politically Exposed Person) - government officials, politicians, executives, people in positions of power
2. VICTIM - someone who appears to be a victim of wrongdoing, fraud, abuse, or crime
3. OTHER - neither of the above

Context about the document: {context}

Text to analyze:
---
{text[:4000]}
---

Respond in JSON format with an array of entities:
{{
    "entities": [
        {{
            "name": "Full Name",
            "email": "email@example.com or null",
            "type": "PEP|VICTIM|OTHER",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation for classification"
        }}
    ]
}}

Important considerations:
- PEPs include: politicians, government officials, executives, judges, military officers, board members
- VICTIMs include: plaintiffs in lawsuits, targets of fraud, abuse survivors, whistleblowers facing retaliation
- Consider the context and tone when classifying
- If unclear, classify as OTHER with lower confidence
- Extract ALL names mentioned, even if you can't classify them confidently"""
    
    def _classify_redacted_content_prompt(self, 
                                          redacted_text: str, 
                                          surrounding_context: str,
                                          document_context: str) -> str:
        """Create prompt for classifying specifically redacted content"""
        return f"""The following text was REDACTED from a document but has been recovered.
Analyze what was hidden and why it might have been redacted.

Document context: {document_context}

Surrounding context:
{surrounding_context[:1000]}

RECOVERED REDACTED TEXT:
---
{redacted_text}
---

Determine:
1. What entities (people/emails) are mentioned in the redacted text
2. Whether each entity appears to be a PEP or VICTIM
3. Why this content was likely redacted (protecting identity, hiding wrongdoing, privacy, etc.)

Respond in JSON format:
{{
    "entities": [
        {{
            "name": "Full Name",
            "email": "email@example.com or null",
            "type": "PEP|VICTIM|OTHER|UNKNOWN",
            "confidence": 0.0-1.0,
            "reasoning": "Why this classification and why it was likely redacted"
        }}
    ],
    "redaction_reason": "Overall reason for redaction"
}}"""
    
    def _call_gpt(self, prompt: str) -> Optional[Dict]:
        """Make a call to GPT-4 and parse the JSON response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert analyst specializing in identifying Politically Exposed Persons (PEPs) and victims in legal and financial documents. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            print(f"    Warning: GPT API call failed: {e}")
            
        return None
    
    def classify_document(self, 
                         pdf_doc: PDFDocument, 
                         unredaction_results: List[UnredactionResult]) -> List[ClassifiedEntity]:
        """
        Classify all entities found in a document, including recovered redacted text
        """
        classified_entities = []
        
        print(f"  Classifying entities in {pdf_doc.filename}...")
        
        # First, analyze the main document text
        document_context = f"Document: {pdf_doc.filename}, Pages: {pdf_doc.num_pages}"
        
        # Split text into chunks for analysis
        text_chunks = self._split_text(pdf_doc.total_text, chunk_size=3000)
        
        for i, chunk in enumerate(text_chunks):
            prompt = self._extract_entities_prompt(chunk, document_context)
            result = self._call_gpt(prompt)
            
            if result and "entities" in result:
                for entity_data in result["entities"]:
                    entity = self._create_entity(
                        entity_data,
                        pdf_doc.filename,
                        was_redacted=False,
                        context=chunk[:200]
                    )
                    if entity:
                        classified_entities.append(entity)
        
        # Now analyze specifically the recovered redacted content
        for unredact_result in unredaction_results:
            if unredact_result.success and unredact_result.recovered_text:
                # Get surrounding context
                surrounding = self._get_surrounding_context(
                    pdf_doc.total_text,
                    unredact_result.original_redaction.page_num
                )
                
                prompt = self._classify_redacted_content_prompt(
                    unredact_result.recovered_text,
                    surrounding,
                    document_context
                )
                result = self._call_gpt(prompt)
                
                if result and "entities" in result:
                    for entity_data in result["entities"]:
                        entity = self._create_entity(
                            entity_data,
                            pdf_doc.filename,
                            was_redacted=True,
                            context=unredact_result.recovered_text,
                            page_num=unredact_result.original_redaction.page_num
                        )
                        if entity:
                            classified_entities.append(entity)
        
        # Also check emails found in the document
        for email in pdf_doc.emails_found:
            # Check if we already have this email classified
            if not any(e.email == email for e in classified_entities):
                classified_entities.append(ClassifiedEntity(
                    name="Unknown",
                    email=email,
                    entity_type=EntityType.UNKNOWN,
                    confidence=0.5,
                    reasoning="Email found in document but entity not yet classified",
                    source_document=pdf_doc.filename,
                    source_page=None,
                    was_redacted=False,
                    context=""
                ))
        
        # Deduplicate entities
        classified_entities = self._deduplicate_entities(classified_entities)
        
        pep_count = sum(1 for e in classified_entities if e.entity_type == EntityType.PEP)
        victim_count = sum(1 for e in classified_entities if e.entity_type == EntityType.VICTIM)
        print(f"    Found {len(classified_entities)} entities: {pep_count} PEPs, {victim_count} victims")
        
        return classified_entities
    
    def _split_text(self, text: str, chunk_size: int = 3000) -> List[str]:
        """Split text into chunks for API processing"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _get_surrounding_context(self, full_text: str, page_num: int) -> str:
        """Get text context around a specific page"""
        # Rough estimate: divide text by pages
        lines = full_text.split('\n')
        lines_per_page = max(1, len(lines) // 10)  # Rough estimate
        
        start = max(0, page_num * lines_per_page - 20)
        end = min(len(lines), (page_num + 1) * lines_per_page + 20)
        
        return "\n".join(lines[start:end])
    
    def _create_entity(self,
                       entity_data: Dict,
                       source_doc: str,
                       was_redacted: bool,
                       context: str,
                       page_num: int = None) -> Optional[ClassifiedEntity]:
        """Create a ClassifiedEntity from API response data"""
        try:
            name = entity_data.get("name", "").strip()
            if not name or name.lower() in ["unknown", "n/a", "none"]:
                return None
                
            entity_type_str = entity_data.get("type", "OTHER").upper()
            try:
                entity_type = EntityType[entity_type_str]
            except KeyError:
                entity_type = EntityType.OTHER
                
            return ClassifiedEntity(
                name=name,
                email=entity_data.get("email"),
                entity_type=entity_type,
                confidence=float(entity_data.get("confidence", 0.5)),
                reasoning=entity_data.get("reasoning", ""),
                source_document=source_doc,
                source_page=page_num,
                was_redacted=was_redacted,
                context=context[:500] if context else ""
            )
        except Exception:
            return None
    
    def _deduplicate_entities(self, entities: List[ClassifiedEntity]) -> List[ClassifiedEntity]:
        """Remove duplicate entities, keeping the one with highest confidence"""
        seen = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.email)
            
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
                
        return list(seen.values())


def classify_entities(pdf_doc: PDFDocument, 
                     unredaction_results: List[UnredactionResult],
                     api_key: str = None) -> List[ClassifiedEntity]:
    """Convenience function to classify entities in a document"""
    classifier = AIClassifier(api_key)
    return classifier.classify_document(pdf_doc, unredaction_results)

