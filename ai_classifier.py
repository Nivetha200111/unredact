"""
AI Classification Module - Uses local HuggingFace model to classify entities as PEP or Victim
"""
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
    Uses local HuggingFace model to analyze text and classify mentioned entities
    as either PEPs (Politically Exposed Persons) or victims
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.HF_MODEL
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model with memory optimizations"""
        print(f"  Loading model: {self.model_name}")
        print(f"  This may take a few minutes on first run...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure 4-bit quantization to save memory
        if config.USE_4BIT_QUANTIZATION:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=config.DEVICE_MAP,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=config.DEVICE_MAP,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        print(f"  Model loaded successfully!")
    
    def _extract_entities_prompt(self, text: str, context: str = "") -> str:
        """Create prompt for entity extraction"""
        return f"""Analyze the following text and extract all person names and email addresses.
For each person found, classify them as:
- PEP: Politicians, government officials, executives, judges, military officers
- VICTIM: Victims of fraud, abuse, crime, or wrongdoing
- OTHER: Neither of the above

Context: {context}

Text:
{text[:2000]}

Respond ONLY with JSON:
{{"entities": [{{"name": "Full Name", "email": "email or null", "type": "PEP|VICTIM|OTHER", "confidence": 0.0-1.0, "reasoning": "brief reason"}}]}}

JSON:"""
    
    def _classify_redacted_content_prompt(self, redacted_text: str, surrounding_context: str, document_context: str) -> str:
        """Create prompt for classifying redacted content"""
        return f"""This text was REDACTED from a document but recovered. Analyze it.

Document: {document_context}
Surrounding context: {surrounding_context[:500]}

RECOVERED TEXT: {redacted_text}

For each person/email found, classify as PEP (official/executive) or VICTIM (wrongdoing target).

Respond ONLY with JSON:
{{"entities": [{{"name": "Name", "email": "email or null", "type": "PEP|VICTIM|OTHER", "confidence": 0.0-1.0, "reasoning": "why classified and why redacted"}}]}}

JSON:"""
    
    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from the model"""
        try:
            # Check if model supports chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            return response.strip()
            
        except Exception as e:
            print(f"    Generation error: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from model response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return None
    
    def classify_document(self, pdf_doc: PDFDocument, unredaction_results: List[UnredactionResult]) -> List[ClassifiedEntity]:
        """Classify all entities found in a document"""
        classified_entities = []
        
        print(f"  Classifying entities in {pdf_doc.filename}...")
        
        document_context = f"Document: {pdf_doc.filename}, Pages: {pdf_doc.num_pages}"
        
        # Split text into chunks
        text_chunks = self._split_text(pdf_doc.total_text, chunk_size=1500)
        
        for i, chunk in enumerate(text_chunks[:5]):  # Limit chunks to save time
            prompt = self._extract_entities_prompt(chunk, document_context)
            response = self._generate(prompt)
            result = self._parse_json_response(response)
            
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
        
        # Analyze recovered redacted content
        for unredact_result in unredaction_results:
            if unredact_result.success and unredact_result.recovered_text:
                surrounding = self._get_surrounding_context(
                    pdf_doc.total_text,
                    unredact_result.original_redaction.page_num
                )
                
                prompt = self._classify_redacted_content_prompt(
                    unredact_result.recovered_text,
                    surrounding,
                    document_context
                )
                response = self._generate(prompt)
                result = self._parse_json_response(response)
                
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
        
        # Add unclassified emails
        for email in pdf_doc.emails_found:
            if not any(e.email == email for e in classified_entities):
                classified_entities.append(ClassifiedEntity(
                    name="Unknown",
                    email=email,
                    entity_type=EntityType.UNKNOWN,
                    confidence=0.5,
                    reasoning="Email found but entity not classified",
                    source_document=pdf_doc.filename,
                    source_page=None,
                    was_redacted=False,
                    context=""
                ))
        
        # Deduplicate
        classified_entities = self._deduplicate_entities(classified_entities)
        
        pep_count = sum(1 for e in classified_entities if e.entity_type == EntityType.PEP)
        victim_count = sum(1 for e in classified_entities if e.entity_type == EntityType.VICTIM)
        print(f"    Found {len(classified_entities)} entities: {pep_count} PEPs, {victim_count} victims")
        
        return classified_entities
    
    def _split_text(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Split text into chunks"""
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
        lines = full_text.split('\n')
        lines_per_page = max(1, len(lines) // 10)
        
        start = max(0, page_num * lines_per_page - 20)
        end = min(len(lines), (page_num + 1) * lines_per_page + 20)
        
        return "\n".join(lines[start:end])
    
    def _create_entity(self, entity_data: Dict, source_doc: str, was_redacted: bool,
                       context: str, page_num: int = None) -> Optional[ClassifiedEntity]:
        """Create a ClassifiedEntity from response data"""
        try:
            name = entity_data.get("name", "").strip()
            if not name or name.lower() in ["unknown", "n/a", "none", ""]:
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
        """Remove duplicate entities"""
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.email)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())


def classify_entities(pdf_doc: PDFDocument, unredaction_results: List[UnredactionResult],
                     model_name: str = None) -> List[ClassifiedEntity]:
    """Convenience function to classify entities"""
    classifier = AIClassifier(model_name)
    return classifier.classify_document(pdf_doc, unredaction_results)
