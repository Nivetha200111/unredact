"""
Unredaction Module - Attempts to recover text from redacted areas
"""
import os
import io
from typing import Optional, List, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import pytesseract
from pdf2image import convert_from_path
import pikepdf

from pdf_processor import PDFDocument, RedactionArea, RedactionType
import config


@dataclass
class UnredactionResult:
    """Result of an unredaction attempt"""
    original_redaction: RedactionArea
    recovered_text: Optional[str]
    method_used: str
    confidence: float
    success: bool


class Unredactor:
    """
    Attempts to recover text from redacted areas using multiple techniques:
    1. PDF Layer Analysis - Check if text exists under redaction layers
    2. OCR Enhancement - Apply image processing to reveal faint text
    3. Metadata Extraction - Check PDF metadata and structure
    4. Font Analysis - Look for character outlines that may remain
    """
    
    def __init__(self):
        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    def _try_layer_removal(self, pdf_path: str, redaction: RedactionArea) -> Optional[str]:
        """
        Try to access text by removing overlay layers
        Some PDFs have redactions as separate layers over the actual text
        """
        try:
            with pikepdf.open(pdf_path) as pdf:
                page = pdf.pages[redaction.page_num]
                
                # Try to access content streams and find text
                if hasattr(page, 'Contents'):
                    # This is a simplified approach - full implementation would
                    # parse the PDF content stream to find text operators
                    pass
                    
            # Try with PyMuPDF - look for text in the redacted area
            doc = fitz.open(pdf_path)
            page = doc.load_page(redaction.page_num)
            
            # Get text with detailed position info
            text_dict = page.get_text("dict")
            recovered_chars = []
            
            x0, y0, x1, y1 = redaction.bbox
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            # Check if span intersects with redaction area
                            span_bbox = span.get("bbox", (0, 0, 0, 0))
                            if (span_bbox[0] < x1 and span_bbox[2] > x0 and
                                span_bbox[1] < y1 and span_bbox[3] > y0):
                                text = span.get("text", "")
                                if text.strip():
                                    recovered_chars.append(text)
            
            doc.close()
            
            if recovered_chars:
                return " ".join(recovered_chars)
                
        except Exception as e:
            pass
            
        return None
    
    def _try_ocr_enhancement(self, pdf_path: str, redaction: RedactionArea) -> Optional[str]:
        """
        Apply image enhancement and OCR to try to read under the redaction
        This works when redactions are not 100% opaque or have artifacts
        """
        try:
            # Convert PDF page to high-res image
            images = convert_from_path(
                pdf_path,
                dpi=config.OCR_DPI,
                first_page=redaction.page_num + 1,
                last_page=redaction.page_num + 1
            )
            
            if not images:
                return None
                
            page_image = np.array(images[0])
            
            # Get redaction coordinates scaled to image
            doc = fitz.open(pdf_path)
            page = doc.load_page(redaction.page_num)
            pdf_width = page.rect.width
            pdf_height = page.rect.height
            doc.close()
            
            img_height, img_width = page_image.shape[:2]
            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            
            x0, y0, x1, y1 = redaction.bbox
            ix0 = int(x0 * scale_x)
            iy0 = int(y0 * scale_y)
            ix1 = int(x1 * scale_x)
            iy1 = int(y1 * scale_y)
            
            # Add padding
            padding = 5
            ix0 = max(0, ix0 - padding)
            iy0 = max(0, iy0 - padding)
            ix1 = min(img_width, ix1 + padding)
            iy1 = min(img_height, iy1 + padding)
            
            # Extract the redaction region
            roi = page_image[iy0:iy1, ix0:ix1]
            
            if roi.size == 0:
                return None
            
            # Apply multiple enhancement techniques
            results = []
            
            # Technique 1: Simple inversion
            inverted = cv2.bitwise_not(roi)
            pil_inv = Image.fromarray(inverted)
            text = pytesseract.image_to_string(pil_inv, lang=config.OCR_LANG).strip()
            if text and len(text) > 2:
                results.append(("inversion", text))
            
            # Technique 2: Contrast enhancement
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            text = pytesseract.image_to_string(enhanced, lang=config.OCR_LANG).strip()
            if text and len(text) > 2:
                results.append(("clahe", text))
            
            # Technique 3: Thresholding variations
            for thresh_val in [50, 100, 127, 150, 200]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(thresh, lang=config.OCR_LANG).strip()
                if text and len(text) > 2:
                    results.append((f"thresh_{thresh_val}", text))
            
            # Technique 4: Edge detection to find text outlines
            edges = cv2.Canny(gray, 50, 150)
            text = pytesseract.image_to_string(edges, lang=config.OCR_LANG).strip()
            if text and len(text) > 2:
                results.append(("edges", text))
            
            # Technique 5: Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(enhanced, kernel, iterations=1)
            text = pytesseract.image_to_string(dilated, lang=config.OCR_LANG).strip()
            if text and len(text) > 2:
                results.append(("morph", text))
            
            # Return the longest/most common result
            if results:
                # Sort by length
                results.sort(key=lambda x: len(x[1]), reverse=True)
                return results[0][1]
                
        except Exception as e:
            pass
            
        return None
    
    def _try_metadata_extraction(self, pdf_path: str, redaction: RedactionArea) -> Optional[str]:
        """
        Check if redaction metadata contains original text
        Some poorly implemented redactions store original text
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(redaction.page_num)
            
            # Check annotations for stored text
            for annot in page.annots() or []:
                annot_rect = annot.rect
                x0, y0, x1, y1 = redaction.bbox
                
                # Check if annotation matches redaction location
                if (abs(annot_rect.x0 - x0) < 5 and abs(annot_rect.y0 - y0) < 5):
                    # Try to get various annotation properties
                    info = annot.info
                    if info:
                        for key in ["content", "subject", "title", "Contents"]:
                            if key in info and info[key]:
                                doc.close()
                                return str(info[key])
            
            doc.close()
            
        except Exception:
            pass
            
        return None
    
    def _try_font_analysis(self, pdf_path: str, redaction: RedactionArea) -> Optional[str]:
        """
        Analyze font information that might reveal text structure
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(redaction.page_num)
            
            # Get detailed text information
            blocks = page.get_text("rawdict")
            
            x0, y0, x1, y1 = redaction.bbox
            
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        line_bbox = line.get("bbox", (0, 0, 0, 0))
                        
                        # Check intersection
                        if (line_bbox[0] < x1 and line_bbox[2] > x0 and
                            line_bbox[1] < y1 and line_bbox[3] > y0):
                            
                            chars = []
                            for span in line.get("spans", []):
                                # Try to get character-level info
                                text = span.get("text", "")
                                if text:
                                    chars.append(text)
                            
                            if chars:
                                doc.close()
                                return "".join(chars)
            
            doc.close()
            
        except Exception:
            pass
            
        return None
    
    def _analyze_image_overlay(self, redaction: RedactionArea) -> Optional[str]:
        """
        Analyze image overlays to see if they contain readable patterns
        """
        if not redaction.image_data:
            return None
            
        try:
            img = Image.open(io.BytesIO(redaction.image_data))
            img_array = np.array(img)
            
            # Check if the image has any variation
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                
            std_dev = np.std(gray)
            
            # If there's some variation, try OCR
            if std_dev > 5:
                # Enhance and OCR
                enhanced = cv2.equalizeHist(gray)
                text = pytesseract.image_to_string(enhanced, lang=config.OCR_LANG).strip()
                if text:
                    return text
                    
        except Exception:
            pass
            
        return None
    
    def unredact(self, pdf_doc: PDFDocument) -> List[UnredactionResult]:
        """
        Attempt to unredact all detected redactions in a PDF document
        """
        results = []
        
        print(f"  Attempting unredaction of {len(pdf_doc.redactions)} redactions...")
        
        for redaction in pdf_doc.redactions:
            recovered_text = None
            method_used = "none"
            confidence = 0.0
            
            # If we already have underlying text from PDF structure
            if redaction.underlying_text:
                recovered_text = redaction.underlying_text
                method_used = "pdf_structure"
                confidence = 0.9
            else:
                # Try different unredaction methods
                
                # Method 1: Layer removal
                text = self._try_layer_removal(pdf_doc.filepath, redaction)
                if text:
                    recovered_text = text
                    method_used = "layer_removal"
                    confidence = 0.8
                
                # Method 2: Metadata extraction
                if not recovered_text:
                    text = self._try_metadata_extraction(pdf_doc.filepath, redaction)
                    if text:
                        recovered_text = text
                        method_used = "metadata"
                        confidence = 0.85
                
                # Method 3: Font analysis
                if not recovered_text:
                    text = self._try_font_analysis(pdf_doc.filepath, redaction)
                    if text:
                        recovered_text = text
                        method_used = "font_analysis"
                        confidence = 0.7
                
                # Method 4: Image overlay analysis
                if not recovered_text and redaction.redaction_type == RedactionType.IMAGE_OVERLAY:
                    text = self._analyze_image_overlay(redaction)
                    if text:
                        recovered_text = text
                        method_used = "image_analysis"
                        confidence = 0.6
                
                # Method 5: OCR enhancement (slower, try last)
                if not recovered_text:
                    text = self._try_ocr_enhancement(pdf_doc.filepath, redaction)
                    if text:
                        recovered_text = text
                        method_used = "ocr_enhancement"
                        confidence = 0.5
            
            # Store the recovered text in the redaction object
            redaction.extracted_text = recovered_text
            
            result = UnredactionResult(
                original_redaction=redaction,
                recovered_text=recovered_text,
                method_used=method_used,
                confidence=confidence,
                success=recovered_text is not None
            )
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        print(f"    Successfully recovered text from {successful}/{len(results)} redactions")
        
        return results


def unredact_document(pdf_doc: PDFDocument) -> List[UnredactionResult]:
    """Convenience function to unredact a processed PDF document"""
    unredactor = Unredactor()
    return unredactor.unredact(pdf_doc)

