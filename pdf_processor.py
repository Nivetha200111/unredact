"""
PDF Processing and Redaction Detection Module
"""
import os
import io
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import cv2
from pdf2image import convert_from_path

import config


class RedactionType(Enum):
    """Types of redactions that can be detected"""
    BLACK_BOX = "black_box"           # Solid black rectangle
    BLACK_HIGHLIGHT = "black_highlight"  # Black highlight over text
    IMAGE_OVERLAY = "image_overlay"     # Image placed over text
    WHITE_BOX = "white_box"            # White rectangle hiding text
    PATTERN_FILL = "pattern_fill"       # Patterned fill over text
    UNKNOWN = "unknown"


@dataclass
class RedactionArea:
    """Represents a detected redaction area"""
    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    redaction_type: RedactionType
    confidence: float
    underlying_text: Optional[str] = None
    extracted_text: Optional[str] = None
    image_data: Optional[bytes] = None


@dataclass
class PDFDocument:
    """Represents a processed PDF document"""
    filepath: str
    filename: str
    num_pages: int
    total_text: str
    redactions: List[RedactionArea] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    emails_found: List[str] = field(default_factory=list)


class PDFProcessor:
    """Processes PDFs to detect and catalog redactions"""
    
    def __init__(self):
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        
    def _detect_black_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect black rectangular regions in an image"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Threshold to find very dark regions
        _, thresh = cv2.threshold(gray, config.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        black_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.MIN_REDACTION_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio - redactions are usually wider than tall
            aspect = w / max(h, 1)
            if aspect > config.REDACTION_ASPECT_RATIO_MAX:
                continue
                
            # Check if region is mostly black (not just an edge)
            roi = gray[y:y+h, x:x+w]
            black_ratio = np.sum(roi < config.BLACK_THRESHOLD) / (w * h)
            
            if black_ratio > 0.7:  # 70% of region is black
                black_regions.append((x, y, x + w, y + h))
                
        return black_regions
    
    def _detect_image_overlays(self, page: fitz.Page) -> List[Tuple[fitz.Rect, bytes]]:
        """Detect images that might be overlaying text"""
        image_overlays = []
        
        # Get all images on the page
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            try:
                # Get image data
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image position
                for img_rect in page.get_image_rects(xref):
                    # Check if image is small enough to be a redaction cover
                    width = img_rect.width
                    height = img_rect.height
                    
                    # Redaction overlays are usually small rectangles
                    if 10 < width < 500 and 5 < height < 100:
                        # Check if mostly solid color
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        img_array = np.array(pil_img)
                        
                        if len(img_array.shape) == 3:
                            std_dev = np.std(img_array)
                        else:
                            std_dev = np.std(img_array)
                            
                        # Low std dev means solid color
                        if std_dev < 30:
                            image_overlays.append((img_rect, image_bytes))
                            
            except Exception:
                continue
                
        return image_overlays
    
    def _detect_annotation_redactions(self, page: fitz.Page) -> List[RedactionArea]:
        """Detect redactions from PDF annotations"""
        redactions = []
        
        for annot in page.annots() or []:
            annot_type = annot.type[0]
            
            # Check for redaction annotations
            if annot_type == 12:  # Redact annotation
                redactions.append(RedactionArea(
                    page_num=page.number,
                    bbox=tuple(annot.rect),
                    redaction_type=RedactionType.BLACK_BOX,
                    confidence=0.95
                ))
            # Check for highlight annotations with dark color
            elif annot_type == 8:  # Highlight
                colors = annot.colors
                if colors and colors.get("stroke"):
                    r, g, b = colors["stroke"][:3] if len(colors["stroke"]) >= 3 else (1, 1, 1)
                    # Dark highlight
                    if r < 0.2 and g < 0.2 and b < 0.2:
                        redactions.append(RedactionArea(
                            page_num=page.number,
                            bbox=tuple(annot.rect),
                            redaction_type=RedactionType.BLACK_HIGHLIGHT,
                            confidence=0.85
                        ))
            # Check for rectangle annotations
            elif annot_type == 4:  # Square
                colors = annot.colors
                if colors:
                    fill = colors.get("fill", (1, 1, 1))
                    if fill and len(fill) >= 3:
                        r, g, b = fill[:3]
                        if r < 0.2 and g < 0.2 and b < 0.2:
                            redactions.append(RedactionArea(
                                page_num=page.number,
                                bbox=tuple(annot.rect),
                                redaction_type=RedactionType.BLACK_BOX,
                                confidence=0.9
                            ))
                            
        return redactions
    
    def _detect_drawing_redactions(self, page: fitz.Page) -> List[RedactionArea]:
        """Detect redactions from PDF drawing commands (filled rectangles)"""
        redactions = []
        
        # Get page drawings
        drawings = page.get_drawings()
        
        for path in drawings:
            # Check for filled rectangles
            if path.get("fill") is not None:
                fill_color = path["fill"]
                
                # Check if it's a dark fill
                if fill_color and len(fill_color) >= 3:
                    r, g, b = fill_color[:3]
                    is_dark = (r < 0.1 and g < 0.1 and b < 0.1)
                    is_white = (r > 0.95 and g > 0.95 and b > 0.95)
                    
                    if is_dark or is_white:
                        rect = path.get("rect")
                        if rect:
                            width = rect.width
                            height = rect.height
                            
                            # Filter for reasonable redaction sizes
                            if 10 < width < 600 and 3 < height < 100:
                                redaction_type = RedactionType.BLACK_BOX if is_dark else RedactionType.WHITE_BOX
                                redactions.append(RedactionArea(
                                    page_num=page.number,
                                    bbox=tuple(rect),
                                    redaction_type=redaction_type,
                                    confidence=0.8
                                ))
                                
        return redactions
    
    def _extract_text_under_redaction(self, page: fitz.Page, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Attempt to extract text that might be under a redaction"""
        try:
            rect = fitz.Rect(bbox)
            # Try to get text from the redacted area
            text = page.get_text("text", clip=rect)
            if text.strip():
                return text.strip()
        except Exception:
            pass
        return None
    
    def _find_emails_in_text(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        import re
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(email_pattern, text)))
    
    def process_pdf(self, filepath: str) -> PDFDocument:
        """Process a PDF and detect all redactions"""
        doc = fitz.open(filepath)
        filename = os.path.basename(filepath)
        
        all_redactions = []
        full_text = []
        
        print(f"  Processing {filename} ({doc.page_count} pages)...")
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # Extract text
            page_text = page.get_text("text")
            full_text.append(page_text)
            
            # Detect annotation-based redactions
            annot_redactions = self._detect_annotation_redactions(page)
            all_redactions.extend(annot_redactions)
            
            # Detect drawing-based redactions
            drawing_redactions = self._detect_drawing_redactions(page)
            all_redactions.extend(drawing_redactions)
            
            # Detect image overlays
            image_overlays = self._detect_image_overlays(page)
            for rect, img_data in image_overlays:
                all_redactions.append(RedactionArea(
                    page_num=page_num,
                    bbox=tuple(rect),
                    redaction_type=RedactionType.IMAGE_OVERLAY,
                    confidence=0.75,
                    image_data=img_data
                ))
            
            # Try to extract underlying text for each redaction
            for redaction in all_redactions:
                if redaction.page_num == page_num and redaction.underlying_text is None:
                    redaction.underlying_text = self._extract_text_under_redaction(page, redaction.bbox)
        
        # Convert to images and detect black regions (skip in fast mode - very slow/memory intensive)
        if not config.SKIP_OCR_ENHANCEMENT:
            try:
                images = convert_from_path(filepath, dpi=config.OCR_DPI)
                for page_num, pil_image in enumerate(images):
                    img_array = np.array(pil_image)
                    black_regions = self._detect_black_regions(img_array)
                    
                    # Scale coordinates back to PDF space
                    page = doc.load_page(page_num)
                    pdf_width = page.rect.width
                    pdf_height = page.rect.height
                    img_height, img_width = img_array.shape[:2]
                    
                    scale_x = pdf_width / img_width
                    scale_y = pdf_height / img_height
                    
                    for x0, y0, x1, y1 in black_regions:
                        # Check if this region overlaps with existing redactions
                        pdf_bbox = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
                        
                        is_duplicate = False
                        for existing in all_redactions:
                            if existing.page_num == page_num:
                                # Check overlap
                                ex0, ey0, ex1, ey1 = existing.bbox
                                if (pdf_bbox[0] < ex1 and pdf_bbox[2] > ex0 and
                                    pdf_bbox[1] < ey1 and pdf_bbox[3] > ey0):
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_redactions.append(RedactionArea(
                                page_num=page_num,
                                bbox=pdf_bbox,
                                redaction_type=RedactionType.BLACK_BOX,
                                confidence=0.7
                            ))
                    # Free memory after each page
                    del img_array
                    del pil_image
                del images
            except Exception as e:
                print(f"    Warning: Could not process images for {filename}: {e}")
        
        doc.close()
        
        combined_text = "\n".join(full_text)
        emails = self._find_emails_in_text(combined_text)
        
        pdf_doc = PDFDocument(
            filepath=filepath,
            filename=filename,
            num_pages=len(full_text),
            total_text=combined_text,
            redactions=all_redactions,
            emails_found=emails
        )
        
        print(f"    Found {len(all_redactions)} potential redactions, {len(emails)} emails")
        
        return pdf_doc


def process_pdf_file(filepath: str) -> PDFDocument:
    """Convenience function to process a single PDF"""
    processor = PDFProcessor()
    return processor.process_pdf(filepath)

