"""
Configuration settings for the PDF Unredactor
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Local LLM Configuration (Hugging Face)
# Options: "openai/gpt-oss-120b" (needs 240GB+ RAM)
#          "mistralai/Mistral-7B-Instruct-v0.2" (needs ~16GB RAM)
#          "microsoft/phi-2" (needs ~6GB RAM - recommended for laptops)
HF_MODEL = os.getenv("HF_MODEL", "microsoft/phi-2")
USE_4BIT_QUANTIZATION = True  # Reduces memory by 4x
DEVICE_MAP = "auto"  # Automatically distribute across GPU/CPU

# Crawler Configuration
MAX_DEPTH = 5
REQUEST_DELAY = 1.0  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# PDF Processing
DOWNLOADS_DIR = "downloads"
OUTPUT_DIR = "output"
TEMP_DIR = "temp"

# Redaction Detection Thresholds
BLACK_THRESHOLD = 30  # RGB values below this are considered "black"
MIN_REDACTION_AREA = 100  # minimum pixels for a redaction box
REDACTION_ASPECT_RATIO_MAX = 50  # max width/height ratio for redactions

# OCR Settings
TESSERACT_CMD = None  # Set path if not in system PATH
OCR_DPI = 150  # Lower DPI for faster processing (was 300)
OCR_LANG = "eng"

# Performance Settings
SKIP_OCR_ENHANCEMENT = False  # Set True to skip slow OCR text recovery (not detection)
SKIP_IMAGE_DETECTION = False  # Never skip - required for scanned PDFs
MAX_PAGES_TO_PROCESS = None  # Limit pages per PDF (None = all)

# Classification Categories
CLASSIFICATION_CATEGORIES = ["PEP", "VICTIM", "UNKNOWN", "OTHER"]

