"""
Configuration settings for the PDF Unredactor
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

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
OCR_DPI = 300
OCR_LANG = "eng"

# Classification Categories
CLASSIFICATION_CATEGORIES = ["PEP", "VICTIM", "UNKNOWN", "OTHER"]

