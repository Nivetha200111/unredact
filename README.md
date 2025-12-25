# PDF Unredactor 🔍

A powerful Python tool that crawls websites to find PDFs, detects and attempts to recover redacted content, and uses AI to classify mentioned entities as **Politically Exposed Persons (PEPs)** or **Victims**.

## Features

- **🌐 DFS Web Crawler** - Crawls websites using depth-first search to find topic-related PDFs
- **📄 Redaction Detection** - Identifies multiple types of redactions:
  - Black box/rectangle redactions
  - Black highlight overlays
  - Image overlays covering text
  - White-out redactions
  - Pattern fills
- **🔓 Unredaction Techniques** - Attempts to recover hidden text using:
  - PDF layer analysis
  - Metadata extraction
  - OCR with image enhancement
  - Font analysis
  - Image processing (contrast, thresholding, edge detection)
- **🤖 AI Classification** - Uses GPT-4 to classify entities as:
  - **PEP** (Politically Exposed Person) - politicians, officials, executives
  - **VICTIM** - victims of fraud, abuse, or wrongdoing
- **📊 Organized Reports** - Generates comprehensive tables in Excel/CSV format

## Installation

### 1. Clone and setup

```bash
cd unredact
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install system dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Arch Linux:**
```bash
sudo pacman -S tesseract poppler
```

### 3. Configure API key

Copy the template and add your OpenAI API key:

```bash
cp env.template .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Crawl a website for PDFs

```bash
python main.py --url "https://example.com/documents" --topic "fraud investigation"
```

### Process local PDFs

```bash
python main.py --pdf-dir "./my_documents" --topic "financial scandal"
```

### Full options

```bash
python main.py \
  --url "https://site.com/archive" \
  --topic "corruption case" \
  --depth 5 \
  --excel \
  --csv
```

### Command-line options

| Option | Short | Description |
|--------|-------|-------------|
| `--url` | `-u` | Base URL to start crawling from |
| `--pdf-dir` | `-d` | Local directory containing PDFs |
| `--topic` | `-t` | Topic keywords for relevance filtering |
| `--depth` | | Maximum crawl depth (default: 3) |
| `--excel` | | Export to Excel (default: True) |
| `--csv` | | Export to CSV files |
| `--quiet` | `-q` | Don't print tables to console |
| `--no-excel` | | Skip Excel export |

## Output

The tool generates several output files in the `output/` directory:

### Excel Report (`unredact_report_TIMESTAMP.xlsx`)
Contains multiple sheets:
- **Summary** - Overview of all processed documents
- **Redactions** - Details of each detected redaction
- **Entities** - Classified entities (PEP/Victim)
- **Emails** - All email addresses found with classifications

### Console Tables
Formatted tables showing:
- Document processing summary
- Redaction detection and recovery results
- Entity classifications with confidence scores
- Email address inventory

## How It Works

### 1. Web Crawling (DFS)
```
Starting URL → Find links → Follow depth-first
                ↓
          Find PDF links
                ↓
          Check relevance to topic
                ↓
          Download relevant PDFs
```

### 2. Redaction Detection
```
PDF Page → Extract annotations → Check for redaction marks
    ↓
    → Extract drawings → Find filled rectangles
    ↓
    → Convert to image → Detect black regions (OpenCV)
    ↓
    → Find image overlays → Check for solid color covers
```

### 3. Unredaction Pipeline
```
Redaction Area → Try PDF layer extraction
      ↓
      → Try metadata extraction
      ↓
      → Try font/character analysis
      ↓
      → Try OCR with enhancements:
         • Contrast enhancement (CLAHE)
         • Multiple thresholds
         • Edge detection
         • Color inversion
         • Morphological operations
```

### 4. AI Classification
```
Extracted Text → GPT-4 Analysis
      ↓
Entity Extraction → Name, Email, Context
      ↓
Classification:
  • PEP: Politicians, officials, executives, judges
  • VICTIM: Plaintiffs, fraud targets, whistleblowers
  • OTHER: Neutral parties
```

## Example Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PROCESSING COMPLETE                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
 PDFs Found:            15
 PDFs Processed:        15
 Redactions Detected:   47
 Text Recovered:        12
 Entities Classified:   23
 PEPs Identified:       8
 Victims Identified:    5
```

## Project Structure

```
unredact/
├── main.py              # Main orchestration script
├── crawler.py           # DFS web crawler
├── pdf_processor.py     # PDF parsing and redaction detection
├── unredactor.py        # Redaction recovery techniques
├── ai_classifier.py     # GPT-4 entity classification
├── table_generator.py   # Report generation
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── env.template         # Environment template
├── downloads/           # Downloaded PDFs
├── output/              # Generated reports
└── temp/                # Temporary processing files
```

## Configuration

Edit `config.py` to customize:

```python
# Crawler settings
MAX_DEPTH = 5              # How deep to crawl
REQUEST_DELAY = 1.0        # Seconds between requests

# Redaction detection
BLACK_THRESHOLD = 30       # RGB threshold for "black"
MIN_REDACTION_AREA = 100   # Minimum pixel area

# OCR settings
OCR_DPI = 300              # Resolution for OCR
OCR_LANG = "eng"           # Tesseract language

# AI
OPENAI_MODEL = "gpt-4-turbo-preview"
```

## Limitations & Ethics

⚠️ **Important Considerations:**

1. **Legal**: Ensure you have permission to access and process the PDFs
2. **Privacy**: Handle recovered personal information responsibly
3. **Accuracy**: Unredaction is not always possible; results should be verified
4. **Rate Limits**: The tool respects website rate limits; don't abuse

Some redactions cannot be recovered:
- Properly applied redactions that permanently remove underlying data
- Scanned documents with no underlying text layer
- Heavily compressed or low-quality PDFs

## License

MIT License - See LICENSE file for details.
