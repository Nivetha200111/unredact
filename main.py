#!/usr/bin/env python3
"""
PDF Unredactor - Main Orchestration Script

This script:
1. Crawls a website using DFS to find topic-related PDFs
2. Downloads and processes PDFs to detect redactions
3. Attempts to recover text from redacted areas
4. Uses GPT-4 to classify entities as PEP (Politically Exposed Person) or Victim
5. Generates organized reports of findings

Usage:
    python main.py --url "https://example.com" --topic "investigation fraud"
    python main.py --pdf-dir "./my_pdfs" --topic "financial scandal"
"""

import argparse
import os
import sys
import zipfile
import tempfile
import shutil
from typing import List, Dict
from pathlib import Path

from colorama import init, Fore, Style
from tqdm import tqdm

# Initialize colorama for cross-platform colored output
init()

# Import local modules
from crawler import PDFInfo
from pdf_processor import PDFProcessor, PDFDocument
from unredactor import Unredactor, UnredactionResult
from ai_classifier import AIClassifier, ClassifiedEntity
from table_generator import TableGenerator
import config


def print_banner():
    """Print the application banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██████╗ ███████╗    ██╗   ██╗███╗   ██╗██████╗ ███████╗██████╗     ║
║   ██╔══██╗██╔══██╗██╔════╝    ██║   ██║████╗  ██║██╔══██╗██╔════╝██╔══██╗    ║
║   ██████╔╝██║  ██║█████╗      ██║   ██║██╔██╗ ██║██████╔╝█████╗  ██║  ██║    ║
║   ██╔═══╝ ██║  ██║██╔══╝      ██║   ██║██║╚██╗██║██╔══██╗██╔══╝  ██║  ██║    ║
║   ██║     ██████╔╝██║         ╚██████╔╝██║ ╚████║██║  ██║███████╗██████╔╝    ║
║   ╚═╝     ╚═════╝ ╚═╝          ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═════╝     ║
║                                                                              ║
║   PDF Redaction Recovery & Entity Classification Tool                        ║
║   Detects redactions • Recovers hidden text • Classifies PEPs & Victims     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def extract_zip_file(zip_path: str) -> str:
    """Extract a zip file and return the path to extracted contents"""
    zip_path = Path(zip_path).expanduser().resolve()
    
    if not zip_path.exists():
        print(f"{Fore.RED}Error: File not found: {zip_path}{Style.RESET_ALL}")
        return None
        
    if not zipfile.is_zipfile(zip_path):
        print(f"{Fore.RED}Error: Not a valid zip file: {zip_path}{Style.RESET_ALL}")
        return None
    
    # Extract to a subdirectory in downloads
    extract_dir = Path(config.DOWNLOADS_DIR) / f"extracted_{zip_path.stem}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{Fore.CYAN}Extracting zip file...{Style.RESET_ALL}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files
        file_list = zip_ref.namelist()
        pdf_count = sum(1 for f in file_list if f.lower().endswith('.pdf'))
        
        print(f"  Found {len(file_list)} files ({pdf_count} PDFs)")
        
        # Extract with progress
        for file in tqdm(file_list, desc="  Extracting"):
            zip_ref.extract(file, extract_dir)
    
    print(f"{Fore.GREEN}✓ Extracted to: {extract_dir}{Style.RESET_ALL}")
    return str(extract_dir)


def interactive_prompt() -> dict:
    """Prompt user for zip file path interactively"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" PDF Unredactor - Interactive Mode")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    args = argparse.Namespace()
    args.url = None
    args.pdf_dir = None
    args.zip_file = None
    args.depth = 3
    
    # Ask for zip file path
    while True:
        zip_path = input(f"{Fore.CYAN}Enter path to zip file containing PDFs: {Style.RESET_ALL}").strip()
        zip_path = os.path.expanduser(zip_path)
        if os.path.exists(zip_path):
            args.zip_file = zip_path
            break
        print(f"{Fore.RED}File not found. Please enter a valid path.{Style.RESET_ALL}")
    
    # Ask for output directory (for hard drive storage)
    print(f"\n{Fore.YELLOW}Where should output be saved? (Leave empty for current directory){Style.RESET_ALL}")
    output_path = input(f"{Fore.CYAN}Output directory path: {Style.RESET_ALL}").strip()
    if output_path:
        output_path = os.path.expanduser(output_path)
        os.makedirs(output_path, exist_ok=True)
        args.output_dir = output_path
    else:
        args.output_dir = None
    
    # Ask for topic
    args.topic = input(f"\n{Fore.CYAN}Enter topic/keywords to search for (or press Enter for all): {Style.RESET_ALL}").strip()
    if not args.topic:
        args.topic = "document"  # Default - process all
    
    # Output options
    args.excel = True
    args.csv = False
    args.quiet = False
    
    csv_choice = input(f"\n{Fore.CYAN}Also export to CSV? (y/N): {Style.RESET_ALL}").strip().lower()
    args.csv = csv_choice in ['y', 'yes']
    
    # Ask about fast mode
    fast_choice = input(f"\n{Fore.CYAN}Use fast mode? (skips slow OCR, recommended for large files) (Y/n): {Style.RESET_ALL}").strip().lower()
    args.fast = fast_choice not in ['n', 'no']
    
    return args


def validate_environment() -> bool:
    """Validate that required tools and configurations are available"""
    issues = []
    
    # Check Gemini API key
    if not config.GEMINI_API_KEY:
        issues.append("GEMINI_API_KEY not set. Create a .env file with your API key.")
    
    # Check tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        issues.append("Tesseract OCR not found. Install with: sudo apt install tesseract-ocr")
    
    # Check poppler (for pdf2image)
    try:
        from pdf2image import convert_from_path
    except Exception:
        issues.append("Poppler not found. Install with: sudo apt install poppler-utils")
    
    if issues:
        print(f"\n{Fore.YELLOW}⚠ Configuration Issues:{Style.RESET_ALL}")
        for issue in issues:
            print(f"  • {issue}")
        
        if "GEMINI_API_KEY" in str(issues):
            print(f"\n{Fore.RED}Cannot proceed without Gemini API key.{Style.RESET_ALL}")
            return False
    
    return True


def process_pdfs_from_directory(pdf_dir: str, topic: str) -> List[PDFInfo]:
    """Load PDFs from a local directory"""
    pdf_files = []
    pdf_dir_path = Path(pdf_dir)
    
    if not pdf_dir_path.exists():
        print(f"{Fore.RED}Error: Directory {pdf_dir} does not exist.{Style.RESET_ALL}")
        return []
    
    for pdf_path in pdf_dir_path.glob("**/*.pdf"):
        # Calculate basic relevance based on filename
        filename = pdf_path.name.lower()
        topic_words = topic.lower().split()
        relevance = sum(1 for word in topic_words if word in filename) / max(len(topic_words), 1)
        
        pdf_files.append(PDFInfo(
            url=str(pdf_path),
            filename=pdf_path.name,
            local_path=str(pdf_path),
            source_page="local",
            topic_relevance=max(relevance, 0.5)  # Give local files minimum 50% relevance
        ))
    
    print(f"\n{Fore.GREEN}Found {len(pdf_files)} PDFs in {pdf_dir}{Style.RESET_ALL}")
    return pdf_files


def run_pipeline(args) -> Dict:
    """Run the complete unredaction pipeline"""
    import gc  # For memory cleanup
    
    results = {
        "pdfs_found": 0,
        "pdfs_processed": 0,
        "redactions_found": 0,
        "redactions_recovered": 0,
        "entities_classified": 0,
        "peps_found": 0,
        "victims_found": 0
    }
    
    # Set custom output directory if provided
    if hasattr(args, 'output_dir') and args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        config.DOWNLOADS_DIR = os.path.join(args.output_dir, "downloads")
        config.TEMP_DIR = os.path.join(args.output_dir, "temp")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.DOWNLOADS_DIR, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        print(f"\n{Fore.GREEN}Output will be saved to: {config.OUTPUT_DIR}{Style.RESET_ALL}")
    
    # Enable fast mode if requested
    if hasattr(args, 'fast') and args.fast:
        config.SKIP_OCR_ENHANCEMENT = True
        config.OCR_DPI = 100
        print(f"{Fore.YELLOW}Fast mode enabled - skipping slow OCR processing{Style.RESET_ALL}")
    
    # Step 1: Get PDFs
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" STEP 1: Extract & Load PDFs")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # Handle zip file extraction
    pdf_dir = args.pdf_dir
    if hasattr(args, 'zip_file') and args.zip_file:
        pdf_dir = extract_zip_file(args.zip_file)
        if not pdf_dir:
            return results
    
    if pdf_dir:
        pdf_infos = process_pdfs_from_directory(pdf_dir, args.topic)
    else:
        print(f"{Fore.RED}Error: Must provide --zip or --dir{Style.RESET_ALL}")
        return results
    
    if not pdf_infos:
        print(f"{Fore.YELLOW}No PDFs found matching the topic.{Style.RESET_ALL}")
        return results
    
    results["pdfs_found"] = len(pdf_infos)
    
    # Process PDFs one at a time to save memory
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" Processing PDFs (memory-optimized, one at a time)")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    processor = PDFProcessor()
    unredactor = Unredactor()
    classifier = AIClassifier() if config.GEMINI_API_KEY else None
    
    # Store minimal info for final report
    pdf_docs: List[PDFDocument] = []
    all_unredaction_results: Dict[str, List[UnredactionResult]] = {}
    all_entities: List[ClassifiedEntity] = []
    
    total_pdfs = len(pdf_infos)
    
    for idx, pdf_info in enumerate(pdf_infos):
        print(f"\n{Fore.YELLOW}[{idx+1}/{total_pdfs}] Processing: {pdf_info.filename}{Style.RESET_ALL}")
        
        try:
            # Step 1: Process PDF
            pdf_doc = processor.process_pdf(pdf_info.local_path)
            results["redactions_found"] += len(pdf_doc.redactions)
            
            # Step 2: Unredact
            if pdf_doc.redactions:
                unredact_results = unredactor.unredact(pdf_doc)
                all_unredaction_results[pdf_doc.filepath] = unredact_results
                results["redactions_recovered"] += sum(1 for r in unredact_results if r.success)
            else:
                unredact_results = []
                all_unredaction_results[pdf_doc.filepath] = []
            
            # Step 3: Classify entities
            if classifier:
                try:
                    entities = classifier.classify_document(pdf_doc, unredact_results)
                    all_entities.extend(entities)
                    results["entities_classified"] += len(entities)
                    results["peps_found"] += sum(1 for e in entities if e.entity_type.value == "PEP")
                    results["victims_found"] += sum(1 for e in entities if e.entity_type.value == "VICTIM")
                except Exception as e:
                    print(f"{Fore.RED}  Classification error: {e}{Style.RESET_ALL}")
            
            # Keep minimal doc info for report
            pdf_docs.append(pdf_doc)
            results["pdfs_processed"] += 1
            
            # Free memory
            gc.collect()
            
        except Exception as e:
            print(f"{Fore.RED}  Error: {e}{Style.RESET_ALL}")
            continue
    
    if not pdf_docs:
        print(f"{Fore.YELLOW}No PDFs could be processed.{Style.RESET_ALL}")
        return results
    
    if not classifier:
        print(f"{Fore.YELLOW}Skipping AI classification (no API key){Style.RESET_ALL}")
    
    # Step 5: Generate Reports
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" STEP 5: Report Generation")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    generator = TableGenerator()
    report = generator.generate_report(
        pdf_docs,
        all_unredaction_results,
        all_entities,
        export_excel=args.excel,
        export_csv=args.csv,
        print_tables=not args.quiet
    )
    
    # Final Summary
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f" PDFs Found:            {results['pdfs_found']}")
    print(f" PDFs Processed:        {results['pdfs_processed']}")
    print(f" Redactions Detected:   {results['redactions_found']}")
    print(f" Text Recovered:        {results['redactions_recovered']}")
    print(f" Entities Classified:   {results['entities_classified']}")
    print(f" PEPs Identified:       {results['peps_found']}")
    print(f" Victims Identified:    {results['victims_found']}")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PDF Unredactor - Extract PDFs from zip, detect redactions, and classify entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (prompts for zip file path):
    python main.py
  
  Extract and process PDFs from a zip file:
    python main.py --zip "~/documents.zip" --topic "fraud investigation"
  
  Process local directory:
    python main.py --dir "./documents" --topic "financial scandal"

Environment:
  Create a .env file with:
    GEMINI_API_KEY=your-api-key-here
  Get your key at: https://aistudio.google.com/apikey
        """
    )
    
    # Input source - not required, will use interactive mode
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--zip", "-z",
        dest="zip_file",
        help="Zip file containing PDFs to extract and process"
    )
    source_group.add_argument(
        "--dir", "-d",
        dest="pdf_dir",
        help="Local directory containing PDFs to process"
    )
    
    # Topic
    parser.add_argument(
        "--topic", "-t",
        required=False,
        default="document",
        help="Topic to search for (used for relevance filtering)"
    )
    
    # Output directory (for storing on external drive)
    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        help="Output directory path (e.g., /mnt/harddrive/unredact_output)"
    )
    
    # Fast mode - skip slow OCR processing
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Fast mode - skip slow OCR enhancement (uses less memory/CPU)"
    )
    
    # Output options
    parser.add_argument(
        "--excel",
        action="store_true",
        default=True,
        help="Export results to Excel (default: True)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV files"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Don't print tables to console"
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Don't export to Excel"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check if we need interactive mode (no source provided)
    if not args.pdf_dir and not args.zip_file:
        args = interactive_prompt()
    
    # Ensure we have required attributes
    if not hasattr(args, 'url'):
        args.url = None
    if not hasattr(args, 'depth'):
        args.depth = 3
    
    # Handle --no-excel flag
    if hasattr(args, 'no_excel') and args.no_excel:
        args.excel = False
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Run the pipeline
    try:
        results = run_pipeline(args)
        
        if results["pdfs_processed"] == 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user.{Style.RESET_ALL}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

