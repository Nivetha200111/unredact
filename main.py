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
from typing import List, Dict
from pathlib import Path

from colorama import init, Fore, Style
from tqdm import tqdm

# Initialize colorama for cross-platform colored output
init()

# Import local modules
from crawler import crawl_for_pdfs, PDFInfo
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


def validate_environment() -> bool:
    """Validate that required tools and configurations are available"""
    issues = []
    
    # Check OpenAI API key
    if not config.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not set. Create a .env file with your API key.")
    
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
        
        if "OPENAI_API_KEY" in str(issues):
            print(f"\n{Fore.RED}Cannot proceed without OpenAI API key.{Style.RESET_ALL}")
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
    results = {
        "pdfs_found": 0,
        "pdfs_processed": 0,
        "redactions_found": 0,
        "redactions_recovered": 0,
        "entities_classified": 0,
        "peps_found": 0,
        "victims_found": 0
    }
    
    # Step 1: Get PDFs
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" STEP 1: PDF Discovery")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    if args.url:
        pdf_infos = crawl_for_pdfs(args.url, args.topic, args.depth)
    elif args.pdf_dir:
        pdf_infos = process_pdfs_from_directory(args.pdf_dir, args.topic)
    else:
        print(f"{Fore.RED}Error: Must provide either --url or --pdf-dir{Style.RESET_ALL}")
        return results
    
    if not pdf_infos:
        print(f"{Fore.YELLOW}No PDFs found matching the topic.{Style.RESET_ALL}")
        return results
    
    results["pdfs_found"] = len(pdf_infos)
    
    # Step 2: Process PDFs and detect redactions
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" STEP 2: PDF Processing & Redaction Detection")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    processor = PDFProcessor()
    pdf_docs: List[PDFDocument] = []
    
    for pdf_info in tqdm(pdf_infos, desc="Processing PDFs"):
        try:
            pdf_doc = processor.process_pdf(pdf_info.local_path)
            pdf_docs.append(pdf_doc)
            results["redactions_found"] += len(pdf_doc.redactions)
        except Exception as e:
            print(f"\n{Fore.RED}  Error processing {pdf_info.filename}: {e}{Style.RESET_ALL}")
    
    results["pdfs_processed"] = len(pdf_docs)
    
    if not pdf_docs:
        print(f"{Fore.YELLOW}No PDFs could be processed.{Style.RESET_ALL}")
        return results
    
    # Step 3: Attempt unredaction
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" STEP 3: Unredaction Attempts")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    unredactor = Unredactor()
    all_unredaction_results: Dict[str, List[UnredactionResult]] = {}
    
    for pdf_doc in pdf_docs:
        if pdf_doc.redactions:
            try:
                unredact_results = unredactor.unredact(pdf_doc)
                all_unredaction_results[pdf_doc.filepath] = unredact_results
                results["redactions_recovered"] += sum(1 for r in unredact_results if r.success)
            except Exception as e:
                print(f"\n{Fore.RED}  Error unredacting {pdf_doc.filename}: {e}{Style.RESET_ALL}")
                all_unredaction_results[pdf_doc.filepath] = []
        else:
            all_unredaction_results[pdf_doc.filepath] = []
    
    # Step 4: AI Classification
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f" STEP 4: AI Entity Classification (PEP/Victim)")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    all_entities: List[ClassifiedEntity] = []
    
    if config.OPENAI_API_KEY:
        classifier = AIClassifier()
        
        for pdf_doc in pdf_docs:
            try:
                doc_results = all_unredaction_results.get(pdf_doc.filepath, [])
                entities = classifier.classify_document(pdf_doc, doc_results)
                all_entities.extend(entities)
            except Exception as e:
                print(f"\n{Fore.RED}  Error classifying {pdf_doc.filename}: {e}{Style.RESET_ALL}")
        
        results["entities_classified"] = len(all_entities)
        results["peps_found"] = sum(1 for e in all_entities if e.entity_type.value == "PEP")
        results["victims_found"] = sum(1 for e in all_entities if e.entity_type.value == "VICTIM")
    else:
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
        description="PDF Unredactor - Crawl websites, unredact PDFs, and classify entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Crawl a website for PDFs about a topic:
    python main.py --url "https://example.com/documents" --topic "fraud investigation"
  
  Process local PDFs:
    python main.py --pdf-dir "./documents" --topic "financial scandal"
  
  Full options:
    python main.py --url "https://site.com" --topic "corruption" --depth 3 --excel --csv

Environment:
  Create a .env file with:
    OPENAI_API_KEY=your-api-key-here
        """
    )
    
    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--url", "-u",
        help="Base URL to start crawling from"
    )
    source_group.add_argument(
        "--pdf-dir", "-d",
        help="Local directory containing PDFs to process"
    )
    
    # Topic
    parser.add_argument(
        "--topic", "-t",
        required=True,
        help="Topic to search for (used for relevance filtering)"
    )
    
    # Crawler options
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum crawl depth (default: 3)"
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
    
    # Handle --no-excel flag
    if args.no_excel:
        args.excel = False
    
    # Print banner
    print_banner()
    
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

