"""
Table Generator - Creates organized output tables of all findings
"""
import os
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style

from pdf_processor import PDFDocument
from unredactor import UnredactionResult
from ai_classifier import ClassifiedEntity, EntityType
import config


class TableGenerator:
    """Generates organized tables and reports from processing results"""
    
    def __init__(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_redaction_table(self, 
                               pdf_docs: List[PDFDocument],
                               unredaction_results: Dict[str, List[UnredactionResult]]) -> pd.DataFrame:
        """Create a table of all redactions and their recovered text"""
        rows = []
        
        for pdf_doc in pdf_docs:
            doc_results = unredaction_results.get(pdf_doc.filepath, [])
            
            for i, result in enumerate(doc_results):
                redaction = result.original_redaction
                
                rows.append({
                    "Document": pdf_doc.filename,
                    "Page": redaction.page_num + 1,
                    "Redaction Type": redaction.redaction_type.value,
                    "Detection Confidence": f"{redaction.confidence:.0%}",
                    "Recovered Text": result.recovered_text or "[Unable to recover]",
                    "Recovery Method": result.method_used,
                    "Recovery Confidence": f"{result.confidence:.0%}" if result.success else "N/A",
                    "Location (x,y,w,h)": f"({redaction.bbox[0]:.0f}, {redaction.bbox[1]:.0f}, "
                                         f"{redaction.bbox[2]-redaction.bbox[0]:.0f}, "
                                         f"{redaction.bbox[3]-redaction.bbox[1]:.0f})"
                })
        
        return pd.DataFrame(rows)
    
    def create_entity_table(self, 
                           entities: List[ClassifiedEntity]) -> pd.DataFrame:
        """Create a table of all classified entities"""
        rows = []
        
        for entity in entities:
            rows.append({
                "Name": entity.name,
                "Email": entity.email or "N/A",
                "Classification": entity.entity_type.value,
                "Confidence": f"{entity.confidence:.0%}",
                "Was Redacted": "Yes" if entity.was_redacted else "No",
                "Source Document": entity.source_document,
                "Page": entity.source_page + 1 if entity.source_page is not None else "N/A",
                "Reasoning": entity.reasoning[:100] + "..." if len(entity.reasoning) > 100 else entity.reasoning
            })
        
        return pd.DataFrame(rows)
    
    def create_email_table(self,
                          pdf_docs: List[PDFDocument],
                          entities: List[ClassifiedEntity]) -> pd.DataFrame:
        """Create a comprehensive table of all emails with their classifications"""
        email_info = {}
        
        # Collect emails from documents
        for pdf_doc in pdf_docs:
            for email in pdf_doc.emails_found:
                if email not in email_info:
                    email_info[email] = {
                        "email": email,
                        "documents": [],
                        "entity_name": None,
                        "classification": EntityType.UNKNOWN,
                        "was_redacted": False,
                        "confidence": 0.0,
                        "reasoning": ""
                    }
                email_info[email]["documents"].append(pdf_doc.filename)
        
        # Enrich with entity information
        for entity in entities:
            if entity.email and entity.email in email_info:
                info = email_info[entity.email]
                if entity.confidence > info["confidence"]:
                    info["entity_name"] = entity.name
                    info["classification"] = entity.entity_type
                    info["was_redacted"] = entity.was_redacted or info["was_redacted"]
                    info["confidence"] = entity.confidence
                    info["reasoning"] = entity.reasoning
        
        rows = []
        for email, info in email_info.items():
            rows.append({
                "Email": email,
                "Associated Name": info["entity_name"] or "Unknown",
                "Classification": info["classification"].value,
                "Confidence": f"{info['confidence']:.0%}",
                "Was Redacted": "Yes" if info["was_redacted"] else "No",
                "Found In Documents": ", ".join(info["documents"][:3]) + 
                                     (f" (+{len(info['documents'])-3} more)" if len(info["documents"]) > 3 else ""),
                "Notes": info["reasoning"][:80] + "..." if len(info["reasoning"]) > 80 else info["reasoning"]
            })
        
        return pd.DataFrame(rows)
    
    def create_summary_table(self,
                            pdf_docs: List[PDFDocument],
                            unredaction_results: Dict[str, List[UnredactionResult]],
                            entities: List[ClassifiedEntity]) -> pd.DataFrame:
        """Create a summary table of all processed documents"""
        rows = []
        
        for pdf_doc in pdf_docs:
            doc_results = unredaction_results.get(pdf_doc.filepath, [])
            doc_entities = [e for e in entities if e.source_document == pdf_doc.filename]
            
            successful_unredactions = sum(1 for r in doc_results if r.success)
            pep_count = sum(1 for e in doc_entities if e.entity_type == EntityType.PEP)
            victim_count = sum(1 for e in doc_entities if e.entity_type == EntityType.VICTIM)
            
            rows.append({
                "Document": pdf_doc.filename,
                "Pages": pdf_doc.num_pages,
                "Total Redactions": len(pdf_doc.redactions),
                "Successful Unredactions": successful_unredactions,
                "Unredaction Rate": f"{successful_unredactions/max(len(doc_results), 1):.0%}",
                "Emails Found": len(pdf_doc.emails_found),
                "PEPs Identified": pep_count,
                "Victims Identified": victim_count,
                "Total Entities": len(doc_entities)
            })
        
        return pd.DataFrame(rows)
    
    def print_table(self, df: pd.DataFrame, title: str):
        """Print a formatted table to console"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f" {title}")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        if df.empty:
            print(f"{Fore.YELLOW}No data to display.{Style.RESET_ALL}")
        else:
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, maxcolwidths=40))
    
    def export_to_excel(self,
                       summary_df: pd.DataFrame,
                       redaction_df: pd.DataFrame,
                       entity_df: pd.DataFrame,
                       email_df: pd.DataFrame,
                       filename: str = None) -> str:
        """Export all tables to an Excel file with multiple sheets"""
        if filename is None:
            filename = f"unredact_report_{self.timestamp}.xlsx"
        
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            redaction_df.to_excel(writer, sheet_name='Redactions', index=False)
            entity_df.to_excel(writer, sheet_name='Entities', index=False)
            email_df.to_excel(writer, sheet_name='Emails', index=False)
        
        return filepath
    
    def export_to_csv(self,
                     summary_df: pd.DataFrame,
                     redaction_df: pd.DataFrame,
                     entity_df: pd.DataFrame,
                     email_df: pd.DataFrame) -> Dict[str, str]:
        """Export all tables to separate CSV files"""
        files = {}
        
        for name, df in [("summary", summary_df), 
                         ("redactions", redaction_df), 
                         ("entities", entity_df),
                         ("emails", email_df)]:
            filepath = os.path.join(config.OUTPUT_DIR, f"{name}_{self.timestamp}.csv")
            df.to_csv(filepath, index=False)
            files[name] = filepath
        
        return files
    
    def generate_report(self,
                       pdf_docs: List[PDFDocument],
                       unredaction_results: Dict[str, List[UnredactionResult]],
                       entities: List[ClassifiedEntity],
                       export_excel: bool = True,
                       export_csv: bool = False,
                       print_tables: bool = True) -> Dict[str, Any]:
        """Generate complete report with all tables"""
        
        # Create all tables
        summary_df = self.create_summary_table(pdf_docs, unredaction_results, entities)
        redaction_df = self.create_redaction_table(pdf_docs, unredaction_results)
        entity_df = self.create_entity_table(entities)
        email_df = self.create_email_table(pdf_docs, entities)
        
        result = {
            "summary": summary_df,
            "redactions": redaction_df,
            "entities": entity_df,
            "emails": email_df,
            "files": {}
        }
        
        # Print tables to console
        if print_tables:
            self.print_table(summary_df, "DOCUMENT SUMMARY")
            self.print_table(redaction_df, "REDACTION DETAILS")
            self.print_table(entity_df, "CLASSIFIED ENTITIES (PEP/VICTIM)")
            self.print_table(email_df, "EMAIL ADDRESSES")
        
        # Export files
        if export_excel:
            excel_path = self.export_to_excel(summary_df, redaction_df, entity_df, email_df)
            result["files"]["excel"] = excel_path
            print(f"\n{Fore.GREEN}✓ Excel report saved: {excel_path}{Style.RESET_ALL}")
        
        if export_csv:
            csv_paths = self.export_to_csv(summary_df, redaction_df, entity_df, email_df)
            result["files"]["csv"] = csv_paths
            print(f"{Fore.GREEN}✓ CSV reports saved to {config.OUTPUT_DIR}/{Style.RESET_ALL}")
        
        return result


def generate_report(pdf_docs: List[PDFDocument],
                   unredaction_results: Dict[str, List[UnredactionResult]],
                   entities: List[ClassifiedEntity],
                   **kwargs) -> Dict[str, Any]:
    """Convenience function to generate a complete report"""
    generator = TableGenerator()
    return generator.generate_report(pdf_docs, unredaction_results, entities, **kwargs)

