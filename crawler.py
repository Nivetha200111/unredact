"""
Depth-First Search Web Crawler for PDF Discovery
"""
import os
import re
import time
import hashlib
from urllib.parse import urljoin, urlparse
from typing import Set, List, Tuple, Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from colorama import Fore, Style

import config


@dataclass
class PDFInfo:
    """Information about a discovered PDF"""
    url: str
    filename: str
    local_path: str
    source_page: str
    topic_relevance: float


class WebCrawler:
    """DFS-based web crawler for finding and downloading PDFs"""
    
    def __init__(self, base_url: str, topic: str, max_depth: int = None):
        self.base_url = base_url
        self.topic = topic.lower()
        self.topic_keywords = self._extract_keywords(topic)
        self.max_depth = max_depth or config.MAX_DEPTH
        
        self.visited_urls: Set[str] = set()
        self.discovered_pdfs: List[PDFInfo] = []
        self.session = self._create_session()
        
        # Create downloads directory
        os.makedirs(config.DOWNLOADS_DIR, exist_ok=True)
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            "User-Agent": config.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        return session
    
    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract meaningful keywords from topic string"""
        # Remove common words and split into keywords
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "are"}
        words = re.findall(r'\b\w+\b', topic.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        base_domain = urlparse(self.base_url).netloc
        url_domain = urlparse(url).netloc
        return url_domain == base_domain or url_domain == ""
    
    def _normalize_url(self, url: str, current_page: str) -> Optional[str]:
        """Normalize and validate a URL"""
        if not url:
            return None
            
        # Skip non-http links
        if url.startswith(("javascript:", "mailto:", "tel:", "#")):
            return None
            
        # Make absolute URL
        absolute_url = urljoin(current_page, url)
        
        # Remove fragments
        parsed = urlparse(absolute_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        if parsed.query:
            clean_url += f"?{parsed.query}"
            
        return clean_url
    
    def _calculate_relevance(self, url: str, page_text: str, link_text: str) -> float:
        """Calculate topic relevance score (0-1)"""
        score = 0.0
        text_to_check = f"{url} {page_text} {link_text}".lower()
        
        for keyword in self.topic_keywords:
            if keyword in text_to_check:
                score += 1.0 / len(self.topic_keywords)
                
        # Boost if keywords appear in URL or link text
        url_lower = url.lower()
        link_lower = link_text.lower()
        
        for keyword in self.topic_keywords:
            if keyword in url_lower:
                score += 0.2
            if keyword in link_lower:
                score += 0.2
                
        return min(score, 1.0)
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF"""
        return url.lower().endswith(".pdf") or "pdf" in url.lower()
    
    def _generate_filename(self, url: str) -> str:
        """Generate a unique filename for a PDF"""
        # Try to extract filename from URL
        parsed = urlparse(url)
        path = parsed.path
        
        if path.endswith(".pdf"):
            filename = os.path.basename(path)
        else:
            # Generate hash-based filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            filename = f"document_{url_hash}.pdf"
            
        # Ensure uniqueness
        base, ext = os.path.splitext(filename)
        counter = 1
        final_filename = filename
        
        while os.path.exists(os.path.join(config.DOWNLOADS_DIR, final_filename)):
            final_filename = f"{base}_{counter}{ext}"
            counter += 1
            
        return final_filename
    
    def _download_pdf(self, url: str, source_page: str, relevance: float) -> Optional[PDFInfo]:
        """Download a PDF file"""
        try:
            response = self.session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            # Verify it's actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                # Check magic bytes
                first_bytes = response.content[:5]
                if first_bytes != b"%PDF-":
                    return None
            
            filename = self._generate_filename(url)
            local_path = os.path.join(config.DOWNLOADS_DIR, filename)
            
            with open(local_path, "wb") as f:
                f.write(response.content)
                
            pdf_info = PDFInfo(
                url=url,
                filename=filename,
                local_path=local_path,
                source_page=source_page,
                topic_relevance=relevance
            )
            
            print(f"{Fore.GREEN}✓ Downloaded: {filename} (relevance: {relevance:.2f}){Style.RESET_ALL}")
            return pdf_info
            
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to download {url}: {e}{Style.RESET_ALL}")
            return None
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page"""
        try:
            response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.content, "lxml")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Could not fetch {url}: {e}{Style.RESET_ALL}")
            return None
    
    def _crawl_dfs(self, url: str, depth: int, pbar: tqdm):
        """Recursive DFS crawl implementation"""
        if depth > self.max_depth:
            return
            
        if url in self.visited_urls:
            return
            
        self.visited_urls.add(url)
        pbar.set_description(f"Depth {depth}, visited {len(self.visited_urls)}")
        pbar.update(1)
        
        # Respect rate limiting
        time.sleep(config.REQUEST_DELAY)
        
        # Check if this is a PDF
        if self._is_pdf_url(url):
            relevance = self._calculate_relevance(url, "", "")
            if relevance > 0:
                pdf_info = self._download_pdf(url, url, relevance)
                if pdf_info:
                    self.discovered_pdfs.append(pdf_info)
            return
        
        # Fetch and parse the page
        soup = self._fetch_page(url)
        if not soup:
            return
            
        # Get page text for relevance calculation
        page_text = soup.get_text()[:1000]  # First 1000 chars for context
        
        # Find all links
        links = soup.find_all("a", href=True)
        
        # Collect links to visit (DFS - process PDFs first, then pages)
        pdf_links = []
        page_links = []
        
        for link in links:
            href = link.get("href", "")
            normalized_url = self._normalize_url(href, url)
            
            if not normalized_url:
                continue
                
            if normalized_url in self.visited_urls:
                continue
                
            link_text = link.get_text(strip=True)
            relevance = self._calculate_relevance(normalized_url, page_text, link_text)
            
            if self._is_pdf_url(normalized_url):
                if relevance > 0:  # Only download relevant PDFs
                    pdf_links.append((normalized_url, link_text, relevance))
            elif self._is_same_domain(normalized_url):
                page_links.append((normalized_url, link_text, relevance))
        
        # Process PDFs first
        for pdf_url, link_text, relevance in pdf_links:
            if pdf_url not in self.visited_urls:
                self.visited_urls.add(pdf_url)
                pdf_info = self._download_pdf(pdf_url, url, relevance)
                if pdf_info:
                    self.discovered_pdfs.append(pdf_info)
        
        # Then recursively crawl pages (DFS - go deep first)
        # Sort by relevance to prioritize more relevant pages
        page_links.sort(key=lambda x: x[2], reverse=True)
        
        for page_url, link_text, relevance in page_links:
            self._crawl_dfs(page_url, depth + 1, pbar)
    
    def crawl(self) -> List[PDFInfo]:
        """Start the DFS crawl from the base URL"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"Starting DFS Crawl")
        print(f"Base URL: {self.base_url}")
        print(f"Topic: {self.topic}")
        print(f"Keywords: {', '.join(self.topic_keywords)}")
        print(f"Max Depth: {self.max_depth}")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        with tqdm(desc="Crawling", unit=" pages") as pbar:
            self._crawl_dfs(self.base_url, 0, pbar)
        
        print(f"\n{Fore.GREEN}Crawl complete!")
        print(f"Pages visited: {len(self.visited_urls)}")
        print(f"PDFs discovered: {len(self.discovered_pdfs)}{Style.RESET_ALL}\n")
        
        return self.discovered_pdfs


def crawl_for_pdfs(base_url: str, topic: str, max_depth: int = None) -> List[PDFInfo]:
    """Convenience function to crawl a website for topic-related PDFs"""
    crawler = WebCrawler(base_url, topic, max_depth)
    return crawler.crawl()

