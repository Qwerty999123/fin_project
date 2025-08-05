import os
import re
import requests
import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict, Optional
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, source: str):
        self.source = source
        self._validate_source()

    def _validate_source(self):
        if self.source.startswith("http"):
            try:
                head = requests.head(self.source, timeout=5, allow_redirects=True)
                head.raise_for_status()
            except Exception as e:
                raise ValueError(f"URL inaccessible: {str(e)}")
        elif not os.path.exists(self.source):
            raise FileNotFoundError(f"Local file not found: {self.source}")

    def extract(self) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement extract()")

class PDFLoader(DocumentLoader):
    def __init__(self, source: str):
        super().__init__(source)
        self.is_url = self.source.startswith("http")
        self.header_pattern = re.compile(
            r"^(§|\d+\.\d+|\bARTICLE\b|\bCLAUSE\b|\bSECTION\b)", 
            re.IGNORECASE
        )

    def _download_pdf(self) -> str:
        local_path = "temp_blob.pdf"
        try:
            response = requests.get(self.source, timeout=10, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path
        except Exception as e:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Download failed: {str(e)}")

    def _extract_with_structure(self, pdf_path: str) -> str:
        full_text = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
                    page_header = f"\n[PAGE {page_num}]\n"
                    full_text.append(page_header + text)
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed: {str(e)}")
        return "\n".join(full_text)

    def _chunk_text(self, text: str) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n§", "\nArticle", "\nClause", "\nSECTION", 
                "\nSubsection", "\n\n", "\n", " ", ""
            ]
        )
        
        chunks = splitter.split_text(text)
        structured_chunks = []
        
        for chunk in chunks:
            header_match = self.header_pattern.search(chunk)
            header = header_match.group(0).strip() if header_match else "General"
            page_match = re.search(r"\[PAGE (\d+)\]", chunk)
            page = int(page_match.group(1)) if page_match else 1
            
            structured_chunks.append({
                "text": chunk,
                "header": header,
                "page": page,
                "type": "clause" if header_match else "text_block"
            })
            
        return structured_chunks

    def _extract_tables(self, pdf_path: str) -> List[Dict]:
        tables = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    for table in page.find_tables():
                        table_data = []
                        for row in table.extract():
                            # Filter None values and convert to strings
                            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                            table_data.append("|".join(cleaned_row))
                        
                        if table_data:  # Only add non-empty tables
                            tables.append({
                                "text": f"[TABLE {page_num}.{len(tables)+1}]\n" + "\n".join(table_data),
                                "header": f"Table {page_num}.{len(tables)+1}",
                                "page": page_num,
                                "type": "table"
                            })
        except Exception as e:
            print(f"Table extraction warning: {str(e)}")
        return tables

    def extract(self) -> List[Dict]:
        pdf_path = self._download_pdf() if self.is_url else self.source
        try:
            full_text = self._extract_with_structure(pdf_path)
            chunks = self._chunk_text(full_text)
            tables = self._extract_tables(pdf_path)
            return chunks + tables
        finally:
            if self.is_url and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except PermissionError:
                    # Handle file lock issues on Windows
                    import time
                    time.sleep(0.1)
                    os.remove(pdf_path)

class DOCXLoader(DocumentLoader):
    def extract(self) -> List[Dict]:
        chunks = []
        current_header = None
        
        try:
            doc = Document(self.source)
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                    
                if para.style.name.lower().startswith(('heading', 'title')):
                    current_header = text
                    continue
                    
                chunks.append({
                    "text": text,
                    "header": current_header or "General",
                    "style": para.style.name,
                    "type": "heading" if current_header else "paragraph"
                })
        except Exception as e:
            raise RuntimeError(f"DOCX processing failed: {str(e)}")
            
        return chunks

def load_document(source: str) -> Dict:
    def _is_pdf(content: bytes) -> bool:
        return content[:4] == b'%PDF'

    def _is_docx(content: bytes) -> bool:
        return (b'word/_rels' in content or 
                b'[Content_Types].xml' in content)

    try:
        # Content-based detection
        if source.startswith("http"):
            response = requests.get(source, stream=True)
            response.raise_for_status()
            sample = response.raw.read(1024)
        else:
            with open(source, 'rb') as f:
                sample = f.read(1024)

        if _is_pdf(sample):
            loader = PDFLoader(source)
        elif _is_docx(sample):
            loader = DOCXLoader(source)
        else:
            raise ValueError("Unrecognized file format")

    except Exception as e:
        # Extension fallback
        ext = os.path.splitext(urlparse(source).path if source.startswith("http") else source)[1].lower()
        if ext == '.pdf':
            loader = PDFLoader(source)
        elif ext == '.docx':
            loader = DOCXLoader(source)
        else:
            raise ValueError(f"Unsupported file type (extension: {ext})")

    return {
        "source": source,
        "chunks": loader.extract()
    }

if __name__ == '__main__':

    output = load_document('https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D')
    print("hello")
    print(len(output['chunks']))