"""Document ingestion module for extracting text from PDFs using OCR when needed."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles PDF document ingestion and text extraction with OCR fallback."""

    def __init__(self, use_ocr: bool = True, ocr_threshold: int = 999999, prefer_ocr: bool = True):
        """
        Initialize the DocumentIngestor.

        Args:
            use_ocr: Whether to use OCR.
            ocr_threshold: Character count threshold to trigger OCR fallback.
            prefer_ocr: If True, always use OCR regardless of text extraction results.
        """
        self.supported_formats = [".pdf"]
        self.use_ocr = bool(use_ocr)
        self.ocr_threshold = int(ocr_threshold)
        self.prefer_ocr = bool(prefer_ocr)

    def extract_text_with_ocr(self, file_path: Path) -> str:
        """
        Extract text from a PDF using OCR.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text from all pages.
        """
        if not self.use_ocr:
            return ""

        try:
            logger.info(f"Using OCR to extract text from {file_path.name}")
            images = convert_from_path(str(file_path), dpi=300)

            text_pages = []
            for i, img in enumerate(images, start=1):
                logger.debug(f"OCR processing page {i}/{len(images)}")
                page_text = pytesseract.image_to_string(img)
                if page_text.strip():
                    text_pages.append(page_text)

            full_text = "\n\n".join(text_pages)
            logger.info(f"OCR extracted {len(full_text)} characters from {len(images)} pages")
            return full_text

        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return ""

    def ingest_pdf(self, file_path: str) -> Optional[Dict]:
        """
        Extract text and metadata from a PDF file, using OCR fallback if needed.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary containing extracted text and metadata.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {path}")
            return None

        if path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported format: {path.suffix}")
            return None

        try:
            with pdfplumber.open(path) as pdf:
                text_pages = []
                tables = []

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)

                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)

                full_text = "\n\n".join(text_pages)

                if self.prefer_ocr:
                    needs_ocr = True
                    logger.info(f"{path.name}: Using OCR (systematic mode)")
                else:
                    needs_ocr = len(full_text.strip()) < self.ocr_threshold
                    if needs_ocr:
                        logger.warning(
                            f"{path.name}: Low text extraction ({len(full_text)} chars). Attempting OCR..."
                        )

                if needs_ocr and self.use_ocr:
                    ocr_text = self.extract_text_with_ocr(path)
                    if len(ocr_text) > len(full_text) or self.prefer_ocr:
                        full_text = ocr_text
                        logger.info(f"Using OCR text for {path.name}")

                metadata = {
                    "file_name": path.name,
                    "file_path": str(path),
                    "num_pages": len(pdf.pages),
                    "file_size": path.stat().st_size,
                    "has_tables": len(tables) > 0,
                    "used_ocr": needs_ocr and self.use_ocr,
                    "text_length": len(full_text),
                }

                result = {"text": full_text, "metadata": metadata, "tables": tables}

                logger.info(
                    f"Successfully ingested: {path.name} ({len(pdf.pages)} pages, {len(full_text)} chars)"
                )
                return result

        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None

    def batch_ingest(self, directory: str, pattern: str = "*.pdf") -> List[Dict]:
        """
        Ingest all PDF files from a directory.

        Args:
            directory: Path to the directory containing PDFs.
            pattern: File pattern to match.

        Returns:
            List of dictionaries containing extracted documents.
        """
        directory_path = Path(directory)

        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []

        pdf_files = list(directory_path.glob(pattern))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results = []
        for pdf_file in pdf_files:
            result = self.ingest_pdf(str(pdf_file))
            if result is not None:
                results.append(result)

        logger.info(f"Successfully ingested {len(results)}/{len(pdf_files)} documents")
        return results
