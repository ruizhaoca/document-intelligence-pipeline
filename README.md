# 📄 Document Intelligence Pipeline

> **Ensemble-based Document Processing with Multi-LLM Orchestration**

An intelligent document processing pipeline that leverages multiple Large Language Models (LLMs) in parallel for robust document classification and data extraction. Built with **LangGraph** for orchestration, the system uses OCR-enabled ingestion and ensemble voting to achieve high-accuracy results.

---

## 🎯 Project Overview

This pipeline automates the extraction of structured data from unstructured documents (invoices, contracts, emails, meeting minutes) by:

1. **Ingesting** PDF documents with intelligent OCR fallback
2. **Classifying** document types using ensemble LLM voting
3. **Extracting** structured fields via parallel multi-model inference
4. **Merging** results using intelligent voting and quality scoring
5. **Exporting** validated data to JSON and CSV formats

The system is designed for **production reliability** through redundancy—if one LLM provider fails or returns low-quality results, others compensate automatically.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Orchestration** | LangGraph (StateGraph, Conditional Edges, Fan-out/Fan-in) |
| **LLM Providers** | OpenAI GPT-4o, Google Gemini 2.5 Flash, Ollama (Qwen2.5:7b) |
| **Document Processing** | pdfplumber, pdf2image, Tesseract OCR |
| **Data Validation** | Pydantic v2 (BaseModel, Field validators) |
| **Data Export** | Pandas, JSON |
| **Language** | Python 3.10+ |

---

## 📁 Repository Structure

```
document-intelligence-pipeline/
│
├── src/
│   ├── ingestion.py      # PDF ingestion with OCR fallback
│   ├── orchestrator.py   # LangGraph ensemble orchestration
│   ├── schemas.py        # Pydantic data models
│   └── export.py         # JSON/CSV export utilities
│
├── data/
│   ├── input/            # Source PDF documents
│   └── output/
│       ├── json/         # Individual document JSONs
│       └── master_data.csv
│
├── test.py               # End-to-end pipeline test
└── README.md
```

### Module Breakdown

| File | Purpose |
|------|---------|
| `ingestion.py` | Handles PDF text extraction with automatic OCR when native extraction fails |
| `orchestrator.py` | LangGraph-based ensemble orchestrator for parallel LLM inference |
| `schemas.py` | Pydantic models for Invoice, Contract, Email, Meeting Minutes |
| `export.py` | Exports structured documents to JSON files and flattened CSV |
| `test.py` | Main entry point demonstrating the full pipeline |

---

## 🔄 Pipeline Workflow (LangGraph Architecture)

The orchestration layer uses **LangGraph** to enable parallel execution across multiple LLM providers with automatic result aggregation.

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INTELLIGENCE PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │     📥 DOCUMENT INGESTION       │
                    │   ─────────────────────────────  │
                    │   • PDF parsing (pdfplumber)    │
                    │   • OCR fallback (Tesseract)    │
                    │   • Metadata extraction         │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   🏷️ ENSEMBLE CLASSIFICATION    │
                    │        (LangGraph Graph)        │
                    └─────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
     │   OpenAI    │         │   Gemini    │         │   Ollama    │
     │   GPT-4o    │         │  2.5 Flash  │         │ Qwen2.5:7b  │
     └─────────────┘         └─────────────┘         └─────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                    ┌─────────────────────────────────┐
                    │      🗳️ VOTING AGGREGATION      │
                    │   ─────────────────────────────  │
                    │   • Majority vote on doc type   │
                    │   • Average confidence score    │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │   📊 ENSEMBLE EXTRACTION        │
                    │        (LangGraph Graph)        │
                    └─────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
     │   OpenAI    │         │   Gemini    │         │   Ollama    │
     │  Extract    │         │  Extract    │         │  Extract    │
     └─────────────┘         └─────────────┘         └─────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                    ┌─────────────────────────────────┐
                    │      🔀 FIELD MERGER            │
                    │   ─────────────────────────────  │
                    │   • Numeric: averaging          │
                    │   • Strings: majority vote      │
                    │   • Lists: union deduplication  │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │      ✅ PYDANTIC VALIDATION     │
                    │   ─────────────────────────────  │
                    │   • Schema enforcement          │
                    │   • Type coercion               │
                    │   • Confidence scoring          │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │      📤 EXPORT                  │
                    │   ─────────────────────────────  │
                    │   • Individual JSON files       │
                    │   • Flattened master CSV        │
                    └─────────────────────────────────┘
```

### LangGraph State Machines

#### Classification Graph

```
                         ┌─────────┐
                         │  START  │
                         └────┬────┘
                              │
                    ┌─────────┴─────────┐
                    │  classification   │
                    │     _router       │
                    │   (Fan-out)       │
                    └─────────┬─────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ classify_openai │  │ classify_gemini │  │ classify_ollama │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                         ┌─────────┐
                         │   END   │
                         │(Fan-in) │
                         └─────────┘
```

#### Extraction Graph

```
                         ┌─────────┐
                         │  START  │
                         └────┬────┘
                              │
                    ┌─────────┴─────────┐
                    │   extraction      │
                    │     _router       │
                    │   (Fan-out)       │
                    └─────────┬─────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ extract_openai  │  │ extract_gemini  │  │ extract_ollama  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   FieldMerger   │
                    │   (Aggregator)  │
                    └────────┬────────┘
                              │
                              ▼
                         ┌─────────┐
                         │   END   │
                         └─────────┘
```

---

## 🔑 Key Technical Features

### 1. LangGraph Orchestration
- **StateGraph** with typed state dictionaries (`TypedDict`)
- **Conditional routing** via `Send` for dynamic fan-out to available providers
- **Annotated reducers** (`operator.add`) for automatic result aggregation
- Compiled graphs for optimized execution

### 2. Multi-Provider LLM Integration
- **OpenAI GPT-4o**: JSON mode for structured outputs
- **Google Gemini 2.5 Flash**: Custom safety settings, generation config
- **Ollama (Local)**: Self-hosted Qwen2.5:7b for privacy/cost optimization
- Graceful degradation when providers are unavailable

### 3. Intelligent OCR Pipeline
- Primary extraction via `pdfplumber` (native PDF text)
- Automatic OCR fallback using `Tesseract` + `pdf2image`
- Configurable thresholds for OCR triggering
- High-DPI (300) image conversion for accuracy

### 4. Ensemble Result Merging
- **Numeric fields**: Averaging across models
- **String fields**: Majority voting (Counter-based)
- **List fields**: Union with deduplication
- Handles partial failures gracefully

### 5. Pydantic Schema Validation
- Strict type enforcement with `BaseModel`
- UUID generation for document tracking
- Enum-based document type classification
- Factory pattern for polymorphic document creation

---

## 📊 Supported Document Types

| Type | Extracted Fields |
|------|------------------|
| **Invoice** | invoice_number, date, vendor, client, amounts, tax, line_items, payment_method |
| **Contract** | contract_id, parties, value, effective/expiry dates, key_terms |
| **Email** | sender, recipients, date, subject, key_points, attachments |
| **Meeting Minutes** | date, title, attendees, agenda, decisions, action_items |

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr poppler-utils

# Install Python dependencies
pip install langgraph openai google-generativeai pydantic pdfplumber pdf2image pytesseract pandas
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
# Ollama runs locally on http://localhost:11434
```

### Run the Pipeline

```bash
# Place PDFs in data/input/
python test.py
```

---

## 📈 Sample Output

```
================================================================================
ADVANCED PIPELINE: ENSEMBLE EXTRACTION
================================================================================

Features:
  - Parallel extraction from OpenAI + Gemini + Ollama
  - Intelligent result merging with voting

Step 3: ENSEMBLE CLASSIFICATION...
   Processing: cargo.pdf
   Result: invoice (95.0%) via openai, gemini, ollama

Step 4: ENSEMBLE EXTRACTION...
   Extracted via: openai, gemini, ollama
   Fields extracted: 10
      - invoice_number: 2011981
      - vendor_name: Cargo Collective, Inc.
      - total_amount: 99.0
      - currency: USD

PIPELINE COMPLETE!
   - Documents processed: 3
   - Ensemble average confidence: 94.2%
```

---

## 🏗️ Architecture Highlights

| Principle | Implementation |
|-----------|----------------|
| **Fault Tolerance** | Multi-provider redundancy; continues if 1-2 providers fail |
| **Scalability** | LangGraph enables easy addition of new LLM providers |
| **Extensibility** | Pydantic schemas allow rapid addition of new document types |
| **Observability** | Comprehensive logging at each pipeline stage |
| **Cost Optimization** | Local Ollama option for development/high-volume scenarios |

---

## 📜 License

MIT License

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional LLM provider integrations (Anthropic Claude, Cohere)
- New document type schemas
- Streaming extraction for large documents
- Web UI for document upload and results visualization
