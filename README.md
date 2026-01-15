# Document Intelligence Pipeline

---
## Project Overview

An intelligent document processing pipeline that leverages **multiple Large Language Models (LLMs)** in parallel for robust document classification and data extraction. Built with **LangGraph** for orchestration, the system uses **OCR-enabled** ingestion and **ensemble voting** to achieve high-accuracy results.

This pipeline automates the extraction of structured data from unstructured documents (invoices, contracts, emails, meeting minutes) by:

1. **Ingesting** PDF documents with intelligent OCR fallback
2. **Classifying** document types using ensemble LLM voting
3. **Extracting** structured fields via parallel multi-model inference
4. **Merging** results using intelligent voting and quality scoring
5. **Exporting** validated data to JSON and CSV formats

---
## Repository Structure

```
document-intelligence-pipeline/
│
├── data/
│   ├── input/                # Source PDF documents
│   │
│   └── output/
│       ├── json/             # Individual document JSONs
│       └── master_data.csv   # Flattened export
│
├── src/
│   ├── __init__.py           # Package initializer
│   ├── ingestion.py          # PDF ingestion with OCR fallback
│   ├── orchestrator.py       # LangGraph ensemble orchestration
│   ├── schemas.py            # Pydantic data models
│   └── export.py             # JSON/CSV export utilities
│
├── test.py                   # End-to-end pipeline test
└── README.md                  
```

---
## Pipeline Workflow (LangGraph Architecture)

The orchestration layer uses **LangGraph** to enable parallel execution across multiple LLM providers with automatic result aggregation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INTELLIGENCE PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │        DOCUMENT INGESTION       │
                    │  ─────────────────────────────  │
                    │   • PDF parsing (pdfplumber)    │
                    │   • OCR fallback (Tesseract)    │
                    │   • Metadata extraction         │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │      ENSEMBLE CLASSIFICATION    │
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
                    │         VOTING AGGREGATION      │
                    │  ─────────────────────────────  │
                    │   • Majority vote on doc type   │
                    │   • Average confidence score    │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │      ENSEMBLE EXTRACTION        │
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
                    │         FIELD MERGER            │
                    │  ─────────────────────────────  │
                    │   • Numeric: averaging          │
                    │   • Strings: majority vote      │
                    │   • Lists: union deduplication  │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         PYDANTIC VALIDATION     │
                    │  ─────────────────────────────  │
                    │   • Schema enforcement          │
                    │   • Type coercion               │
                    │   • Confidence scoring          │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         EXPORT                  │
                    │  ─────────────────────────────  │
                    │   • Individual JSON files       │
                    │   • Flattened master CSV        │
                    └─────────────────────────────────┘
```

---
## Key Technical Features

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
## Architecture Highlights

| Principle | Implementation |
|-----------|----------------|
| **Fault Tolerance** | Multi-provider redundancy; continues if 1-2 providers fail |
| **Scalability** | LangGraph enables easy addition of new LLM providers |
| **Extensibility** | Pydantic schemas allow rapid addition of new document types |
| **Observability** | Comprehensive logging at each pipeline stage |
| **Cost Optimization** | Local Ollama option for development/high-volume scenarios |

---
## Team

Rui Zhao, Othmane Zizi, Florence Wang, Yasmine Zhao, Calvin Chun Fung Yip
