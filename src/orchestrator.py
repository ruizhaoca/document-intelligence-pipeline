"""Ensemble orchestrator using LangGraph for parallel OpenAI, Gemini, and Ollama extraction."""

import json
import logging
import operator
import os
from collections import Counter
from typing import Annotated, Any, Dict, List, Tuple

import google.generativeai as genai
import requests
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Gemini safety settings
GEMINI_SAFETY = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


class ClassificationResult(TypedDict):
    provider: str
    doc_type: str
    confidence: float


class ExtractionResult(TypedDict):
    provider: str
    fields: Dict[str, Any]


class ClassificationState(TypedDict):
    text: str
    prompt_template: str
    results: Annotated[List[ClassificationResult], operator.add]


class ExtractionState(TypedDict):
    text: str
    doc_type: str
    prompt_template: str
    results: Annotated[List[ExtractionResult], operator.add]


class FieldMerger:
    """Merges extraction results from multiple models using voting and quality scoring."""

    @staticmethod
    def merge_extractions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple extraction results into one best result."""
        if not results:
            return {}
        if len(results) == 1:
            return results[0]

        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        merged = {}
        for key in all_keys:
            values = [r.get(key) for r in results if r.get(key) is not None]
            if not values:
                merged[key] = None
                continue

            sample = values[0]
            if isinstance(sample, (int, float)):
                merged[key] = sum(values) / len(values)
            elif isinstance(sample, list):
                all_items = []
                for v in values:
                    if isinstance(v, list):
                        all_items.extend(v)
                merged[key] = list(set(all_items))
            elif isinstance(sample, str):
                merged[key] = Counter(values).most_common(1)[0][0]
            else:
                merged[key] = values[0]

        logger.info(f"Merged {len(results)} results -> {len(merged)} fields")
        return merged


class LLMClients:
    """Manages LLM client connections."""

    def __init__(self):
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize available LLM clients."""
        # OpenAI
        try:
            self.clients["openai"] = {"client": OpenAI(api_key=OPENAI_API_KEY), "model": OPENAI_MODEL}
            logger.info("OpenAI initialized")
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

        # Gemini
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.clients["gemini"] = {"client": genai.GenerativeModel(GEMINI_MODEL), "model": GEMINI_MODEL}
            logger.info("Gemini initialized")
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

        # Ollama
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if response.status_code == 200:
                self.clients["ollama"] = {"url": OLLAMA_URL, "model": OLLAMA_MODEL}
                logger.info("Ollama initialized")
        except Exception:
            logger.info("Ollama not available")

    def get_available_providers(self) -> List[str]:
        return list(self.clients.keys())

    def classify(self, provider: str, text: str, prompt_template: str) -> Tuple[str, float]:
        """Classify using a single provider."""
        prompt = prompt_template.format(text=text[:3000])

        if provider == "openai":
            client = self.clients["openai"]["client"]
            response = client.chat.completions.create(
                model=self.clients["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are a document classifier. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("type", "unknown"), float(result.get("confidence", 0.5))

        elif provider == "gemini":
            model = self.clients["gemini"]["client"]
            config = genai.GenerationConfig(temperature=0.1, max_output_tokens=512)
            response = model.generate_content(prompt, generation_config=config, safety_settings=GEMINI_SAFETY)
            if not response.candidates or not response.text:
                raise ValueError("Response blocked")
            json_str = response.text[response.text.find("{") : response.text.rfind("}") + 1]
            result = json.loads(json_str)
            return result.get("type", "unknown"), float(result.get("confidence", 0.5))

        elif provider == "ollama":
            response = requests.post(
                f"{self.clients['ollama']['url']}/api/generate",
                json={"model": self.clients["ollama"]["model"], "prompt": prompt, "stream": False, "temperature": 0.1},
                timeout=30,
            )
            response_text = response.json().get("response", "")
            json_str = response_text[response_text.find("{") : response_text.rfind("}") + 1]
            result = json.loads(json_str)
            return result.get("type", "unknown"), float(result.get("confidence", 0.5))

        raise ValueError(f"Unknown provider: {provider}")

    def extract(self, provider: str, text: str, prompt_template: str) -> Dict[str, Any]:
        """Extract using a single provider."""
        prompt = prompt_template.format(text=text)

        if provider == "openai":
            client = self.clients["openai"]["client"]
            response = client.chat.completions.create(
                model=self.clients["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are a precise data extraction specialist. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)

        elif provider == "gemini":
            model = self.clients["gemini"]["client"]
            config = genai.GenerationConfig(temperature=0.1, max_output_tokens=2048)
            response = model.generate_content(prompt, generation_config=config, safety_settings=GEMINI_SAFETY)
            if not response.candidates or not response.text:
                raise ValueError("Response blocked by safety filters")
            json_str = response.text[response.text.find("{") : response.text.rfind("}") + 1]
            return json.loads(json_str)

        elif provider == "ollama":
            response = requests.post(
                f"{self.clients['ollama']['url']}/api/generate",
                json={"model": self.clients["ollama"]["model"], "prompt": prompt, "stream": False, "temperature": 0.1},
                timeout=60,
            )
            response_text = response.json().get("response", "")
            json_str = response_text[response_text.find("{") : response_text.rfind("}") + 1]
            return json.loads(json_str)

        raise ValueError(f"Unknown provider: {provider}")


# Global LLM clients instance
_llm_clients: LLMClients | None = None


def get_llm_clients() -> LLMClients:
    global _llm_clients
    if _llm_clients is None:
        _llm_clients = LLMClients()
    return _llm_clients


def classification_router(state: ClassificationState) -> List[Send]:
    """Fan out to all available providers for parallel classification."""
    clients = get_llm_clients()
    providers = clients.get_available_providers()
    logger.info(f"ENSEMBLE CLASSIFICATION: Routing to {len(providers)} providers via LangGraph")
    return [Send(f"classify_{provider}", state) for provider in providers]


def extraction_router(state: ExtractionState) -> List[Send]:
    """Fan out to all available providers for parallel extraction."""
    clients = get_llm_clients()
    providers = clients.get_available_providers()
    logger.info(f"ENSEMBLE EXTRACTION: Routing to {len(providers)} providers via LangGraph")
    return [Send(f"extract_{provider}", state) for provider in providers]


def classify_openai(state: ClassificationState) -> Dict:
    """Classification node for OpenAI."""
    clients = get_llm_clients()
    try:
        doc_type, confidence = clients.classify("openai", state["text"], state["prompt_template"])
        logger.info(f"  openai: {doc_type} ({confidence:.1%})")
        return {"results": [{"provider": "openai", "doc_type": doc_type, "confidence": confidence}]}
    except Exception as e:
        logger.error(f"  openai failed: {e}")
        return {"results": []}


def classify_gemini(state: ClassificationState) -> Dict:
    """Classification node for Gemini."""
    clients = get_llm_clients()
    try:
        doc_type, confidence = clients.classify("gemini", state["text"], state["prompt_template"])
        logger.info(f"  gemini: {doc_type} ({confidence:.1%})")
        return {"results": [{"provider": "gemini", "doc_type": doc_type, "confidence": confidence}]}
    except Exception as e:
        logger.error(f"  gemini failed: {e}")
        return {"results": []}


def classify_ollama(state: ClassificationState) -> Dict:
    """Classification node for Ollama."""
    clients = get_llm_clients()
    try:
        doc_type, confidence = clients.classify("ollama", state["text"], state["prompt_template"])
        logger.info(f"  ollama: {doc_type} ({confidence:.1%})")
        return {"results": [{"provider": "ollama", "doc_type": doc_type, "confidence": confidence}]}
    except Exception as e:
        logger.error(f"  ollama failed: {e}")
        return {"results": []}


def extract_openai(state: ExtractionState) -> Dict:
    """Extraction node for OpenAI."""
    clients = get_llm_clients()
    try:
        fields = clients.extract("openai", state["text"], state["prompt_template"])
        if fields:
            logger.info(f"  openai: {len(fields)} fields extracted")
            return {"results": [{"provider": "openai", "fields": fields}]}
        logger.warning("  openai: No fields extracted")
        return {"results": []}
    except Exception as e:
        logger.error(f"  openai failed: {e}")
        return {"results": []}


def extract_gemini(state: ExtractionState) -> Dict:
    """Extraction node for Gemini."""
    clients = get_llm_clients()
    try:
        fields = clients.extract("gemini", state["text"], state["prompt_template"])
        if fields:
            logger.info(f"  gemini: {len(fields)} fields extracted")
            return {"results": [{"provider": "gemini", "fields": fields}]}
        logger.warning("  gemini: No fields extracted")
        return {"results": []}
    except Exception as e:
        logger.error(f"  gemini failed: {e}")
        return {"results": []}


def extract_ollama(state: ExtractionState) -> Dict:
    """Extraction node for Ollama."""
    clients = get_llm_clients()
    try:
        fields = clients.extract("ollama", state["text"], state["prompt_template"])
        if fields:
            logger.info(f"  ollama: {len(fields)} fields extracted")
            return {"results": [{"provider": "ollama", "fields": fields}]}
        logger.warning("  ollama: No fields extracted")
        return {"results": []}
    except Exception as e:
        logger.error(f"  ollama failed: {e}")
        return {"results": []}


def build_classification_graph() -> StateGraph:
    """Build the LangGraph for ensemble classification."""
    graph = StateGraph(ClassificationState)
    graph.add_node("classify_openai", classify_openai)
    graph.add_node("classify_gemini", classify_gemini)
    graph.add_node("classify_ollama", classify_ollama)
    graph.add_conditional_edges(START, classification_router)
    graph.add_edge("classify_openai", END)
    graph.add_edge("classify_gemini", END)
    graph.add_edge("classify_ollama", END)
    return graph.compile()


def build_extraction_graph() -> StateGraph:
    """Build the LangGraph for ensemble extraction."""
    graph = StateGraph(ExtractionState)
    graph.add_node("extract_openai", extract_openai)
    graph.add_node("extract_gemini", extract_gemini)
    graph.add_node("extract_ollama", extract_ollama)
    graph.add_conditional_edges(START, extraction_router)
    graph.add_edge("extract_openai", END)
    graph.add_edge("extract_gemini", END)
    graph.add_edge("extract_ollama", END)
    return graph.compile()


class Orchestrator:
    """Ensemble orchestrator with parallel extraction via LangGraph."""

    def __init__(self):
        global _llm_clients
        _llm_clients = LLMClients()
        self.clients = _llm_clients.clients
        self.classification_graph = build_classification_graph()
        self.extraction_graph = build_extraction_graph()
        self.merger = FieldMerger()
        logger.info("Orchestrator initialized (LangGraph + Ensemble)")

    def classify_ensemble(self, text: str, prompt_template: str) -> Tuple[str, float, List[str]]:
        """Classify using ensemble approach via LangGraph."""
        initial_state: ClassificationState = {"text": text, "prompt_template": prompt_template, "results": []}
        final_state = self.classification_graph.invoke(initial_state)
        results = final_state.get("results", [])

        if not results:
            return "unknown", 0.0, []

        providers_used = [r["provider"] for r in results]
        doc_types = [r["doc_type"] for r in results]
        confidences = [r["confidence"] for r in results]

        most_common_type = Counter(doc_types).most_common(1)[0][0]
        avg_confidence = sum(confidences) / len(confidences)

        logger.info(f"ENSEMBLE VOTE: {most_common_type} ({avg_confidence:.1%})")
        return most_common_type, avg_confidence, providers_used

    def ensemble_extract(self, text: str, doc_type: str, prompt_template: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract using all models in parallel via LangGraph, then merge results."""
        initial_state: ExtractionState = {
            "text": text,
            "doc_type": doc_type,
            "prompt_template": prompt_template,
            "results": [],
        }
        final_state = self.extraction_graph.invoke(initial_state)
        results = final_state.get("results", [])

        if not results:
            logger.error("All models failed extraction")
            return {}, []

        providers_used = [r["provider"] for r in results]
        field_results = [r["fields"] for r in results]
        merged = self.merger.merge_extractions(field_results)

        logger.info(f"ENSEMBLE COMPLETE: Merged {len(results)} results from {providers_used}")
        return merged, providers_used
