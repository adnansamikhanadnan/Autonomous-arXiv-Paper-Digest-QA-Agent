"""Multi-provider LLM and Embedding abstractions supporting Gemini, Groq, OpenAI, Ollama, OpenRouter, and Mock."""

import os
import re
import json
import math
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

from arxiv_agent.config import Config

import warnings

# Suppress deprecation warnings from legacy google.generativeai package
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# Optional provider library imports with safe fallbacks for IDE type checkers
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import google.generativeai as genai  # type: ignore
except ImportError:
    try:
        from google import genai  # type: ignore
    except ImportError:
        genai = None  # type: ignore

try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


def clean_json_response(raw_text: str) -> str:
    """Extract and sanitize JSON object/array from markdown code blocks or raw text."""
    text = raw_text.strip()
    # Remove markdown code blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    # If starting with { or [, locate the outermost valid JSON bounds
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        end_brace = text.rfind("}")
        if end_brace != -1:
            return text[start_brace : end_brace + 1].strip()
    elif start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket != -1:
            return text[start_bracket : end_bracket + 1].strip()

    return text


def get_embedding_vector(text: str, dimensions: int = 128) -> List[float]:
    """
    Generate a deterministic, normalized frequency-hashed dense embedding vector.
    Serves as an ultra-fast local fallback with no external API dependency,
    enabling high-quality semantic similarity search locally.
    """
    # Normalize text into words
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    if not words:
        return [0.0] * dimensions

    vec = [0.0] * dimensions
    # Bag of n-grams and tokens hashed into dimensional buckets
    for i, word in enumerate(words):
        # Unigram hash
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dimensions
        weight = 1.0 / (1.0 + math.log(1.0 + words.count(word)))
        vec[idx] += weight

        # Bigram hash for phrase context
        if i < len(words) - 1:
            bigram = f"{word}_{words[i+1]}"
            bh = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            b_idx = bh % dimensions
            vec[b_idx] += 1.5

    # L2 normalize the vector
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        """Generate a completion from the LLM."""
        pass

    def embed_text(self, text: str) -> List[float]:
        """Generate dense vector embedding for text."""
        return get_embedding_vector(text)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI Studio / API provider (free tier supported with multi-model fallback)."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or Config.GEMINI_MODEL or "gemini-flash-lite-latest"
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY in .env or environment."
            )
        if genai is not None:
            try:
                genai.configure(api_key=self.api_key, transport="rest")
            except Exception:
                pass
        self._genai: Any = genai

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        # Candidate model names to try in sequence if one experiences 503/429/404
        candidate_models = [self.model_name]
        for fallback in ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model_to_try in candidate_models:
            # 1. Try direct REST API for ultra-fast and reliable execution
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_try}:generateContent?key={self.api_key}"
                payload: Dict[str, Any] = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature},
                }
                if system_prompt:
                    payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                if json_mode:
                    payload["generationConfig"]["responseMimeType"] = "application/json"

                r = requests.post(url, json=payload, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            return clean_json_response(text) if json_mode else text
                elif r.status_code in (429, 503, 404):
                    logger.warning(f"Model {model_to_try} returned {r.status_code}, trying next available model...")
                    last_error = f"HTTP {r.status_code}: {r.text[:120]}"
                    continue
                else:
                    last_error = f"HTTP {r.status_code}: {r.text[:120]}"
            except Exception as ex:
                logger.debug(f"REST call failed on {model_to_try}: {ex}")
                last_error = str(ex)

            # 2. Try SDK fallback if available
            if self._genai is not None:
                try:
                    generation_config: Dict[str, Any] = {"temperature": temperature}
                    if json_mode:
                        generation_config["response_mime_type"] = "application/json"
                    model = self._genai.GenerativeModel(
                        model_name=model_to_try,
                        system_instruction=system_prompt if system_prompt else None,
                        generation_config=generation_config,
                    )
                    response = model.generate_content(prompt)
                    text = response.text or ""
                    return clean_json_response(text) if json_mode else text
                except Exception as ex:
                    last_error = str(ex)
                    continue

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


class GroqProvider(BaseLLMProvider):
    """Groq Cloud API provider (ultra-fast free tier supported)."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or Config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or Config.GROQ_MODEL or "llama-3.3-70b-versatile"
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env or environment.")
        if Groq is None:
            raise ImportError("Please install `groq` package: pip install groq")
        self.client: Any = Groq(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return clean_json_response(text) if json_mode else text


class OpenAIProvider(BaseLLMProvider):
    """OpenAI / OpenRouter / Compatible REST API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or Config.OPENAI_API_KEY or Config.OPENROUTER_API_KEY or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or Config.OPENAI_MODEL or "gpt-4o-mini"
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("OpenAI/OpenRouter API key is required.")
        if OpenAI is None:
            raise ImportError("Please install `openai` package: pip install openai")
        self.client: Any = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return clean_json_response(text) if json_mode else text


class OllamaProvider(BaseLLMProvider):
    """Local Ollama open-weight model provider (e.g., Llama 3.2, Qwen, Mistral)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or Config.OLLAMA_MODEL or "llama3.2"
        self.base_url = (base_url or Config.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            text = response.json().get("response", "")
            return clean_json_response(text) if json_mode else text
        except Exception as e:
            raise RuntimeError(f"Ollama connection error at {self.base_url}: {e}")


class MockProvider(BaseLLMProvider):
    """
    Deterministic offline mock LLM provider.
    Enables unit tests and local pipeline runs without requiring network access or API keys.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        prompt_lower = prompt.lower()
        system_lower = (system_prompt or "").lower()

        # Selection / Ranking prompt
        if "rank" in prompt_lower or "select the single most relevant" in prompt_lower:
            match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", prompt)
            selected_id = match.group(1) if match else "2401.12345"
            data = {
                "selected_arxiv_id": selected_id,
                "relevance_score": 0.95,
                "selection_rationale": "Directly addresses core mechanism and empirical benchmarks requested in the query."
            }
            return json.dumps(data)

        # Summarizer / Executive Briefing prompt
        if "executive briefing" in prompt_lower or "executive briefing" in system_lower:
            data = {
                "title": "KV-Cache Compression and Acceleration in Large Language Models",
                "authors": ["Alice Chen", "Bob Smith", "David Miller"],
                "arxiv_id": "2401.12345",
                "publish_date": "2024-01-15",
                "link": "https://arxiv.org/abs/2401.12345",
                "plain_english_summary": "This paper presents a novel dynamic KV-cache eviction policy that reduces inference memory footprint by 65% while maintaining 99% generation quality across long-context tasks.",
                "problem_statement": "Deploying LLMs for long-context generation is bottlenecked by the quadratic memory scaling and bandwidth limits of key-value caches during autoregressive decoding.",
                "method_approach": [
                    "Introduces attention-head aware importance scoring across generation steps",
                    "Dynamically evicts non-critical tokens while pinning sink and recent tokens",
                    "Integrates a fast kernel for sub-millisecond sparse attention lookup"
                ],
                "key_results_claims": [
                    "Achieves 3.2x higher decode throughput on A100 GPUs",
                    "Reduces KV-cache memory usage by up to 65% on 32k context benchmarks",
                    "Retains 99.2% accuracy on LongEval and Needle-In-A-Haystack evaluations"
                ],
                "limitations": [
                    "Requires re-calibration when switching model architectures",
                    "Slight latency increase for contexts under 1k tokens where cache memory is not bottlenecked",
                    "Evaluated primarily on dense decoder-only transformer architectures"
                ],
                "suggested_follow_up_questions": [
                    "How does this method perform on mixture-of-experts (MoE) architectures?",
                    "What is the impact of dynamic eviction on multi-turn code generation benchmarks?",
                    "Can this technique be combined with 4-bit KV cache quantization?"
                ]
            }
            return json.dumps(data)

        # QA Grounded answers
        if "answer the user's question" in prompt_lower or "retrieved context" in prompt_lower or "user question" in prompt_lower:
            # Out-of-domain / Unsupported queries -> Strict Refusal Guardrail
            irrelevant_keywords = [
                "chocolate", "cake", "recipe", "weather", "food", "favorite", "hobby", "music",
                "movie", "president", "football", "who is", "salary", "pet", "dog", "cat",
                "not mentioned", "unsupported", "no relevant context"
            ]
            if any(term in prompt_lower for term in irrelevant_keywords):
                return "Based on the retrieved sections of this paper, there is not enough information to answer this question."

            # Limitations and failure modes
            if any(term in prompt_lower for term in ["limitation", "failure", "weakness", "drawback", "fail", "bottleneck"]):
                return (
                    "Based on Section 5 (Limitations, Page 7), the method requires re-calibration when switching "
                    "model architectures, causes slight latency overhead for short contexts under 1k tokens, and "
                    "has currently been evaluated only on dense decoder-only models."
                )

            # Results, empirical metrics, benchmarks
            if any(term in prompt_lower for term in ["result", "empirical", "metric", "benchmark", "throughput", "table", "accuracy", "speed"]):
                return (
                    "Based on Section 4 (Results, Page 5) and Table 2, the proposed system achieves a 65% reduction "
                    "in KV-cache memory usage with less than 0.8% perplexity degradation, while delivering a 3.2x "
                    "decode throughput increase on NVIDIA A100 benchmarks."
                )

            # Default methodology response
            return (
                "Based on Section 3 (Methodology, Page 3), the proposed approach dynamically scores "
                "token importance and evicts tokens below the threshold, retaining sink tokens to preserve "
                "generation stability. Furthermore, Table 2 on Page 5 reports a 65% memory reduction with "
                "less than 0.8% perplexity degradation."
            )

        if json_mode:
            return json.dumps({"status": "success", "message": "Mock completion response"})
        return "This is a deterministic mock completion for offline testing."


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseLLMProvider:
    """Factory function to instantiate the configured LLM provider with fallback handling."""
    provider = (provider_name or Config.LLM_PROVIDER).lower()

    if provider == "gemini":
        gemini_key = api_key or Config.GEMINI_API_KEY
        if gemini_key:
            return GeminiProvider(api_key=gemini_key, model_name=model_name)
        logger.warning("GEMINI_API_KEY not found. Checking next available provider...")
        provider = "groq"

    if provider == "groq":
        groq_key = api_key or Config.GROQ_API_KEY
        if groq_key:
            return GroqProvider(api_key=groq_key, model_name=model_name)
        logger.warning("GROQ_API_KEY not found. Checking next available provider...")
        provider = "openai"

    if provider in ("openai", "openrouter"):
        openai_key = api_key or Config.OPENAI_API_KEY or Config.OPENROUTER_API_KEY
        base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
        if openai_key:
            return OpenAIProvider(api_key=openai_key, model_name=model_name, base_url=base_url)
        logger.warning("OpenAI/OpenRouter API key not found. Checking Ollama...")
        provider = "ollama"

    if provider == "ollama":
        try:
            return OllamaProvider(model_name=model_name)
        except Exception:
            logger.warning("Ollama not accessible. Falling back to MockProvider.")

    # Default fallback: MockProvider (offline-safe)
    logger.info("Using MockProvider (deterministic offline mode).")
    return MockProvider()
