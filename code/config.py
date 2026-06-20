import os
import logging
from typing import Any
from dotenv import load_dotenv


# Try to find and load .env from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

class AppSettings:
    def __init__(self):
        # Centralized configurations with fail-fast validations
        self.ollama_base_url = self._get_required_env("OLLAMA_BASE_URL", "http://localhost:11434")
        if not (self.ollama_base_url.startswith("http://") or self.ollama_base_url.startswith("https://")):
            raise ValueError(f"OLLAMA_BASE_URL must start with http:// or https://, got: {self.ollama_base_url}")
            
        self.vision_model = self._get_required_env("VISION_MODEL", "qwen2.5-vl")
        self.text_model = self._get_required_env("TEXT_MODEL", "qwen2.5")
        
        self.ollama_request_timeout = self._get_typed_env("OLLAMA_REQUEST_TIMEOUT", int, 300)
        self.max_concurrent_vision_tasks = self._get_typed_env("MAX_CONCURRENT_VISION_TASKS", int, 2)
        self.max_concurrent_text_tasks = self._get_typed_env("MAX_CONCURRENT_TEXT_TASKS", int, 10)
        
        self.prompt_version = self._get_required_env("PROMPT_VERSION", "v1")
        
        self.provider = self._get_required_env("PROVIDER", "ollama").strip().lower()
        if self.provider not in ["ollama", "openai", "gemini", "anthropic", "openrouter"]:
            raise ValueError(f"PROVIDER must be one of: ollama, openai, gemini, anthropic, openrouter. Got: {self.provider}")
            
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        # Validate that required keys are present
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when PROVIDER is 'openai'")
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when PROVIDER is 'gemini'")
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when PROVIDER is 'anthropic'")
        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when PROVIDER is 'openrouter'")
        
        # Cache paths resolve relative to project root if relative
        cache_db_raw = self._get_required_env("CACHE_DB_PATH", "code/cache/pipeline_cache.db")
        if not os.path.isabs(cache_db_raw):
            self.cache_db_path = os.path.abspath(os.path.join(project_root, cache_db_raw))
        else:
            self.cache_db_path = cache_db_raw
            
        self.log_level = self._get_required_env("LOG_LEVEL", "INFO").upper()
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL. Got: {self.log_level}")

        # Model Generation parameters
        self.text_model_temperature = self._get_typed_env("TEXT_MODEL_TEMPERATURE", float, 0.0)
        self.vision_model_temperature = self._get_typed_env("VISION_MODEL_TEMPERATURE", float, 0.0)
        self.text_model_num_ctx = self._get_typed_env("TEXT_MODEL_NUM_CTX", int, 8192)
        self.vision_model_num_ctx = self._get_typed_env("VISION_MODEL_NUM_CTX", int, 8192)
        self.text_model_top_p = self._get_typed_env("TEXT_MODEL_TOP_P", float, 0.1)
        self.vision_model_top_p = self._get_typed_env("VISION_MODEL_TOP_P", float, 0.1)
        
        self.reference_input_cost_per_million = self._get_typed_env("REFERENCE_INPUT_COST_PER_MILLION", float, 5.0)
        self.reference_output_cost_per_million = self._get_typed_env("REFERENCE_OUTPUT_COST_PER_MILLION", float, 15.0)


    def _get_required_env(self, key: str, default: str = None) -> str:
        val = os.getenv(key, default)
        if val is None:
            raise RuntimeError(f"Missing required environment variable: {key}")
        return val

    def _get_typed_env(self, key: str, type_fn: type, default: Any = None) -> Any:
        raw_val = os.getenv(key)
        if raw_val is None:
            if default is not None:
                return default
            raise RuntimeError(f"Missing required environment variable: {key}")
        try:
            return type_fn(raw_val)
        except ValueError:
            raise ValueError(f"Environment variable {key} must be of type {type_fn.__name__}. Got: {raw_val}")

# Single global settings object
settings = AppSettings()
