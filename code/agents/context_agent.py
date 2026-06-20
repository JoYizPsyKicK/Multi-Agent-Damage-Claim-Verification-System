import json
import time
import logging
import ollama
from typing import Optional, Dict, Any, Tuple
from code.schemas.claim_schemas import ContextBrief
from code.utils.retry import async_retry
from code.utils.helpers import normalize_issue_type, normalize_object_part, METRICS
from code.utils.client_wrapper import unified_chat_completion
from code.config import settings

logger = logging.getLogger(__name__)

class ContextAgent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_handler: Optional[Any] = None,
        semaphore: Optional[Any] = None
    ):
        self.model_name = model_name if model_name is not None else settings.text_model
        self.cache_handler = cache_handler
        self.semaphore = semaphore
        if settings.provider == "ollama":
            self.client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=settings.ollama_request_timeout)
        else:
            self.client = None

    def _generate_prompt(self, claim_object: str, user_claim: str) -> Tuple[str, str]:
        # Define allowed parts based on object
        allowed_parts = []
        if claim_object == "car":
            allowed_parts = ["front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror", "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"]
        elif claim_object == "laptop":
            allowed_parts = ["screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"]
        elif claim_object == "package":
            allowed_parts = ["box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"]
        else:
            allowed_parts = ["unknown"]

        allowed_issues = ["dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part", "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"]

        system_prompt = (
            f"Extract the damage claim from the customer chat transcript.\n"
            f"Object type: '{claim_object}'.\n"
            f"Allowed parts: {allowed_parts}.\n"
            f"Allowed issues: {allowed_issues}.\n"
            "Respond in JSON format: {\"claimed_issue_type\": \"...\", \"claimed_object_part\": \"...\", \"claim_intent_summary\": \"...\"}"
        )

        user_prompt = f"Transcript:\n{user_claim}"
        return system_prompt, user_prompt

    @async_retry(max_retries=3, base_delay=1.0)
    async def extract_context(self, user_claim: str, claim_object: str, user_id: Optional[str] = None) -> ContextBrief:
        system_prompt, user_prompt = self._generate_prompt(claim_object, user_claim)
        
        # Check cache
        cache_key = f"{system_prompt}\n{user_prompt}"
        if self.cache_handler:
            cached = self.cache_handler.get_context(cache_key, self.model_name)
            if cached:
                logger.info(f"[TEXT] Cache Hit for {user_id or 'unknown'}")
                METRICS["cache_hits"] += 1
                try:
                    if isinstance(cached, dict) and "claim_object" not in cached:
                        cached["claim_object"] = claim_object
                    return ContextBrief.model_validate(cached)
                except Exception as e:
                    logger.warning(f"Failed to load cached ContextBrief: {e}. Re-generating...")
                    METRICS["cache_hits"] -= 1

        METRICS["cache_misses"] += 1
        METRICS["context_agent_calls"] += 1

        # Unified Call
        queue_start = time.time()
        logger.info(f"[TEXT] Queued {user_id or 'unknown'}")
        
        class DummySemaphore:
            async def __aenter__(self): return self
            async def __aexit__(self, exc_type, exc_val, exc_tb): pass

        sem = self.semaphore if self.semaphore else DummySemaphore()
        queue_wait_time = 0.0
        async with sem:
            if self.semaphore:
                queue_wait_time = time.time() - queue_start
            METRICS["active_text_requests"] += 1
            active_count = METRICS["active_text_requests"]
            logger.info(f"[TEXT] Acquired semaphore {user_id or 'unknown'} (active={active_count}, queue_wait={queue_wait_time:.4f}s)")
            start_time = time.time()
            try:
                response = await unified_chat_completion(
                    model_name=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.text_model_temperature,
                    num_ctx=settings.text_model_num_ctx,
                    top_p=settings.text_model_top_p,
                    response_format="json",
                    ollama_client=self.client
                )
                active_exec_time = time.time() - start_time
                logger.info(f"[TEXT] Completed LLM call {user_id or 'unknown'} in {active_exec_time:.4f}s")
            finally:
                METRICS["active_text_requests"] -= 1
                active_count = METRICS["active_text_requests"]
                logger.info(f"[TEXT] Released semaphore {user_id or 'unknown'} (active={active_count})")

        # Record tokens metrics
        prompt_tokens = response.get("prompt_tokens", 0)
        completion_tokens = response.get("completion_tokens", 0)
        
        if prompt_tokens > 0 and completion_tokens > 0:
            active_prompt = prompt_tokens
            active_completion = completion_tokens
        else:
            # Fallback estimates for Context Agent
            active_prompt = 300
            active_completion = 50
            METRICS["used_fallback_estimates"] = True
            logger.info(f"[TEXT] API stats unavailable for ContextAgent. Using estimates.")

        METRICS["prompt_tokens"] += active_prompt
        METRICS["completion_tokens"] += active_completion

        content = response["content"]
        logger.debug(f"Context Agent raw output: {content}")
        
        raw_json = json.loads(content)
        
        # Normalize and validate raw JSON fields before Pydantic parsing to prevent validation crashes
        raw_json["claim_object"] = claim_object
        raw_json["claimed_issue_type"] = normalize_issue_type(raw_json.get("claimed_issue_type", "unknown"))
        raw_json["claimed_object_part"] = normalize_object_part(raw_json.get("claimed_object_part", "unknown"), claim_object)
        
        brief = ContextBrief.model_validate(raw_json)
        
        # Save to cache
        if self.cache_handler:
            self.cache_handler.set_context(cache_key, self.model_name, brief.model_dump())
            
        return brief
