import os
import json
import time
import logging
import ollama
from typing import List, Optional, Dict, Any, Tuple
from code.schemas.claim_schemas import ImageAnalysis, VisualEvidenceReport
from code.utils.retry import async_retry
from code.utils.helpers import (
    normalize_issue_type, normalize_object_part, normalize_severity,
    normalize_risk_flag, METRICS
)
from code.utils.client_wrapper import unified_chat_completion
from code.config import settings

logger = logging.getLogger(__name__)

class VisionAgent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_handler: Optional[Any] = None,
        semaphore: Optional[Any] = None
    ):
        self.model_name = model_name if model_name is not None else settings.vision_model
        self.cache_handler = cache_handler
        self.semaphore = semaphore
        if settings.provider == "ollama":
            self.client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=settings.ollama_request_timeout)
        else:
            self.client = None

    def _resolve_image_path(self, relative_path: str) -> str:
        """Resolves raw CSV paths to actual local system paths."""
        # Handles "images/sample/..." -> "dataset/images/sample/..."
        path = relative_path.strip()
        if not path.startswith("dataset/") and (path.startswith("images/") or path.startswith("dataset_images/")):
            path = os.path.join("dataset", path)
        return os.path.abspath(path)

    def _generate_prompt(self, claim_object: str, claimed_part: str, claimed_issue: str) -> Tuple[str, str]:
        # Allowed enums based on object type
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
            f"Analyze the image for a {claim_object} damage claim (claimed: {claimed_issue} on {claimed_part}).\n"
            f"Allowed visible parts: {allowed_parts}.\n"
            f"Allowed visible issues: {allowed_issues}.\n"
            "Allowed quality_flags: [\"blurry_image\", \"cropped_or_obstructed\", \"low_light_or_glare\", \"wrong_angle\", \"wrong_object\", \"wrong_object_part\", \"damage_not_visible\", \"claim_mismatch\", \"possible_manipulation\", \"non_original_image\", \"text_instruction_present\"].\n"
            "Respond in JSON format: {\"valid_image\": bool, \"quality_flags\": [], \"visible_issue_type\": \"...\", \"visible_object_part\": \"...\", \"severity\": \"none/low/medium/high/unknown\", \"justification\": \"...\"}"
        )
        
        user_prompt = "Forensically inspect the attached image and output the structured JSON analysis."
        return system_prompt, user_prompt

    @async_retry(max_retries=3, base_delay=2.0)
    async def analyze_single_image(
        self, image_path: str, claim_object: str, claimed_part: str, claimed_issue: str, user_id: Optional[str] = None
    ) -> ImageAnalysis:
        resolved_path = self._resolve_image_path(image_path)
        image_id = os.path.splitext(os.path.basename(resolved_path))[0]
        
        if not os.path.exists(resolved_path):
            logger.warning(f"Image path does not exist: {resolved_path}")
            return ImageAnalysis(
                image_id=image_id,
                valid_image=False,
                quality_flags=["damage_not_visible"],
                visible_issue_type="unknown",
                visible_object_part="unknown",
                severity="unknown",
                justification=f"File not found: {image_path}"
            )

        system_prompt, user_prompt = self._generate_prompt(claim_object, claimed_part, claimed_issue)
        # Check cache
        if self.cache_handler:
            cached = self.cache_handler.get_vision([resolved_path], system_prompt + user_prompt, self.model_name)
            if cached:
                logger.info(f"[VISION] Cache Hit for {user_id or 'unknown'} ({image_id})")
                METRICS["cache_hits"] += 1
                try:
                    return ImageAnalysis.model_validate(cached)
                except Exception as e:
                    logger.warning(f"Failed to load cached ImageAnalysis: {e}. Re-generating...")
                    METRICS["cache_hits"] -= 1

        METRICS["cache_misses"] += 1
        METRICS["vision_agent_calls"] += 1

        # Ollama multi-modal call
        queue_start = time.time()
        logger.info(f"[VISION] Queued {user_id or 'unknown'} ({image_id})")
        
        class DummySemaphore:
            async def __aenter__(self): return self
            async def __aexit__(self, exc_type, exc_val, exc_tb): pass

        sem = self.semaphore if self.semaphore else DummySemaphore()
        queue_wait_time = 0.0
        async with sem:
            if self.semaphore:
                queue_wait_time = time.time() - queue_start
            METRICS["active_vision_requests"] += 1
            active_count = METRICS["active_vision_requests"]
            logger.info(f"[VISION] Acquired semaphore {user_id or 'unknown'} ({image_id}) (active={active_count}, queue_wait={queue_wait_time:.4f}s)")
            start_time = time.time()
            try:
                response = await unified_chat_completion(
                    model_name=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.vision_model_temperature,
                    num_ctx=settings.vision_model_num_ctx,
                    top_p=settings.vision_model_top_p,
                    response_format="json",
                    images=[resolved_path],
                    ollama_client=self.client
                )
                active_exec_time = time.time() - start_time
                logger.info(f"[VISION] Completed VLM call {user_id or 'unknown'} ({image_id}) in {active_exec_time:.4f}s")
            finally:
                METRICS["active_vision_requests"] -= 1
                active_count = METRICS["active_vision_requests"]
                logger.info(f"[VISION] Released semaphore {user_id or 'unknown'} ({image_id}) (active={active_count})")
        
        # Record tokens metrics
        prompt_tokens = response.get("prompt_tokens", 0)
        completion_tokens = response.get("completion_tokens", 0)
        
        if prompt_tokens > 0 and completion_tokens > 0:
            active_prompt = prompt_tokens
            active_completion = completion_tokens
        else:
            # Fallback estimates for Vision Agent
            active_prompt = 1500
            active_completion = 120
            METRICS["used_fallback_estimates"] = True
            logger.info(f"[VISION] API stats unavailable for VisionAgent. Using estimates.")

        METRICS["prompt_tokens"] += active_prompt
        METRICS["completion_tokens"] += active_completion

        content = response["content"]
        logger.debug(f"Vision Agent raw output: {content}")
        
        raw_json = json.loads(content)
        
        # Normalize and validate raw JSON values
        raw_json["image_id"] = image_id
        raw_json["valid_image"] = bool(raw_json.get("valid_image", True))
        
        q_flags = raw_json.get("quality_flags", [])
        if not isinstance(q_flags, list):
            q_flags = [q_flags] if q_flags else []
        normalized_q_flags = [normalize_risk_flag(f) for f in q_flags if f]
        raw_json["quality_flags"] = normalized_q_flags if normalized_q_flags else ["none"]
        
        raw_json["visible_issue_type"] = normalize_issue_type(raw_json.get("visible_issue_type", "unknown"))
        raw_json["visible_object_part"] = normalize_object_part(raw_json.get("visible_object_part", "unknown"), claim_object)
        raw_json["severity"] = normalize_severity(raw_json.get("severity", "unknown"))
        
        analysis = ImageAnalysis.model_validate(raw_json)
        
        # Cache results
        if self.cache_handler:
            self.cache_handler.set_vision([resolved_path], system_prompt + user_prompt, self.model_name, analysis.model_dump())
            
        return analysis

    def aggregate_analyses(
        self, analyses: List[ImageAnalysis], claim_object: str, claimed_part: str, claimed_issue: str
    ) -> VisualEvidenceReport:
        """Deterministic Visual Aggregation Layer (Step 3/7)"""
        if not analyses:
            return VisualEvidenceReport(
                valid_image=False,
                image_quality_flags=["damage_not_visible"],
                visible_issue_type="unknown",
                visible_object_part="unknown",
                supporting_image_ids=["none"],
                severity="unknown",
                visual_analysis_justification="No images submitted."
            )

        # Usable images check
        valid_analyses = [a for a in analyses if a.valid_image]
        if not valid_analyses:
            # Combine quality flags
            all_flags = set()
            for a in analyses:
                all_flags.update(a.quality_flags)
            all_flags.discard("none")
            
            return VisualEvidenceReport(
                valid_image=False,
                image_quality_flags=list(all_flags) if all_flags else ["blurry_image"],
                visible_issue_type="unknown",
                visible_object_part="unknown",
                supporting_image_ids=["none"],
                severity="unknown",
                visual_analysis_justification="All submitted images are invalid or unreadable."
            )

        # Aggregate risk/quality flags
        agg_flags = set()
        for a in analyses:
            agg_flags.update(a.quality_flags)
        agg_flags.discard("none")

        # Find supporting images (where VLM found matching damage on the claimed part)
        supporting_images = []
        for a in valid_analyses:
            # Normalize and check match
            part_match = a.visible_object_part == claimed_part
            issue_match = a.visible_issue_type == claimed_issue
            
            # If matching part and issue is not none/unknown, it supports!
            if part_match and issue_match and a.visible_issue_type not in ["none", "unknown"]:
                supporting_images.append(a.image_id)
        
        # Determine visible issue and part to report
        # Prioritize matching the claimed part and issue, then matching the claimed part with any damage,
        # then matching the claimed part, then any image with damage, and finally the first valid image.
        primary_analysis = None
        
        # 1. Match both claimed part and claimed issue
        for a in valid_analyses:
            if a.visible_object_part == claimed_part and a.visible_issue_type == claimed_issue:
                primary_analysis = a
                break
                
        # 2. Match claimed part and has any visible damage
        if not primary_analysis:
            for a in valid_analyses:
                if a.visible_object_part == claimed_part and a.visible_issue_type not in ["none", "unknown"]:
                    primary_analysis = a
                    break
                    
        # 3. Match claimed part
        if not primary_analysis:
            for a in valid_analyses:
                if a.visible_object_part == claimed_part:
                    primary_analysis = a
                    break
                    
        # 4. Any image with visible damage
        if not primary_analysis:
            damaged_analyses = [a for a in valid_analyses if a.visible_issue_type not in ["none", "unknown"]]
            if damaged_analyses:
                primary_analysis = damaged_analyses[0]
                
        # 5. Fallback to first valid analysis
        if not primary_analysis:
            primary_analysis = valid_analyses[0]

        # Severity aggregation: highest severity found on valid images
        severity_hierarchy = {"none": 0, "unknown": 1, "low": 2, "medium": 3, "high": 4}
        max_severity = "none"
        for a in valid_analyses:
            if severity_hierarchy.get(a.severity, 0) > severity_hierarchy.get(max_severity, 0):
                max_severity = a.severity

        # Combined justifications
        justifications = []
        for a in analyses:
            status_str = "Usable" if a.valid_image else "Unusable"
            justifications.append(f"[{a.image_id} - {status_str}]: {a.justification}")
        combined_justification = " | ".join(justifications)

        return VisualEvidenceReport(
            valid_image=True,
            image_quality_flags=list(agg_flags) if agg_flags else ["none"],
            visible_issue_type=primary_analysis.visible_issue_type,
            visible_object_part=primary_analysis.visible_object_part,
            supporting_image_ids=supporting_images if supporting_images else ["none"],
            severity=max_severity,
            visual_analysis_justification=combined_justification
        )
