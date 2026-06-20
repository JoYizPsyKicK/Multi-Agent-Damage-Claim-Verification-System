import os
import time
import argparse
import asyncio
import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from code.schemas.claim_schemas import ImageAnalysis, ContextBrief
from code.utils.cache import PipelineCache
from code.utils.helpers import DeterministicLookup, METRICS, reset_metrics, attempt_deterministic_extraction
from code.agents.context_agent import ContextAgent
from code.agents.vision_agent import VisionAgent
from code.agents.adjudicator import Adjudicator
from code.agents.justification_agent import JustificationAgent
from code.config import settings

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("PipelineOrchestrator")

async def process_single_claim(
    row: Dict[str, Any],
    lookup: DeterministicLookup,
    context_agent: ContextAgent,
    vision_agent: VisionAgent,
    justification_agent: JustificationAgent
) -> Dict[str, Any]:
    """Processes a single claim through all pipeline stages concurrently."""
    user_id = row.get("user_id")
    image_paths_raw = row.get("image_paths", "")
    user_claim = row.get("user_claim", "")
    claim_object = row.get("claim_object", "car")
    
    # Split semicolon-separated image paths
    image_paths = [p.strip() for p in str(image_paths_raw).split(";") if p.strip()]
    
    logger.info(f"Starting processing for claim: User {user_id}, Object {claim_object}")
    start_time = time.time()
    try:
        # 1. Deterministic Pre-processing Lookups (Instant, In-Memory)
        user_history = lookup.lookup_user_history(user_id)
        # 2. Context Agent (Text LLM or Deterministic Extraction)
        det_context = attempt_deterministic_extraction(user_claim, claim_object)
        if det_context and det_context.get("success"):
            logger.info(f"[{user_id}] Deterministic context extraction succeeded.")
            brief = ContextBrief.model_validate(det_context)
        else:
            issues = det_context.get("issue_matches_found", []) if det_context else []
            parts = det_context.get("part_matches_found", []) if det_context else []
            reason = det_context.get("fallback_reason", "unknown") if det_context else "unknown"
            logger.info(
                f"[{user_id}] Deterministic extraction failed.\n"
                f"Issue matches: {json.dumps(issues)}\n"
                f"Part matches: {json.dumps(parts)}\n"
                f"Fallback reason: {reason}"
            )
            brief = await context_agent.extract_context(user_claim, claim_object, user_id=user_id)
        # 3. Vision Agent (Multi-Modal per-image analysis)
        analyses_tasks = [
            vision_agent.analyze_single_image(
                img_path, claim_object, brief.claimed_object_part, brief.claimed_issue_type, user_id=user_id
            )
            for img_path in image_paths
        ]
        # Run image audits concurrently
        analyses = await asyncio.gather(*analyses_tasks, return_exceptions=True)
        
        cleaned_analyses = []
        for i, res in enumerate(analyses):
            if isinstance(res, Exception):
                logger.error(f"[{user_id}] Image analysis failed for {image_paths[i]}: {res}")
                img_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                cleaned_analyses.append(
                    ImageAnalysis(
                        image_id=img_name,
                        valid_image=False,
                        quality_flags=["damage_not_visible"],
                        visible_issue_type="unknown",
                        visible_object_part="unknown",
                        severity="unknown",
                        justification=f"Analysis failed: {res}"
                    )
                )
            else:
                cleaned_analyses.append(res)

        # 4. Deterministic Aggregation
        vision_report = vision_agent.aggregate_analyses(
            cleaned_analyses, claim_object, brief.claimed_object_part, brief.claimed_issue_type
        )
        
        # 5. Deterministic Adjudication (Instant, In-Memory)
        validation_start = time.time()
        (
            evidence_standard_met,
            evidence_standard_met_reason,
            risk_flags,
            issue_type,
            object_part,
            claim_status,
            supporting_images,
            valid_image,
            severity
        ) = Adjudicator.adjudicate(brief, vision_report, user_history, user_id=user_id)
        logger.info(f"Validation: {time.time() - validation_start:.4f}s")
        
        # 6. Justification Agent (Deterministic)
        req_text = lookup.lookup_evidence_requirements(claim_object, brief.claimed_issue_type)
        evidence_met_reason_full = f"{evidence_standard_met_reason} Requirement: {req_text}"
        
        claim_justification = justification_agent.generate_justification(
            brief=brief,
            vision_report=vision_report,
            user_history=user_history,
            claim_status=claim_status,
            evidence_standard_met=evidence_standard_met,
            evidence_standard_met_reason=evidence_met_reason_full,
            supporting_images=supporting_images,
            risk_flags=risk_flags,
            user_id=user_id
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Claim processed successfully in {elapsed:.2f}s (User {user_id})")
        
        return {
            "user_id": user_id,
            "image_paths": image_paths_raw,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": str(evidence_standard_met).lower(),
            "evidence_standard_met_reason": evidence_standard_met_reason,
            "risk_flags": ";".join(risk_flags),
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": claim_justification,
            "supporting_image_ids": ";".join(supporting_images),
            "valid_image": str(valid_image).lower(),
            "severity": severity
        }
        
    except Exception as e:
        logger.error(f"Failed to process claim for User {user_id}: {e}", exc_info=True)
        return {
            "user_id": user_id,
            "image_paths": image_paths_raw,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": f"System processing error: {e}",
            "risk_flags": "manual_review_required",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": f"Claim verification failed due to internal pipeline error: {e}",
            "supporting_image_ids": "none",
            "valid_image": "false",
            "severity": "unknown"
        }

async def async_main(args):
    # Load input claims CSV
    if not os.path.exists(args.claims):
        logger.error(f"Input claims file not found: {args.claims}")
        return
        
    df_claims = pd.read_csv(args.claims)
    logger.info(f"Loaded {len(df_claims)} claims to process.")
    
    # Initialize cache and lookups (disable cache if specified)
    cache_handler = None if args.disable_cache else PipelineCache()
    lookup = DeterministicLookup(history_path=args.history, requirements_path=args.requirements)
    
    # Reset tracking metrics
    reset_metrics()
    
    # Define separate semaphores for text vs vision models
    text_semaphore = asyncio.Semaphore(settings.max_concurrent_text_tasks)
    vision_semaphore = asyncio.Semaphore(settings.max_concurrent_vision_tasks)
    
    # Initialize Agents
    context_agent = ContextAgent(model_name=args.text_model, cache_handler=cache_handler, semaphore=text_semaphore)
    vision_agent = VisionAgent(model_name=args.vision_model, cache_handler=cache_handler, semaphore=vision_semaphore)
    justification_agent = JustificationAgent()
    
    start_time = time.time()
    
    # Schedule all claims concurrently using --workers concurrency control
    claims_semaphore = asyncio.Semaphore(args.workers)
    
    async def sem_process_single_claim(row):
        async with claims_semaphore:
            return await process_single_claim(row, lookup, context_agent, vision_agent, justification_agent)
            
    tasks = [
        sem_process_single_claim(row)
        for _, row in df_claims.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_runtime = total_runtime / len(df_claims) if len(df_claims) > 0 else 0
    
    # Cost calculations
    input_cost_M = settings.reference_input_cost_per_million
    output_cost_M = settings.reference_output_cost_per_million
    
    estimated_cloud_cost = (
        (METRICS["prompt_tokens"] * input_cost_M / 1_000_000.0) + 
        (METRICS["completion_tokens"] * output_cost_M / 1_000_000.0)
    )

    # Build output DataFrame and ensure correct column sequencing
    df_out = pd.DataFrame(results)
    cols_order = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity"
    ]
    df_out = df_out[cols_order]
    
    # Create output dir if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    
    logger.info("=========================================")
    logger.info("Pipeline Execution Summary:")
    logger.info(f"- Total claims processed: {len(df_claims)}")
    logger.info(f"- Total runtime: {total_runtime:.2f} seconds")
    logger.info(f"- Average runtime per claim: {avg_runtime:.2f} seconds")
    logger.info(f"- Cache hits: {METRICS['cache_hits']}")
    logger.info(f"- Cache misses: {METRICS['cache_misses']}")
    logger.info(f"- Context Agent calls: {METRICS['context_agent_calls']}")
    logger.info(f"- Vision Agent calls: {METRICS['vision_agent_calls']}")
    logger.info(f"- Deterministic extraction successes: {METRICS['deterministic_extraction_successes']}")
    logger.info(f"- Total Prompt Tokens: {METRICS['prompt_tokens']}{' (Estimates)' if METRICS['used_fallback_estimates'] else ''}")
    logger.info(f"- Total Completion Tokens: {METRICS['completion_tokens']}{' (Estimates)' if METRICS['used_fallback_estimates'] else ''}")
    logger.info(f"- Estimated Cloud Cost (equivalent): ${estimated_cloud_cost:.4f} USD")
    logger.info("=========================================")
def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Damage Claim Verification System")
    parser.add_argument("--claims", type=str, default="dataset/claims.csv", help="Path to input claims CSV")
    parser.add_argument("--history", type=str, default="dataset/user_history.csv", help="Path to user history CSV")
    parser.add_argument("--requirements", type=str, default="dataset/evidence_requirements.csv", help="Path to evidence requirements CSV")
    parser.add_argument("--output", type=str, default="output.csv", help="Path to write output predictions CSV")
    parser.add_argument("--text-model", type=str, default=settings.text_model, help="Ollama model for text tasks")
    parser.add_argument("--vision-model", type=str, default=settings.vision_model, help="Ollama model for vision tasks")
    parser.add_argument("--workers", type=int, default=settings.max_concurrent_text_tasks, help="Number of concurrent claim workers")
    parser.add_argument("--disable-cache", action="store_true", help="Bypass and disable pipeline cache")
    
    args = parser.parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
