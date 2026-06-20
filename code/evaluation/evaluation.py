import os
import sys
import time
import asyncio
import logging
import pandas as pd
from typing import Dict, Any, List

# Add the parent directory of code/ to sys.path so we can import from code/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from code.utils.helpers import METRICS, reset_metrics, normalize_issue_type, normalize_object_part, normalize_severity, DeterministicLookup
from code.utils.cache import PipelineCache
from code.agents.context_agent import ContextAgent
from code.agents.vision_agent import VisionAgent
from code.agents.justification_agent import JustificationAgent
from code.main import process_single_claim
from code.config import settings

# Setup Logging for Evaluation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EvaluationFramework")

async def run_evaluation(disable_cache=False, workers=settings.max_concurrent_text_tasks):
    sample_claims_path = "dataset/sample_claims.csv"
    user_history_path = "dataset/user_history.csv"
    requirements_path = "dataset/evidence_requirements.csv"
    
    if not os.path.exists(sample_claims_path):
        logger.error(f"Sample claims file not found at {sample_claims_path}")
        return

    # Reset tracking metrics
    reset_metrics()
    
    df_samples = pd.read_csv(sample_claims_path)
    logger.info(f"Loaded {len(df_samples)} evaluation samples.")
    
    # Initialize pipeline modules (disable cache if specified)
    cache_handler = None if disable_cache else PipelineCache()
    lookup = DeterministicLookup(history_path=user_history_path, requirements_path=requirements_path)
    
    # Use configuration models for evaluation
    text_model = settings.text_model
    vision_model = settings.vision_model
    
    # Define separate semaphores for text vs vision models
    text_semaphore = asyncio.Semaphore(settings.max_concurrent_text_tasks)
    vision_semaphore = asyncio.Semaphore(settings.max_concurrent_vision_tasks)
    
    context_agent = ContextAgent(model_name=text_model, cache_handler=cache_handler, semaphore=text_semaphore)
    vision_agent = VisionAgent(model_name=vision_model, cache_handler=cache_handler, semaphore=vision_semaphore)
    justification_agent = JustificationAgent()
    
    start_time = time.time()
    
    # Schedule all claims concurrently using concurrency control semaphore
    claims_semaphore = asyncio.Semaphore(workers)
    
    async def sem_process_single_claim(row):
        async with claims_semaphore:
            return await process_single_claim(row, lookup, context_agent, vision_agent, justification_agent)
            
    tasks = [
        sem_process_single_claim(row)
        for _, row in df_samples.iterrows()
    ]
    
    predictions = await asyncio.gather(*tasks)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Write predictions to output
    df_preds = pd.DataFrame(predictions)
    preds_out_path = "code/evaluation/sample_predictions.csv"
    os.makedirs(os.path.dirname(preds_out_path), exist_ok=True)
    df_preds.to_csv(preds_out_path, index=False)
    
    # Perform field-by-field accuracy checks
    total_rows = len(df_samples)
    matches = {
        "claim_status": 0,
        "issue_type": 0,
        "object_part": 0,
        "severity": 0
    }
    mismatches = []
    
    for idx, row in df_samples.iterrows():
        pred_row = df_preds.iloc[idx]
        
        # Normalize fields for fair comparison
        true_status = str(row["claim_status"]).strip().lower()
        pred_status = str(pred_row["claim_status"]).strip().lower()
        
        true_issue = normalize_issue_type(str(row["issue_type"]))
        pred_issue = normalize_issue_type(str(pred_row["issue_type"]))
        
        true_part = normalize_object_part(str(row["object_part"]), str(row["claim_object"]))
        pred_part = normalize_object_part(str(pred_row["object_part"]), str(row["claim_object"]))
        
        true_sev = normalize_severity(str(row["severity"]))
        pred_sev = normalize_severity(str(pred_row["severity"]))
        
        if true_status == pred_status:
            matches["claim_status"] += 1
        if true_issue == pred_issue:
            matches["issue_type"] += 1
        if true_part == pred_part:
            matches["object_part"] += 1
        if true_sev == pred_sev:
            matches["severity"] += 1
            
        diff = []
        if true_status != pred_status:
            diff.append(f"status: '{true_status}' vs '{pred_status}'")
        if true_issue != pred_issue:
            diff.append(f"issue: '{true_issue}' vs '{pred_issue}'")
        if true_part != pred_part:
            diff.append(f"part: '{true_part}' vs '{pred_part}'")
        if true_sev != pred_sev:
            diff.append(f"severity: '{true_sev}' vs '{pred_sev}'")
            
        if diff:
            mismatches.append((str(row["user_id"]), str(row["claim_object"]), diff, str(row["user_claim"])))
            
    accuracies = {k: (v / total_rows) * 100 for k, v in matches.items()}

    # Pricing from config settings
    input_cost_M = settings.reference_input_cost_per_million
    output_cost_M = settings.reference_output_cost_per_million
    
    # Calculate estimated cloud cost
    estimated_cloud_cost = (
        (METRICS["prompt_tokens"] * input_cost_M / 1_000_000.0) + 
        (METRICS["completion_tokens"] * output_cost_M / 1_000_000.0)
    )
    
    avg_runtime = runtime / total_rows if total_rows > 0 else 0.0

    # Generate Markdown Report
    report_md = f"""# Evaluation Report

This report evaluates the accuracy and performance of the multi-agent damage claim verification system.

## 1. Architecture Overview
The system implements a damage claim verification pipeline featuring:
- **Deterministic Context Extraction**: A fast-path keyword/phrase lookup to identify the claimed issue type and object part without LLM invocation.
- **Context Agent**: A local text LLM (`{text_model}`) to extract details when deterministic extraction is inconclusive.
- **Vision Agent**: A local multi-modal VLM (`{vision_model}`) to inspect and audit visual evidence.
- **Deterministic Adjudicator**: A rules-based decision engine that adjudicates claim status, evidence standard fulfillment, and risk indicators.
- **Deterministic Justification Generator**: A template-based natural language summary generator.
- **SQLite Cache**: A SQLite database in WAL mode to speed up repetitive queries.

## 2. Dataset Summary
- **Dataset Evaluated**: `dataset/sample_claims.csv`
- **Total Claims**: {total_rows} claims

## 3. Runtime Statistics
- **Total Pipeline Runtime**: {runtime:.2f} seconds
- **Average Runtime per Claim**: {avg_runtime:.2f} seconds

## 4. Token Usage Statistics
- **Total Prompt Tokens**: {METRICS["prompt_tokens"]:,}{" (Estimates)" if METRICS["used_fallback_estimates"] else ""}
- **Total Completion Tokens**: {METRICS["completion_tokens"]:,}{" (Estimates)" if METRICS["used_fallback_estimates"] else ""}
- **Total Processed Tokens**: {METRICS["prompt_tokens"] + METRICS["completion_tokens"]:,}

## 5. Cache & Model Call Statistics
- **Cache Hits**: {METRICS["cache_hits"]}
- **Cache Misses**: {METRICS["cache_misses"]}
- **Context Agent Calls**: {METRICS["context_agent_calls"]}
- **Vision Agent Calls**: {METRICS["vision_agent_calls"]}
- **Deterministic Extraction Successes**: {METRICS["deterministic_extraction_successes"]}

## 6. Estimated Cloud Cost
*Based on GPT-4o equivalent pricing: ${input_cost_M:.2f}/1M input tokens, ${output_cost_M:.2f}/1M output tokens.*
- **Estimated Cloud Cost**: ${estimated_cloud_cost:.4f} USD
- **Actual Local Inference Cost**: $0.00 USD (Local Ollama Execution)

## 7. Accuracy Metrics
- **Claim Status Accuracy**: {accuracies["claim_status"]:.1f}% ({matches["claim_status"]}/{total_rows})
- **Issue Type Accuracy**: {accuracies["issue_type"]:.1f}% ({matches["issue_type"]}/{total_rows})
- **Object Part Accuracy**: {accuracies["object_part"]:.1f}% ({matches["object_part"]}/{total_rows})
- **Severity Accuracy**: {accuracies["severity"]:.1f}% ({matches["severity"]}/{total_rows})

## 8. Limitations
- **VLM Speed**: Local VLM inference remains the primary pipeline latency bottleneck.
- **Synonym Coverage**: Deterministic extraction relies on regex keyword matching, requiring clear user claims.

## 9. Future Improvements
- **Synonym Expansion**: Add support for more colloquial terms in deterministic extraction.
- **Quantization Optimization**: Use smaller quantized VLM models to minimize inference latency.
"""

    if mismatches:
        mismatches_str = f"\n## 10. Detailed Mismatches ({len(mismatches)} / {total_rows})\n"
        for user_id, obj, diff, claim in mismatches:
            mismatches_str += f"- **User**: {user_id} ({obj})\n"
            mismatches_str += f"  - **Mismatches**: {', '.join(diff)}\n"
            mismatches_str += f"  - **Claim**: {claim[:120]}...\n\n"
        report_md += mismatches_str
    else:
        report_md += "\n## 10. Detailed Mismatches\nNo mismatches found. Perfect accuracy!\n"

    report_path = "code/evaluation/evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info(f"Evaluation report written successfully to {report_path}")
    print(f"\n--- Evaluation Results ---")
    print(f"Claim Status Accuracy: {accuracies['claim_status']:.1f}%")
    print(f"Issue Type Accuracy: {accuracies['issue_type']:.1f}%")
    print(f"Object Part Accuracy: {accuracies['object_part']:.1f}%")
    print(f"Severity Accuracy: {accuracies['severity']:.1f}%")
    print(f"--------------------------")
    print(f"Total Runtime: {runtime:.2f} seconds")
    print(f"Average Runtime: {avg_runtime:.2f} seconds")
    print(f"--------------------------")
    print(f"Cache hits: {METRICS['cache_hits']}")
    print(f"Cache misses: {METRICS['cache_misses']}")
    print(f"Context Agent calls: {METRICS['context_agent_calls']}")
    print(f"Vision Agent calls: {METRICS['vision_agent_calls']}")
    print(f"Deterministic extraction successes: {METRICS['deterministic_extraction_successes']}")
    print(f"--------------------------")
    print(f"Total Prompt Tokens: {METRICS['prompt_tokens']:,}{' (Estimates)' if METRICS['used_fallback_estimates'] else ''}")
    print(f"Total Completion Tokens: {METRICS['completion_tokens']:,}{' (Estimates)' if METRICS['used_fallback_estimates'] else ''}")
    print(f"Estimated Cloud Cost: ${estimated_cloud_cost:.4f} USD")
    print(f"--------------------------\n")
    
    if mismatches:
        print(f"--- Detailed Mismatches ({len(mismatches)} / {total_rows}) ---")
        for user_id, obj, diff, claim in mismatches:
            print(f"User: {user_id} ({obj})")
            print(f"  Mismatches: {', '.join(diff)}")
            print(f"  Claim: {claim[:80]}...")
            print("-" * 40)
        print("\n")
    else:
        print("--- Detailed Mismatches ---")
        print("No mismatches found. Perfect accuracy!\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation Framework")
    parser.add_argument("--disable-cache", action="store_true", help="Bypass and disable pipeline cache")
    parser.add_argument("--workers", type=int, default=settings.max_concurrent_text_tasks, help="Number of concurrent workers")
    args = parser.parse_args()
    asyncio.run(run_evaluation(disable_cache=args.disable_cache, workers=args.workers))
