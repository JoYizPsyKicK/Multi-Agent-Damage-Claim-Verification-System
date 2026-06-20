# Evaluation Report

This report evaluates the accuracy and performance of the multi-agent damage claim verification system.

## 1. Architecture Overview
The system implements a damage claim verification pipeline featuring:
- **Deterministic Context Extraction**: A fast-path keyword/phrase lookup to identify the claimed issue type and object part without LLM invocation.
- **Context Agent**: A local text LLM (`qwen3:8b`) to extract details when deterministic extraction is inconclusive.
- **Vision Agent**: A local multi-modal VLM (`qwen2.5vl:7b`) to inspect and audit visual evidence.
- **Deterministic Adjudicator**: A rules-based decision engine that adjudicates claim status, evidence standard fulfillment, and risk indicators.
- **Deterministic Justification Generator**: A template-based natural language summary generator.
- **SQLite Cache**: A SQLite database in WAL mode to speed up repetitive queries.

## 2. Dataset Summary
- **Dataset Evaluated**: `dataset/sample_claims.csv`
- **Total Claims**: 20 claims

## 3. Runtime Statistics
- **Total Pipeline Runtime**: 0.47 seconds
- **Average Runtime per Claim**: 0.02 seconds

## 4. Token Usage Statistics
- **Total Prompt Tokens**: 0
- **Total Completion Tokens**: 0
- **Total Processed Tokens**: 0

## 5. Cache & Model Call Statistics
- **Cache Hits**: 43
- **Cache Misses**: 0
- **Context Agent Calls**: 0
- **Vision Agent Calls**: 0
- **Deterministic Extraction Successes**: 6

## 6. Estimated Cloud Cost
*Based on GPT-4o equivalent pricing: $5.00/1M input tokens, $15.00/1M output tokens.*
- **Estimated Cloud Cost**: $0.0000 USD
- **Actual Local Inference Cost**: $0.00 USD (Local Ollama Execution)

## 7. Accuracy Metrics
- **Claim Status Accuracy**: 95.0% (19/20)
- **Issue Type Accuracy**: 85.0% (17/20)
- **Object Part Accuracy**: 100.0% (20/20)
- **Severity Accuracy**: 90.0% (18/20)

## 8. Limitations
- **VLM Speed**: Local VLM inference remains the primary pipeline latency bottleneck.
- **Synonym Coverage**: Deterministic extraction relies on regex keyword matching, requiring clear user claims.

## 9. Future Improvements
- **Synonym Expansion**: Add support for more colloquial terms in deterministic extraction.
- **Quantization Optimization**: Use smaller quantized VLM models to minimize inference latency.

## 10. Detailed Mismatches (4 / 20)
- **User**: user_005 (car)
  - **Mismatches**: issue: 'scratch' vs 'none', severity: 'low' vs 'none'
  - **Claim**: Customer: I want to file this as bumper damage. | Support: Can you tell me what happened? | Customer: The car was tapped...

- **User**: user_010 (laptop)
  - **Mismatches**: status: 'supported' vs 'not_enough_information', severity: 'medium' vs 'unknown'
  - **Claim**: Customer: The laptop no longer opens smoothly. | Support: What seems to be wrong mechanically? | Customer: The hinge are...

- **User**: user_011 (laptop)
  - **Mismatches**: issue: 'stain' vs 'water_damage'
  - **Claim**: Customer: I spilled water near my laptop while working. | Support: What part was affected? | Customer: It went over the ...

- **User**: user_018 (laptop)
  - **Mismatches**: issue: 'crack' vs 'glass_shatter'
  - **Claim**: Customer: I am not sure if this should be a repair claim or a replacement claim, so I wanted to ask first. | Support: We...

