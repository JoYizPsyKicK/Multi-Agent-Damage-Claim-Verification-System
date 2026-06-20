import logging
from typing import Optional, Dict, Any, List
from code.schemas.claim_schemas import ContextBrief, VisualEvidenceReport

logger = logging.getLogger(__name__)

class JustificationAgent:
    def __init__(self):
        pass

    def generate_justification(
        self,
        brief: ContextBrief,
        vision_report: VisualEvidenceReport,
        user_history: Dict[str, Any],
        claim_status: str,
        evidence_standard_met: bool,
        evidence_standard_met_reason: str,
        supporting_images: List[str],
        risk_flags: List[str],
        user_id: Optional[str] = None
    ) -> str:
        """
        Deterministic, template-based justification generator.
        Replaces the LLM to run in 0ms with 100% reliability and grounding.
        """
        # Defensive validation
        if not isinstance(brief, ContextBrief):
            raise TypeError(f"Expected brief to be ContextBrief, got {type(brief).__name__}")
        
        required_fields = ["claim_object", "claimed_object_part", "claimed_issue_type", "claim_intent_summary"]
        for field in required_fields:
            if not hasattr(brief, field):
                raise ValueError(f"ContextBrief object is missing required field: '{field}'")
            if getattr(brief, field) is None:
                raise ValueError(f"ContextBrief required field '{field}' is None")

        claimed_part = brief.claimed_object_part
        claimed_issue = brief.claimed_issue_type
        claim_object = brief.claim_object
        
        visible_part = vision_report.visible_object_part
        visible_issue = vision_report.visible_issue_type
        
        # Format list of supporting images
        images_str = ", ".join(supporting_images) if supporting_images and supporting_images != ["none"] else ""
        
        # Risk flags check
        active_risk_flags = [f for f in risk_flags if f not in ["none", "manual_review_required"]]
        risk_str = f" Risk flags identified: {', '.join(active_risk_flags)}." if active_risk_flags else ""
        
        if claim_status == "supported":
            justification = (
                f"Claim for {claim_object} {claimed_part} {claimed_issue} is supported. "
                f"Visual evidence in image(s) {images_str} confirms visible {visible_issue} on the {visible_part}."
            )
        elif claim_status == "contradicted":
            if visible_issue == "none":
                justification = (
                    f"Claim for {claim_object} {claimed_part} {claimed_issue} is contradicted. "
                    f"Visual analysis indicates no visible damage on the {claimed_part}."
                )
            elif visible_part != claimed_part and visible_part not in ["unknown", "none"]:
                justification = (
                    f"Claim for {claim_object} {claimed_part} {claimed_issue} is contradicted. "
                    f"Visual evidence shows a different part ({visible_part}) is damaged instead of the claimed part."
                )
            else:
                justification = (
                    f"Claim for {claim_object} {claimed_part} {claimed_issue} is contradicted. "
                    f"Visual analysis shows a different issue ({visible_issue}) on the {visible_part}."
                )
        else: # not_enough_information
            if not vision_report.valid_image:
                justification = (
                    f"Claim for {claim_object} {claimed_part} {claimed_issue} cannot be verified. "
                    f"All submitted images are invalid or unreadable."
                )
            elif visible_part == "unknown" or visible_part == "none":
                justification = (
                    f"Claim for {claim_object} {claimed_part} {claimed_issue} cannot be verified. "
                    f"The claimed part is not visible in any submitted image."
                )
            else:
                justification = (
                    f"Claim for {claim_object} {claimed_part} {claimed_issue} cannot be verified. "
                    f"{evidence_standard_met_reason}"
                )
                
        # Append risk warning if manual review is flagged
        if "manual_review_required" in risk_flags:
            justification += f" Escalate to manual review.{risk_str}"
            
        return justification
