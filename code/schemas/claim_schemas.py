from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

# Allowed issue types
IssueType = Literal[
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
]

# Allowed risk flags
RiskFlag = Literal[
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
]

# Allowed severities
Severity = Literal["none", "low", "medium", "high", "unknown"]

# Allowed parts across all objects
CarPart = Literal[
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"
]

LaptopPart = Literal[
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"
]

PackagePart = Literal[
    "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"
]

class ContextBrief(BaseModel):
    claim_object: str = Field(
        ..., description="The type of object under claim (e.g. car, laptop, package)"
    )
    claimed_issue_type: IssueType = Field(
        ..., description="The category of damage the user claims in the chat transcript"
    )
    claimed_object_part: str = Field(
        ..., description="The part of the object claimed to be damaged (normalized to snake_case, e.g. rear_bumper, screen, lid, box)"
    )
    claim_intent_summary: str = Field(
        ..., description="A clear, concise summary of the claim being made by the user"
    )

class ImageAnalysis(BaseModel):
    image_id: str = Field(
        ..., description="The filename of the image being analyzed (e.g. img_1)"
    )
    valid_image: bool = Field(
        ..., description="True if the image is clear and usable; False if it is blurry, dark, or not readable"
    )
    quality_flags: List[RiskFlag] = Field(
        default_factory=list,
        description="List of risk and quality flags found in this image. Use 'none' if there are no quality issues."
    )
    visible_issue_type: IssueType = Field(
        ..., description="The visible damage category in this image"
    )
    visible_object_part: str = Field(
        ..., description="The object part visible in this image (e.g. front_bumper, windshield, screen, box)"
    )
    severity: Severity = Field(
        ..., description="Visual severity of the damage visible in this image"
    )
    justification: str = Field(
        ..., description="A detailed visual justification explanation for the observations on this specific image"
    )

class VisualEvidenceReport(BaseModel):
    valid_image: bool = Field(
        ..., description="True if at least one image in the set is valid and usable for review"
    )
    image_quality_flags: List[RiskFlag] = Field(
        default_factory=list,
        description="Aggregated risk and quality flags across all analyzed images"
    )
    visible_issue_type: IssueType = Field(
        ..., description="Aggregated primary visible issue type across the valid images"
    )
    visible_object_part: str = Field(
        ..., description="Aggregated primary visible object part across the valid images"
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="List of image IDs (filenames without extensions) that show the claimed damage"
    )
    severity: Severity = Field(
        ..., description="Overall visual severity across all images"
    )
    visual_analysis_justification: str = Field(
        ..., description="Aggregated visual justification explaining the findings"
    )

class FinalClaimDecision(BaseModel):
    evidence_standard_met: bool = Field(
        ..., description="True if the visual evidence is sufficient to evaluate the claim"
    )
    evidence_standard_met_reason: str = Field(
        ..., description="Short explanation of why the evidence was or was not sufficient"
    )
    risk_flags: List[RiskFlag] = Field(
        ..., description="All relevant risk/quality flags aggregated for the final report"
    )
    issue_type: IssueType = Field(
        ..., description="The final adjudicated visible issue type"
    )
    object_part: str = Field(
        ..., description="The final adjudicated visible object part"
    )
    claim_status: Literal["supported", "contradicted", "not_enough_information"] = Field(
        ..., description="The final decision: supported, contradicted, or not_enough_information"
    )
    claim_status_justification: str = Field(
        ..., description="Concise explanation grounded in the visual evidence and history"
    )
    supporting_image_ids: List[str] = Field(
        ..., description="List of supporting image IDs (e.g. ['img_1']), or ['none']"
    )
    valid_image: bool = Field(
        ..., description="True if the image set is usable; False otherwise"
    )
    severity: Severity = Field(
        ..., description="Overall adjudicated severity"
    )
