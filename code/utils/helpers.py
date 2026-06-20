import re
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Compiled list of allowed enums for strict normalizations
ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"
}

ALLOWED_SEVERITIES = {"none", "low", "medium", "high", "unknown"}

ALLOWED_CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
    "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"
}

ALLOWED_LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base", "body", "unknown"
}

ALLOWED_PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal", "label", "contents", "item", "unknown"
}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
}

def normalize_enum_val(val: str, allowed_set: set, default: str = "unknown") -> str:
    """Normalizes string inputs to match exact values in the allowed_set."""
    if not val:
        return default
    
    # Strip spaces and convert to lowercase
    s = val.strip().lower()
    
    # Check exact match
    if s in allowed_set:
        return s
    
    # Replace spaces, dashes and punctuation with underscores
    s_cleaned = re.sub(r'[\s\-]+', '_', s)
    s_cleaned = re.sub(r'[^\w]', '', s_cleaned)
    
    if s_cleaned in allowed_set:
        return s_cleaned
        
    # Check if any allowed value is in the cleaned string
    for allowed in allowed_set:
        if allowed != "unknown" and allowed != "none":
            if allowed in s_cleaned or s_cleaned in allowed:
                return allowed
                
    return default

def normalize_issue_type(val: str) -> str:
    return normalize_enum_val(val, ALLOWED_ISSUE_TYPES, "unknown")

def normalize_severity(val: str) -> str:
    return normalize_enum_val(val, ALLOWED_SEVERITIES, "unknown")

def normalize_object_part(val: str, claim_object: str) -> str:
    claim_object = claim_object.strip().lower()
    if claim_object == "car":
        return normalize_enum_val(val, ALLOWED_CAR_PARTS, "unknown")
    elif claim_object == "laptop":
        return normalize_enum_val(val, ALLOWED_LAPTOP_PARTS, "unknown")
    elif claim_object == "package":
        return normalize_enum_val(val, ALLOWED_PACKAGE_PARTS, "unknown")
    else:
        # Generic check
        all_parts = ALLOWED_CAR_PARTS | ALLOWED_LAPTOP_PARTS | ALLOWED_PACKAGE_PARTS
        return normalize_enum_val(val, all_parts, "unknown")

def normalize_risk_flag(val: str) -> str:
    if not val:
        return "none"
    s = val.strip().lower()
    
    # If the description contains negation words, ignore it
    negations = ["no ", "not ", "none", "false", "ok", "okay", "clear", "normal"]
    for neg in negations:
        if neg in s:
            return "none"
            
    # Check exact match first after basic cleaning
    s_cleaned = re.sub(r'[\s\-]+', '_', s)
    s_cleaned = re.sub(r'[^\w]', '', s_cleaned)
    if s_cleaned in ALLOWED_RISK_FLAGS:
        return s_cleaned
        
    # Check synonym map
    risk_synonyms = {
        "blurry_image": ["blurry", "blur", "blurry_image"],
        "cropped_or_obstructed": ["cropped", "obstructed", "crop", "obstruction", "cropped_or_obstructed"],
        "low_light_or_glare": ["glare", "dark", "low_light", "low_light_or_glare", "reflection"],
        "wrong_angle": ["wrong_angle", "bad_angle", "angle"],
        "wrong_object": ["wrong_object", "different_object", "wrong_item"],
        "wrong_object_part": ["wrong_object_part", "wrong_part", "different_part"],
        "damage_not_visible": ["damage_not_visible", "no_damage_visible", "not_visible"],
        "claim_mismatch": ["claim_mismatch", "mismatch"],
        "possible_manipulation": ["manipulation", "edited", "photoshopped", "manipulated", "possible_manipulation"],
        "non_original_image": ["non_original", "screenshot", "stock_image", "non_original_image"],
        "text_instruction_present": ["text_instruction", "text_present", "instruction_text", "text_instruction_present"]
    }
    for flag, synonyms in risk_synonyms.items():
        for syn in synonyms:
            if syn in s_cleaned:
                return flag
                
    return "none"

class DeterministicLookup:
    def __init__(
        self,
        history_path: str = "dataset/user_history.csv",
        requirements_path: str = "dataset/evidence_requirements.csv"
    ):
        try:
            self.history_df = pd.read_csv(history_path)
            # Index by user_id
            self.history_df.set_index("user_id", inplace=True)
        except Exception as e:
            self.history_df = None
            print(f"Warning: Failed to load user history from {history_path}: {e}")

        try:
            self.requirements_df = pd.read_csv(requirements_path)
        except Exception as e:
            self.requirements_df = None
            print(f"Warning: Failed to load evidence requirements from {requirements_path}: {e}")

    def lookup_user_history(self, user_id: str) -> Dict[str, Any]:
        """Looks up user claim history stats and risk indicators."""
        default_res = {
            "past_claim_count": 0,
            "accept_claim": 0,
            "manual_review_claim": 0,
            "rejected_claim": 0,
            "last_90_days_claim_count": 0,
            "history_flags": "none",
            "history_summary": "New user with no prior claim history",
            "user_history_risk_flagged": False
        }
        
        if self.history_df is None or user_id not in self.history_df.index:
            return default_res
            
        row = self.history_df.loc[user_id]
        
        # Determine if history has risk flag
        history_flags = str(row.get("history_flags", "none")).strip().lower()
        has_risk = "risk" in history_flags or int(row.get("rejected_claim", 0)) > 2
        
        return {
            "past_claim_count": int(row.get("past_claim_count", 0)),
            "accept_claim": int(row.get("accept_claim", 0)),
            "manual_review_claim": int(row.get("manual_review_claim", 0)),
            "rejected_claim": int(row.get("rejected_claim", 0)),
            "last_90_days_claim_count": int(row.get("last_90_days_claim_count", 0)),
            "history_flags": history_flags,
            "history_summary": str(row.get("history_summary", "Prior claims record")),
            "user_history_risk_flagged": has_risk
        }

    def lookup_evidence_requirements(self, claim_object: str, claimed_issue_type: str) -> str:
        """Looks up the minimum evidence requirement text from evidence_requirements.csv."""
        if self.requirements_df is None:
            return "The submitted images should provide visual evidence that is usable and relevant to the claim."
            
        claim_object = claim_object.strip().lower()
        claimed_issue_type = claimed_issue_type.strip().lower()
        
        # Classify issue families matching the csv "applies_to" categories
        applies_to_val = "general claim review"
        if claim_object == "car":
            if claimed_issue_type in ["dent", "scratch"]:
                applies_to_val = "dent or scratch"
            elif claimed_issue_type in ["crack", "broken_part", "missing_part", "glass_shatter"]:
                applies_to_val = "crack, broken, or missing part"
            elif claimed_issue_type in ["unknown", "none"]:
                applies_to_val = "vehicle identity or orientation"
        elif claim_object == "laptop":
            if claimed_issue_type in ["crack", "stain", "none"]:
                applies_to_val = "screen, keyboard, or trackpad"
            else:
                applies_to_val = "hinge, lid, corner, body, or port"
        elif claim_object == "package":
            if claimed_issue_type in ["crushed_packaging", "torn_packaging"]:
                applies_to_val = "crushed, torn, or seal damage"
            elif claimed_issue_type in ["water_damage", "stain"]:
                applies_to_val = "water, stain, or label damage"
            else:
                applies_to_val = "contents or inner item"
                
        # Query requirements DataFrame
        mask = (
            (self.requirements_df["claim_object"].str.lower() == claim_object) |
            (self.requirements_df["claim_object"].str.lower() == "all")
        ) & (self.requirements_df["applies_to"].str.lower() == applies_to_val)
        
        filtered = self.requirements_df[mask]
        if not filtered.empty:
            return str(filtered.iloc[0]["minimum_image_evidence"])
            
        # Fallback to REQ_REVIEW_TRUST or generic review
        trust_mask = self.requirements_df["requirement_id"] == "REQ_REVIEW_TRUST"
        if not self.requirements_df[trust_mask].empty:
            return str(self.requirements_df[trust_mask].iloc[0]["minimum_image_evidence"])
            
        return "The submitted images should show the claimed object and relevant part clearly enough to inspect the claimed condition."

# Global metrics dictionary to track VLM/LLM calls, cache stats, and tokens
METRICS = {
    "cache_hits": 0,
    "cache_misses": 0,
    "context_agent_calls": 0,
    "vision_agent_calls": 0,
    "deterministic_extraction_successes": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "used_fallback_estimates": False,
    "active_text_requests": 0,
    "active_vision_requests": 0
}

def reset_metrics():
    for k in METRICS:
        if isinstance(METRICS[k], bool):
            METRICS[k] = False
        elif isinstance(METRICS[k], float):
            METRICS[k] = 0.0
        else:
            METRICS[k] = 0

def attempt_deterministic_extraction(user_claim: str, claim_object: str) -> Dict[str, Any]:
    """
    Attempts to extract issue_type and object_part deterministically from transcript
    using keyword synonym maps and exact boundaries. Returns a structured dictionary
    indicating success or failure with reasons.
    """
    text = user_claim.strip().lower()
    
    # Define synonyms maps for parts based on claim_object
    part_synonyms = {}
    if claim_object == "car":
        part_synonyms = {
            "front_bumper": ["front bumper", "front side bumper", "front bumper area"],
            "rear_bumper": ["rear bumper", "back bumper", "back bumper area", "rear bumper area"],
            "door": ["door", "side door", "car door", "door panel"],
            "hood": ["hood", "bonnet", "car hood"],
            "windshield": ["windshield", "windscreen", "front glass", "front windshield"],
            "side_mirror": ["side mirror", "wing mirror", "rearview mirror", "mirror"],
            "headlight": ["headlight", "front light", "headlamp", "head lamp"],
            "taillight": ["taillight", "tail light", "back light", "tail lamp"],
            "fender": ["fender"],
            "quarter_panel": ["quarter panel"],
            "body": ["body", "body panel"]
        }
    elif claim_object == "laptop":
        part_synonyms = {
            "screen": ["screen", "display", "monitor", "glass"],
            "keyboard": ["keyboard", "keys", "buttons"],
            "trackpad": ["trackpad", "touchpad", "mousepad"],
            "hinge": ["hinge", "hinges"],
            "lid": ["lid", "top cover", "outer cover"],
            "corner": ["corner", "corners"],
            "port": ["port", "ports", "usb", "hdmi", "charging port"],
            "base": ["base", "bottom", "bottom cover"],
            "body": ["body", "chassis"]
        }
    elif claim_object == "package":
        part_synonyms = {
            "package_corner": ["package corner", "corner of the box", "corner of box", "box corner"],
            "package_side": ["package side", "side of the box", "side of box", "box side"],
            "seal": ["seal", "tape", "sealing tape"],
            "label": ["label", "shipping label", "address label"],
            "contents": ["contents", "item inside", "inside", "product"],
            "item": ["item", "product", "device"],
            "box": ["box", "carton", "package"]
        }
    else:
        fallback_reason = "unsupported_synonym"
        return {
            "success": False,
            "issue_matches_found": [],
            "part_matches_found": [],
            "fallback_reason": fallback_reason
        }

    # Synonym map for issues
    issue_synonyms = {
        "dent": ["dent", "dented", "bump"],
        "scratch": ["scratch", "scratches", "scraped", "scrape", "scratched"],
        "crack": ["crack", "cracked", "fractured"],
        "glass_shatter": ["shatter", "shattered", "broken glass", "smashed glass"],
        "broken_part": ["broken", "broke", "damaged"],
        "missing_part": ["missing", "not inside", "lost", "empty", "no item"],
        "torn_packaging": ["torn", "tear", "ripped", "opened", "open"],
        "crushed_packaging": ["crushed", "crush", "smashed box", "flattened"],
        "water_damage": ["water", "wet", "liquid", "spill", "spilled"],
        "stain": ["stain", "stains", "stained"]
    }

    # Find matching parts
    matched_parts = []
    for part, synonyms in part_synonyms.items():
        for syn in synonyms:
            pattern = r'\b' + re.escape(syn) + r'\b'
            if re.search(pattern, text):
                matched_parts.append(part)
                break

    # Find matching issues
    matched_issues = []
    for issue, synonyms in issue_synonyms.items():
        for syn in synonyms:
            pattern = r'\b' + re.escape(syn) + r'\b'
            if re.search(pattern, text):
                matched_issues.append(issue)
                break

    # For diagnostics and logging
    issues_found = matched_issues.copy()
    parts_found = matched_parts.copy()

    # Prefer the most specific detected part rather than treating container-level parts as ambiguity
    if claim_object == "package" and "box" in matched_parts and len(matched_parts) > 1:
        matched_parts.remove("box")
    elif claim_object == "car" and "body" in matched_parts and len(matched_parts) > 1:
        matched_parts.remove("body")
    elif claim_object == "laptop" and "body" in matched_parts and len(matched_parts) > 1:
        matched_parts.remove("body")

    selected_issue = matched_issues[0] if len(matched_issues) == 1 else None
    selected_part = matched_parts[0] if len(matched_parts) == 1 else None

    # Determine fallback reason if not successful
    if not (selected_issue and selected_part):
        if len(issues_found) == 0:
            fallback_reason = "no_issue_match"
        elif len(parts_found) == 0:
            fallback_reason = "no_part_match"
        elif len(issues_found) > 1:
            fallback_reason = "ambiguous_issue"
        elif len(parts_found) > 1:
            fallback_reason = "ambiguous_part"
        else:
            fallback_reason = "confidence_too_low"

    # Log diagnostics
    logger.debug(f"issues_found: {issues_found}")
    logger.debug(f"parts_found: {parts_found}")
    logger.debug(f"selected_issue: {selected_issue}")
    logger.debug(f"selected_part: {selected_part}")

    if selected_issue and selected_part:
        summary = f"Deterministic extraction: {selected_issue} on {selected_part}"
        METRICS["deterministic_extraction_successes"] += 1
        return {
            "success": True,
            "claim_object": claim_object,
            "claimed_issue_type": selected_issue,
            "claimed_object_part": selected_part,
            "claim_intent_summary": summary
        }
    
    return {
        "success": False,
        "issue_matches_found": issues_found,
        "part_matches_found": parts_found,
        "fallback_reason": fallback_reason
    }


