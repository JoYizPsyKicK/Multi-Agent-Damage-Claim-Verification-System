import unittest
import sys
import os

# Add the parent directory of code/ to sys.path so we can import from code/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from code.utils.helpers import attempt_deterministic_extraction
from code.schemas.claim_schemas import ContextBrief

class TestDeterministicExtraction(unittest.TestCase):
    def test_extraction_success_car(self):
        claim = "The back of the car has a dent now. Mostly the rear bumper area."
        res = attempt_deterministic_extraction(claim, "car")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("success"))
        self.assertEqual(res["claimed_issue_type"], "dent")
        self.assertEqual(res["claimed_object_part"], "rear_bumper")
        
        brief = ContextBrief.model_validate(res)
        self.assertEqual(brief.claim_object, "car")
        self.assertEqual(brief.claimed_issue_type, "dent")
        self.assertEqual(brief.claimed_object_part, "rear_bumper")

    def test_extraction_success_laptop(self):
        claim = "My screen is cracked after it fell."
        res = attempt_deterministic_extraction(claim, "laptop")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("success"))
        self.assertEqual(res["claimed_issue_type"], "crack")
        self.assertEqual(res["claimed_object_part"], "screen")
        
        brief = ContextBrief.model_validate(res)
        self.assertEqual(brief.claim_object, "laptop")
        self.assertEqual(brief.claimed_issue_type, "crack")
        self.assertEqual(brief.claimed_object_part, "screen")

    def test_extraction_success_package(self):
        claim = "The box arrived and the seal is torn."
        res = attempt_deterministic_extraction(claim, "package")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("success"))
        self.assertEqual(res["claimed_issue_type"], "torn_packaging")
        self.assertEqual(res["claimed_object_part"], "seal")
        
        brief = ContextBrief.model_validate(res)
        self.assertEqual(brief.claim_object, "package")
        self.assertEqual(brief.claimed_issue_type, "torn_packaging")
        self.assertEqual(brief.claimed_object_part, "seal")

    def test_extraction_fallback_ambiguous_issue(self):
        # Two issues mentioned -> fallback to LLM
        claim = "The package arrived wet and stained."
        res = attempt_deterministic_extraction(claim, "package")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("success"))
        self.assertEqual(res.get("fallback_reason"), "ambiguous_issue")

    def test_extraction_fallback_ambiguous_part(self):
        # Two parts mentioned -> fallback to LLM
        claim = "The door has a scratch and the hood is also scratched."
        res = attempt_deterministic_extraction(claim, "car")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("success"))
        self.assertEqual(res.get("fallback_reason"), "ambiguous_part")

    def test_extraction_fallback_no_match(self):
        # No match -> fallback to LLM
        claim = "I want to report an issue with my delivery."
        res = attempt_deterministic_extraction(claim, "package")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("success"))
        self.assertEqual(res.get("fallback_reason"), "no_issue_match")

    def test_schema_unification(self):
        # Both paths must validate to the exact same schema structure
        det_res = {
            "claim_object": "car",
            "claimed_issue_type": "scratch",
            "claimed_object_part": "door",
            "claim_intent_summary": "Deterministic extraction"
        }
        llm_res = {
            "claim_object": "car",
            "claimed_issue_type": "scratch",
            "claimed_object_part": "door",
            "claim_intent_summary": "LLM extraction"
        }
        
        det_brief = ContextBrief.model_validate(det_res)
        llm_brief = ContextBrief.model_validate(llm_res)
        
        self.assertEqual(type(det_brief), type(llm_brief))
        self.assertEqual(det_brief.claim_object, llm_brief.claim_object)
        self.assertEqual(det_brief.claimed_issue_type, llm_brief.claimed_issue_type)
        self.assertEqual(det_brief.claimed_object_part, llm_brief.claimed_object_part)

if __name__ == "__main__":
    unittest.main()
