import os
import pandas as pd
from code.utils.helpers import normalize_issue_type, normalize_object_part, normalize_severity

def print_diffs():
    true_path = "dataset/sample_claims.csv"
    pred_path = "code/evaluation/sample_predictions.csv"
    
    if not os.path.exists(pred_path):
        print(f"Predictions not found at {pred_path}. Please run evaluation first.")
        return
        
    df_true = pd.read_csv(true_path)
    df_pred = pd.read_csv(pred_path)
    
    print(f"Loaded {len(df_true)} true rows, {len(df_pred)} prediction rows.\n")
    
    mismatches = []
    
    for idx, row in df_true.iterrows():
        pred_row = df_pred.iloc[idx]
        user_id = row["user_id"]
        claim_object = row["claim_object"]
        
        true_status = str(row["claim_status"]).strip().lower()
        pred_status = str(pred_row["claim_status"]).strip().lower()
        
        true_issue = normalize_issue_type(str(row["issue_type"]))
        pred_issue = normalize_issue_type(str(pred_row["issue_type"]))
        
        true_part = normalize_object_part(str(row["object_part"]), claim_object)
        pred_part = normalize_object_part(str(pred_row["object_part"]), claim_object)
        
        true_sev = normalize_severity(str(row["severity"]))
        pred_sev = normalize_severity(str(pred_row["severity"]))
        
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
            mismatches.append((user_id, claim_object, diff, row["user_claim"]))
            
    print(f"Total mismatches: {len(mismatches)} / {len(df_true)}\n")
    for user_id, obj, diff, claim in mismatches:
        print(f"User: {user_id} ({obj})")
        print(f"  Mismatches: {', '.join(diff)}")
        print(f"  Claim: {claim[:120]}...")
        print("-" * 60)

if __name__ == "__main__":
    print_diffs()
