import sqlite3
import json

def inspect():
    db_path = "code/cache/pipeline_cache.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("--- Vision Cache Contents ---")
    cursor.execute("SELECT key, image_hashes, response_json FROM vision_cache")
    rows = cursor.fetchall()
    print(f"Total rows in vision_cache: {len(rows)}")
    for i, row in enumerate(rows):
        resp = json.loads(row["response_json"])
        print(f"Row {i+1}:")
        print(f"  Image Hashes: {row['image_hashes']}")
        print(f"  Response: {json.dumps(resp, indent=2)}")
        print("-" * 50)
        
    print("\n--- Context Cache Contents ---")
    cursor.execute("SELECT key, response_json FROM context_cache")
    rows = cursor.fetchall()
    print(f"Total rows in context_cache: {len(rows)}")
    for i, row in enumerate(rows):
        resp = json.loads(row["response_json"])
        print(f"Row {i+1}:")
        print(f"  Response: {json.dumps(resp, indent=2)}")
        print("-" * 50)

if __name__ == "__main__":
    inspect()
