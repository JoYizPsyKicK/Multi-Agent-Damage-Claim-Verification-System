import os
import sqlite3
import hashlib
import json
from typing import Optional, Dict, Any, List
from code.config import settings


class PipelineCache:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else settings.cache_db_path
        # Ensure parent directory exists
        if os.path.dirname(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Enable WAL mode for concurrent write safety
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # Table for text context agent
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    key TEXT PRIMARY KEY,
                    prompt TEXT,
                    model_name TEXT,
                    response_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for vision agent
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vision_cache (
                    key TEXT PRIMARY KEY,
                    image_hashes TEXT,
                    prompt TEXT,
                    model_name TEXT,
                    response_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def get_image_hash(image_path: str) -> str:
        """Computes SHA256 of the image file bytes. Falls back to hashing the path if missing."""
        try:
            if os.path.exists(image_path):
                hasher = hashlib.sha256()
                with open(image_path, "rb") as f:
                    # Read in 64kb chunks
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()
        except Exception:
            pass
        # Fallback to hashing the path string itself
        return hashlib.sha256(image_path.encode("utf-8")).hexdigest()

    def get_context(self, prompt: str, model_name: str) -> Optional[Dict[str, Any]]:
        key_src = f"{prompt}:{model_name}:{settings.prompt_version}"
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM context_cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["response_json"])
        return None

    def set_context(self, prompt: str, model_name: str, response: Dict[str, Any]):
        key_src = f"{prompt}:{model_name}:{settings.prompt_version}"
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        response_json = json.dumps(response)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO context_cache (key, prompt, model_name, response_json) VALUES (?, ?, ?, ?)",
                (key, prompt, model_name, response_json)
            )
            conn.commit()

    def get_vision(self, image_paths: List[str], prompt: str, model_name: str) -> Optional[Dict[str, Any]]:
        # Hash image contents
        img_hashes = [self.get_image_hash(path) for path in image_paths]
        img_hashes_str = ";".join(img_hashes)
        
        # Combined key including prompt version
        key_src = f"{img_hashes_str}:{prompt}:{model_name}:{settings.prompt_version}"
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM vision_cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["response_json"])
        return None

    def set_vision(self, image_paths: List[str], prompt: str, model_name: str, response: Dict[str, Any]):
        img_hashes = [self.get_image_hash(path) for path in image_paths]
        img_hashes_str = ";".join(img_hashes)
        
        key_src = f"{img_hashes_str}:{prompt}:{model_name}:{settings.prompt_version}"
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
        response_json = json.dumps(response)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vision_cache (key, image_hashes, prompt, model_name, response_json) VALUES (?, ?, ?, ?, ?)",
                (key, img_hashes_str, prompt, model_name, response_json)
            )
            conn.commit()

