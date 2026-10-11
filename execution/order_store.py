"""
Durable order store for the paper execution gateway (Sprint 10.1).

SQLite (stdlib) persistence so that idempotency, order history and fills
survive process restarts. The UNIQUE constraint on idempotency_key is the
authoritative duplicate guard, independent of in-memory state.

Additive only: creates its own tables and never touches existing ones.
"""

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional


class OrderStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_orders (
                   seq INTEGER PRIMARY KEY AUTOINCREMENT,
                   order_id TEXT NOT NULL UNIQUE,
                   idempotency_key TEXT NOT NULL UNIQUE,
                   status TEXT NOT NULL,
                   payload TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def insert_if_new(self, order_dict: Dict[str, Any]) -> bool:
        """Returns True if inserted, False if the idempotency key already exists."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO paper_orders (order_id, idempotency_key, status, payload) VALUES (?,?,?,?)",
                    (order_dict["order_id"], order_dict["idempotency_key"],
                     order_dict["status"], json.dumps(order_dict, default=str)),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return False

    def update(self, order_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE paper_orders SET status=?, payload=? WHERE order_id=?",
                (order_dict["status"], json.dumps(order_dict, default=str), order_dict["order_id"]),
            )
            self._conn.commit()

    def get_by_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM paper_orders WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def all_orders(self) -> List[Dict[str, Any]]:
        """All orders in insertion order."""
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM paper_orders ORDER BY seq").fetchall()
        return [json.loads(r[0]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
