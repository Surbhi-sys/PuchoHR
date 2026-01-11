# db_util.py
import sqlite3
from datetime import datetime

DB_FILE = "hr_chat.db"

def init_db():
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                source_doc TEXT,
                similarity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def store_chat(session_id, question, answer, source_doc=None, similarity_score=None):
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history
            (session_id, question, answer, source_doc, similarity_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, question, answer, source_doc, similarity_score, datetime.now()))
        conn.commit()

def get_recent_chat(session_id, limit=3):
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, answer 
            FROM chat_history
            WHERE session_id=?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit))
        return cursor.fetchall()
