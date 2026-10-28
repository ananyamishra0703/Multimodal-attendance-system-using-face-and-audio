"""
db.py
Local SQLite storage for enrolled students and attendance records.
Uses Python's built-in sqlite3 module — no extra dependency, nothing to
compile, works identically on every platform.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "attendance.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            enrolled_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL REFERENCES students(id),
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            face_confidence REAL,
            voice_confidence REAL,
            status TEXT NOT NULL CHECK (status IN ('Present', 'Unverified')),
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()


def add_student(name, roll_no):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO students (name, roll_no) VALUES (?, ?)", (name, roll_no)
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_student(student_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_student_by_roll(roll_no):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE roll_no = ?", (roll_no,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_students():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_attendance(student_id, face_confidence, voice_confidence, status):
    now = datetime.now()
    conn = get_connection()
    # Prevent duplicate "Present" entries for the same student on the same day.
    existing = conn.execute(
        "SELECT id FROM attendance WHERE student_id = ? AND date = ? AND status = 'Present'",
        (student_id, now.strftime("%Y-%m-%d")),
    ).fetchone()
    if existing:
        conn.close()
        return None, "already_marked"

    cur = conn.execute(
        """INSERT INTO attendance (student_id, date, time, face_confidence, voice_confidence, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            student_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            face_confidence,
            voice_confidence,
            status,
        ),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id, "ok"


def list_attendance(date=None):
    conn = get_connection()
    query = """
        SELECT a.*, s.name, s.roll_no
        FROM attendance a JOIN students s ON s.id = a.student_id
    """
    params = ()
    if date:
        query += " WHERE a.date = ?"
        params = (date,)
    query += " ORDER BY a.date DESC, a.time DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
