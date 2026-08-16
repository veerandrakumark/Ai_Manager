"""
Loop Closer — State Engine (PostgreSQL / Neon / Supabase)
=========================================================

Manages all persistent state using a psycopg2 connection pool backed
by a Serverless PostgreSQL database (Neon, Supabase, or any Postgres URL).

Key differences from the SQLite version
-----------------------------------------
• Parameter binding : %s  (not ?)
• Auto-increment    : SERIAL PRIMARY KEY  (not INTEGER AUTOINCREMENT)
• Default timestamp : DEFAULT CURRENT_TIMESTAMP  (not DEFAULT (datetime('now')))
• Concurrency       : SimpleConnectionPool (not per-call sqlite3.connect())
• Row access        : RealDictCursor — columns are accessible by name (row["id"])
• WAL pragma        : Removed — Postgres handles concurrency natively

Connection
----------
Reads DATABASE_URL from the environment.  Neon / Supabase both expose this
in the format:
    postgresql://user:pass@host/dbname?sslmode=require
"""

import os
from contextlib import contextmanager
from datetime import datetime

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Connection pool — initialised once at module import time
# ---------------------------------------------------------------------------

def _build_pool() -> SimpleConnectionPool:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "[DB] DATABASE_URL is not set. "
            "Add it to your .env file or Render environment variables."
        )
    return SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=db_url,
    )

_pool: SimpleConnectionPool = _build_pool()


@contextmanager
def _get_conn():
    """Context manager that borrows a connection from the pool and returns
    it automatically — even if an exception is raised inside the block.

    Usage:
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(...)
    """
    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Schema initialisation — idempotent (safe to call from multiple processes)
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the `users` and `tasks` tables if they don't already exist.

    Safe to call simultaneously from the web service, the Caspian listener,
    and the cron nudge engine on startup — CREATE TABLE IF NOT EXISTS is
    atomic in Postgres.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              SERIAL          PRIMARY KEY,
                    slack_handle    TEXT            UNIQUE NOT NULL,
                    telegram_handle TEXT,
                    github_handle   TEXT
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id                  SERIAL          PRIMARY KEY,
                    task_description    TEXT            NOT NULL,
                    assignee_id         INTEGER         NOT NULL
                                                        REFERENCES users(id),
                    status              TEXT            NOT NULL
                                                        DEFAULT 'in_progress',
                    deadline_timestamp  TIMESTAMP,
                    nudge_count         INTEGER         NOT NULL DEFAULT 0,
                    created_at          TIMESTAMP       NOT NULL
                                                        DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    print("[DB] Tables initialised successfully.", flush=True)


# ---------------------------------------------------------------------------
# CRUD — Users
# ---------------------------------------------------------------------------

def add_user(slack_handle: str,
             telegram_handle: str | None = None,
             github_handle: str | None = None) -> int:
    """Insert or update a user. Returns the row id.

    Uses an upsert so users can re-register to update their linked handles
    without hitting a UNIQUE constraint error.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (slack_handle, telegram_handle, github_handle)
                VALUES (%s, %s, %s)
                ON CONFLICT (slack_handle) DO UPDATE SET
                    telegram_handle = EXCLUDED.telegram_handle,
                    github_handle   = EXCLUDED.github_handle
                RETURNING id
                """,
                (slack_handle, telegram_handle, github_handle),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
    return row_id


def get_user_by_any_handle(handle: str) -> dict | None:
    """Look up a user by any of their known handles. Returns a dict or None."""
    # Defensively add/remove '@' for telegram handle matching
    alt_handle = handle[1:] if handle.startswith('@') else f"@{handle}"
    
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM users 
                WHERE slack_handle = %s 
                   OR telegram_handle = %s 
                   OR telegram_handle = %s
                   OR github_handle = %s
                """,
                (handle, handle, alt_handle, handle),
            )
            return cur.fetchone()


def get_user_by_id(user_id: int) -> dict | None:
    """Look up a user by primary key."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM users WHERE id = %s",
                (user_id,),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# CRUD — Tasks
# ---------------------------------------------------------------------------

def create_task(description: str,
                assignee_id: int,
                deadline_iso: str | None = None) -> int:
    """Insert a new task. Returns the new row id."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tasks (task_description, assignee_id, deadline_timestamp)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (description, assignee_id, deadline_iso),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
    return row_id


def add_commitment(github_handle: str,
                   description: str,
                   deadline: str | None = None) -> int | None:
    """High-level helper called by the message handler after LLM extraction.

    Looks up the user by their GitHub handle, then inserts a task row.
    Returns the new task id, or None if the github_handle isn't in the DB.

    Parameters
    ----------
    github_handle : str
        The user's GitHub handle (stored in the users table).
    description : str
        Concise task summary produced by extract_commitment().
    deadline : str | None
        Absolute deadline string ("YYYY-MM-DD HH:MM:SS") from the LLM,
        or None if the LLM returned null.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM users WHERE github_handle = %s",
                (github_handle,),
            )
            user_row = cur.fetchone()
            if user_row is None:
                return None

            cur.execute(
                """
                INSERT INTO tasks (task_description, assignee_id, deadline_timestamp)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (description, user_row[0], deadline),
            )
            task_id = cur.fetchone()[0]
        conn.commit()
    return task_id


def get_tasks_for_user(assignee_id: int) -> list[dict]:
    """Return all tasks assigned to a user."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM tasks WHERE assignee_id = %s ORDER BY created_at DESC",
                (assignee_id,),
            )
            return cur.fetchall()


def get_overdue_tasks() -> list[dict]:
    """Return all in-progress/pending tasks whose deadline has passed.

    Uses naive local time (YYYY-MM-DD HH:MM:SS) to match the format
    produced by intelligence.py and stored in deadline_timestamp.
    """
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM tasks
                WHERE  status IN ('in_progress', 'pending')
                  AND  deadline_timestamp IS NOT NULL
                  AND  deadline_timestamp < %s
                ORDER BY deadline_timestamp ASC
                """,
                (now_local,),
            )
            return cur.fetchall()


def get_overdue_tasks_with_users() -> list[dict]:
    """Return overdue tasks joined with user info (slack_handle, telegram_handle).

    Used exclusively by cron_nudge.py so it never touches a raw connection.
    """
    from datetime import timedelta
    # Add 5.5 hours to server's UTC time to match India Standard Time (IST)
    now_ist = (datetime.now() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.id          AS task_id,
                       t.task_description,
                       t.deadline_timestamp,
                       u.slack_handle,
                       u.telegram_handle
                FROM   tasks t
                JOIN   users u ON t.assignee_id = u.id
                WHERE  t.status IN ('in_progress', 'pending')
                  AND  t.deadline_timestamp IS NOT NULL
                  AND  t.deadline_timestamp < %s
                """,
                (now_ist,),
            )
            return cur.fetchall()


def lock_task_as_nudged(task_id: int) -> int:
    """Atomically transition a task from pending/in_progress → nudged.

    Returns the number of rows updated (1 = success, 0 = already locked
    by another process — caller should skip this task).
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks SET status = 'nudged'
                WHERE  id = %s
                  AND  status IN ('in_progress', 'pending')
                """,
                (task_id,),
            )
            rowcount = cur.rowcount
        conn.commit()
    return rowcount


def revert_task_to_in_progress(task_id: int) -> None:
    """Revert a task that failed to send a DM back to 'in_progress'
    so the nudge engine retries it on the next cycle.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET status = 'in_progress' WHERE id = %s",
                (task_id,),
            )
        conn.commit()


def update_task_status(task_id: int, new_status: str) -> None:
    """Update the status of a task (e.g. 'completed', 'in_progress')."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET status = %s WHERE id = %s",
                (new_status, task_id),
            )
        conn.commit()


def increment_nudge(task_id: int) -> None:
    """Bump the nudge counter by 1 for a given task."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET nudge_count = nudge_count + 1 WHERE id = %s",
                (task_id,),
            )
        conn.commit()


def mark_task_completed(github_handle: str) -> int:
    """Mark all open tasks for a specific GitHub user as completed.

    Looks up the user by their GitHub handle and updates the status
    of any of their 'in_progress' or 'pending' tasks to 'completed'.
    Returns the number of tasks updated.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM users WHERE github_handle = %s",
                (github_handle,),
            )
            user_row = cur.fetchone()
            if user_row is None:
                return 0

            cur.execute(
                """
                UPDATE tasks SET status = 'completed'
                WHERE  assignee_id = %s
                  AND  status IN ('in_progress', 'pending')
                """,
                (user_row[0],),
            )
            rowcount = cur.rowcount
        conn.commit()
    return rowcount
