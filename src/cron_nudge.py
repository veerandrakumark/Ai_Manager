"""
Loop Closer — Cron Nudge Engine
===============================

A continuous background process that polls the database every 60 seconds.
If a task is overdue (status is 'in_progress' or 'pending' and the deadline
has passed), it sends a direct message to the user via Caspian SDK and updates
the task status to 'nudged' to prevent spam.

This module is fully database-agnostic — it never touches a raw connection.
All SQL is encapsulated inside database.py functions.
"""

import os
import sys
import ast
import time
from dotenv import load_dotenv

from caspian_sdk import CommClient
from database import (
    init_db,
    get_overdue_tasks_with_users,
    lock_task_as_nudged,
    revert_task_to_in_progress,
)


def _parse_slack_handle(raw_handle) -> str:
    """Defensively extract a clean Slack user ID from the stored handle.

    The DB may contain:
      - A plain string:  "U0BP90ZAUF3"
      - A stringified dict: "{'address': 'U0BP90ZAUF3', 'name': None}"
      - A real dict (shouldn't happen but defensive)
    """
    if isinstance(raw_handle, dict):
        return raw_handle.get("address") or raw_handle.get("id") or str(raw_handle)
    if isinstance(raw_handle, str) and raw_handle.startswith("{"):
        try:
            parsed = ast.literal_eval(raw_handle)
            return parsed.get("address") or parsed.get("id") or raw_handle
        except Exception:
            return raw_handle
    return raw_handle or ""


def process_overdue_tasks(client: CommClient) -> None:
    """Scan the database for overdue tasks and nudge the users."""

    overdue_tasks = get_overdue_tasks_with_users()

    if not overdue_tasks:
        return

    # Fetch the active Slack connection once for the whole batch
    connections = client._request("GET", "/v1/connections")
    slack_conns = [
        c for c in connections
        if c.get("channel") == "slack" and c.get("status") == "active"
    ]
    conn_id = slack_conns[0]["id"] if slack_conns else None

    if not conn_id:
        print("[NUDGE] Warning: No active Slack connection found to send nudges.", flush=True)
        return

    for row in overdue_tasks:
        task_id     = row["task_id"]
        desc        = row["task_description"]
        deadline    = row["deadline_timestamp"]
        slack_handle = _parse_slack_handle(row["slack_handle"])

        message_text = (
            f"⚠️ *Nudge!* You committed to: '{desc}' by {deadline}. Is this done?"
        )
        print(f"[NUDGE] Reminding {slack_handle} about Task #{task_id}...", flush=True)

        # ── Optimistic lock: set status → 'nudged' BEFORE sending the DM.
        # If a second cron instance runs concurrently, it will see rowcount=0
        # and skip this task, preventing duplicate DMs. ───────────────────────
        rows_locked = lock_task_as_nudged(task_id)
        if rows_locked == 0:
            print(
                f"[NUDGE] Task #{task_id} already locked by another process, skipping.",
                flush=True,
            )
            continue

        # Now attempt the DM — revert status on failure so it retries next cycle
        try:
            client._request("POST", "/v1/messages", json={
                "connection_id": conn_id,
                "text": message_text,
                "channel": slack_handle,
            })
            print(f"[NUDGE] Task #{task_id} nudged successfully.", flush=True)

        except Exception as e:
            print(
                f"[NUDGE ERROR] DM failed for task #{task_id}: {e}. "
                "Reverting status to 'in_progress' for retry.",
                flush=True,
            )
            try:
                revert_task_to_in_progress(task_id)
            except Exception:
                pass  # Best-effort revert; the log entry is sufficient


def main() -> None:
    print("==================================================")
    print("Loop Closer — Cron Nudge Engine Started")
    print("==================================================")

    load_dotenv()

    caspian_key = os.getenv("CASPIAN_API_KEY")
    if not caspian_key:
        print("[ERROR] CASPIAN_API_KEY is not set. Exiting.")
        sys.exit(1)

    # Ensure tables exist before the first poll
    init_db()

    client = CommClient(api_key=caspian_key)
    print("[NUDGE] Polling database every 60 seconds for overdue tasks...", flush=True)

    while True:
        try:
            process_overdue_tasks(client)
        except Exception as e:
            print(f"[CRITICAL ERROR] Nudge engine caught exception: {e}", flush=True)

        time.sleep(60)


if __name__ == "__main__":
    main()
