"""
Loop Closer — Multi-Channel Handler (Caspian SDK)

Entry point for the application.
  • Loads environment variables
  • Initialises the database
  • Connects Slack + Telegram via Caspian CommClient
  • Handles identity registration and incoming messages
"""

import os
import sys
import time
import re
from dotenv import load_dotenv
from caspian_sdk import CommClient

from database import init_db, add_user, add_commitment, get_user_by_slack
from intelligence import extract_commitment


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load environment
    # ------------------------------------------------------------------
    load_dotenv()

    caspian_key = os.getenv("CASPIAN_API_KEY")
    slack_token = os.getenv("SLACK_TOKEN")
    slack_app_token = os.getenv("SLACK_APP_TOKEN")
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

    missing = []
    if not caspian_key:
        missing.append("CASPIAN_API_KEY")
    if not slack_token:
        missing.append("SLACK_TOKEN")
    if not slack_app_token:
        missing.append("SLACK_APP_TOKEN")
    if not telegram_token:
        missing.append("TELEGRAM_BOT_TOKEN")

    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print("        Copy .env.example → .env and fill in your credentials.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Initialise database
    # ------------------------------------------------------------------
    init_db()

    # ------------------------------------------------------------------
    # 3. Set up Caspian CommClient with Slack & Telegram channels
    # ------------------------------------------------------------------
    client = CommClient(api_key=caspian_key)

    slack_conn = client.connect_slack(bot_token=slack_token, app_token=slack_app_token)
    if slack_conn.get("status") == "pending_oauth" or slack_conn.get("authorize_url"):
        print("\n" + "=" * 76, flush=True)
        print(" [ACTION REQUIRED] SLACK AUTHORIZATION NEEDED", flush=True)
        print(" Open this link in your browser to complete the Slack workspace setup:", flush=True)
        print(f" -> {slack_conn.get('authorize_url')}", flush=True)
        print("=" * 76 + "\n", flush=True)
    else:
        print(f"[CASPIAN] Slack channel connected: {slack_conn.get('id', 'N/A')} (status: {slack_conn.get('status', 'unknown')})", flush=True)

    telegram_conn = client.connect_telegram(bot_token=telegram_token)
    print(f"[CASPIAN] Telegram channel connected: {telegram_conn.get('id', 'N/A')} (status: {telegram_conn.get('status', 'unknown')})")

    # ------------------------------------------------------------------
    # 4. Global message listener — catches ALL incoming messages
    # ------------------------------------------------------------------
    @client.on_message
    def handle_message(message):
        raw_text = message.text or ""
        text = raw_text.strip().lower()

        # Extract sender string safely (message.sender can be dict, str, or None)
        if isinstance(message.sender, dict):
            sender_handle = (
                message.sender.get("name")
                or message.sender.get("username")
                or message.sender.get("address")   # Slack user ID e.g. U0BP90ZAUF3
                or message.sender.get("id")
                or str(message.sender)
            )
        else:
            sender_handle = str(message.sender or "unknown")

        print(f"[EVENT RECEIVED] [{message.channel}] {sender_handle}: {raw_text}", flush=True)

        # --- FEATURE: IDENTITY REGISTRATION ---
        if "!register" in text:
            try:
                # Extract GitHub and Telegram handles using regex
                # Use raw_text (not lowercased text) to preserve GitHub handle case.
                # Allow optional whitespace after the colon (e.g. "github: handle").
                gh_match = re.search(r'github:\s*(\S+)', raw_text, re.IGNORECASE)
                tg_match = re.search(r'telegram:\s*(\S+)', raw_text, re.IGNORECASE)

                # Extract and strip common markdown characters (like backticks)
                github_handle = gh_match.group(1).strip(" `") if gh_match else None
                telegram_handle = tg_match.group(1).strip(" `").lower() if tg_match else None

                if not github_handle or not telegram_handle:
                    message.reply(
                        "⚠️ *Missing handles!* Please use the exact format:\n"
                        "`!register github:your_gh_handle telegram:@your_telegram_handle`"
                    )
                    return

                # Save the identity mapping to the SQLite database
                add_user(
                    slack_handle=sender_handle,
                    telegram_handle=telegram_handle,
                    github_handle=github_handle
                )

                # Confirm success back to the chat thread
                message.reply(
                    f"✅ *Identity Linked Successfully!*\n"
                    f"• GitHub: `{github_handle}`\n"
                    f"• Telegram: `{telegram_handle}`"
                )
                print(f"[DB] Registered {sender_handle} -> GH: {github_handle}, TG: {telegram_handle}", flush=True)

            except Exception as e:
                message.reply(f"❌ Database error during registration: {e}")
                print(f"[ERROR] Registration failed: {e}", flush=True)

            return  # Stop processing further logic for this message

        # --- FEATURE: COMMITMENT DETECTION ---
        # Pass every non-command message through the LLM extraction engine.
        # extract_commitment() never raises — returns safe fallback on error.
        result = extract_commitment(raw_text)

        if result["is_commitment"]:
            description = result["task_description"]
            deadline    = result["deadline"]

            print(
                f"[INTELLIGENCE] Commitment detected from {sender_handle}: "
                f"{description!r} | deadline: {deadline}",
                flush=True,
            )

            # Look up the sender in the DB by their Slack handle
            user_row = get_user_by_slack(sender_handle)

            if user_row is None:
                # Sender is not registered — prompt them to register first
                message.reply(
                    "📋 *Commitment detected!* But you're not registered yet.\n"
                    "Use `!register github:your_gh_handle telegram:@your_handle` "
                    "so I can track this for you."
                )
                print(
                    f"[DB] Commitment ignored — {sender_handle} is not registered.",
                    flush=True,
                )
                return

            # User is registered — save the commitment to the DB
            github_handle = user_row["github_handle"]
            try:
                task_id = add_commitment(
                    github_handle=github_handle,
                    description=description,
                    deadline=deadline,
                )
                # Format deadline for display (strip seconds for readability)
                deadline_display = deadline[:16] if deadline else "no deadline set"
                message.reply(
                    f"✅ *Commitment logged!*\n"
                    f"• Task: {description}\n"
                    f"• Deadline: `{deadline_display}`\n"
                    f"• ID: `#{task_id}`\n"
                    f"_I'll nudge you if this isn't closed before the deadline._"
                )
                print(
                    f"[DB] Task #{task_id} saved for {github_handle}: "
                    f"{description!r} | due {deadline}",
                    flush=True,
                )
            except Exception as e:
                message.reply(f"❌ Failed to save commitment: {e}")
                print(f"[ERROR] add_commitment failed: {e}", flush=True)

    # ------------------------------------------------------------------
    # 5. Start listening (blocking keep-alive loop)
    # ------------------------------------------------------------------
    print("[LOOP CLOSER] Listening for messages… (Ctrl+C to stop)")

    try:
        client.listen()
    except KeyboardInterrupt:
        print("\n[LOOP CLOSER] Shutting down gracefully...")


if __name__ == "__main__":
    main()