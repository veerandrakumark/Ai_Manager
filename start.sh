#!/bin/bash
# ============================================================
# Loop Closer — Render Startup Script
# ============================================================
# Runs the background workers and the web server simultaneously
# to bypass Render's single-process free tier restriction.

# 1. Start the Caspian SDK listener in the background
echo "Starting Listener..."
python src/main.py &

# 2. Start the Nudge Engine in the background
echo "Starting Nudge Engine..."
python src/cron_nudge.py &

# 3. Start the Flask/Gunicorn Webhook in the foreground
# (Render requires the main process to bind to $PORT)
echo "Starting Webhook on port $PORT..."
gunicorn --bind 0.0.0.0:$PORT --workers 2 src.github_webhook:app
