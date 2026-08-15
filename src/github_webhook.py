"""
Loop Closer — GitHub Webhook
============================

A Flask application that listens for GitHub webhook events.
When a pull request is merged, it extracts the author's GitHub handle
and updates their open commitments in the SQLite database to 'completed'.
"""

import os
import hmac
import hashlib
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from database import mark_task_completed

load_dotenv()

app = Flask(__name__)

# Security warning on startup
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    print("[CRITICAL] GITHUB_WEBHOOK_SECRET is not set in the environment!", flush=True)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """Handle incoming GitHub webhook events with HMAC verification."""
    
    # Handle Render Health Checks (GET)
    if request.method == 'GET':
        return "OK", 200
    
    # 1. Cryptographic Signature Verification (Intercept before business logic)
    signature_header = request.headers.get("X-Hub-Signature-256")
    
    if not signature_header:
        print("[SECURITY ALERT] Rejected webhook: Missing X-Hub-Signature-256 header", flush=True)
        return jsonify({"error": "Missing X-Hub-Signature-256 header"}), 401

    if not WEBHOOK_SECRET:
        print("[CRITICAL] Cannot verify signature because GITHUB_WEBHOOK_SECRET is not set.", flush=True)
        return jsonify({"error": "Server misconfigured: No webhook secret"}), 500

    # Read raw request body bytes to prevent JSON re-serialization issues
    raw_payload_bytes = request.get_data()
    secret_bytes = WEBHOOK_SECRET.encode("utf-8")
    
    # Compute the expected signature
    expected_hash = hmac.new(secret_bytes, raw_payload_bytes, hashlib.sha256).hexdigest()
    expected_signature = f"sha256={expected_hash}"
    
    # Compare against the provided header signature safely to prevent timing attacks
    if not hmac.compare_digest(expected_signature, signature_header):
        print("[SECURITY ALERT] Rejected webhook: Invalid HMAC signature", flush=True)
        return jsonify({"error": "Invalid signature"}), 403

    try:
        # 2. Check if payload is JSON
        if not request.is_json:
            return jsonify({"status": "received", "reason": "not_json"}), 200

        # Safely parse JSON after signature validation
        payload = request.get_json()
        
        # 3. We only care about Pull Request events
        if 'pull_request' not in payload:
            return jsonify({"status": "received", "reason": "not_a_pr_event"}), 200
            
        # 4. We only care when action == "closed" and merged == true
        action = payload.get('action')
        merged = payload.get('pull_request', {}).get('merged')
        
        if action != "closed" or not merged:
            return jsonify({"status": "received", "reason": "not_a_merged_pr"}), 200
            
        # 5. Extract the GitHub handle of the person who opened the PR
        github_handle = payload.get('pull_request', {}).get('user', {}).get('login')
        
        if not github_handle:
            return jsonify({"status": "received", "reason": "no_github_handle"}), 200
            
        # 6. Mark all open tasks for this user as completed
        updated_count = mark_task_completed(github_handle)
        
        print(f"[WEBHOOK] Merged PR by {github_handle}. Updated {updated_count} open task(s) to 'completed'.", flush=True)
        return jsonify({"status": "received", "updated_tasks": updated_count}), 200

    except Exception as e:
        print(f"[WEBHOOK ERROR] Exception handling webhook payload: {e}", flush=True)
        # Always return 200 to GitHub to prevent endless retries / DDOS
        return jsonify({"status": "received", "error": "internal_error_ignored"}), 200

if __name__ == '__main__':
    # Run the Flask app on port 5000 (0.0.0.0 allows external traffic if exposed via proxy/ngrok)
    app.run(host='0.0.0.0', port=5000)
