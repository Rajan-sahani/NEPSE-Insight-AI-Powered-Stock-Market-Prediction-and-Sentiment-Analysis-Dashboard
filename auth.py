
import os
import time
from logger import logging
import hashlib
import secrets
from functools import wraps
from flask import session, redirect, url_for, flash, request

logger = logging.getLogger(__name__)

# ─── User Store (replace with DB in production) ─────────────────────────────────
# Passwords are SHA-256 hashed. Default: admin/nepse2024, analyst/nepse2024

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": {
        "password_hash": _hash("rajan123"),
        "role": "admin",
        "name": "System Administrator",
        "email": "admin@nepse.ai",
    },
    "analyst": {
        "password_hash": _hash("anlyst123"),
        "role": "analyst",
        "name": "Market Analyst",
        "email": "analyst@nepse.ai",
    },
    # "demo": {
    #     "password_hash": _hash("demo123"),
    #     "role": "viewer",
    #     "name": "Demo User",
    #     "email": "demo@nepse.ai",
    # },
}

# Brute-force protection: track failed attempts per IP
_failed_attempts: dict = {}  # ip -> [timestamp, ...]
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes


def _is_locked_out(ip: str) -> bool:
    now = time.time()
    attempts = _failed_attempts.get(ip, [])
    # Remove attempts older than lockout window
    recent = [t for t in attempts if now - t < LOCKOUT_SECONDS]
    _failed_attempts[ip] = recent
    return len(recent) >= MAX_ATTEMPTS


def _record_failure(ip: str):
    _failed_attempts.setdefault(ip, []).append(time.time())


def _clear_attempts(ip: str):
    _failed_attempts.pop(ip, None)


# ─── Auth Functions ──────────────────────────────────────────────────────────────

def login_user(username: str, password: str, ip: str = "0.0.0.0") -> dict:
    """Validate credentials. Returns {"success": bool, "message": str, "user": dict|None}"""
    if _is_locked_out(ip):
        return {"success": False, "message": "Too many failed attempts. Please wait 5 minutes.", "user": None}

    user = USERS.get(username)
    if user and user["password_hash"] == _hash(password):
        _clear_attempts(ip)
        logger.info(f"Successful login: {username} from {ip}")
        return {"success": True, "message": "Login successful", "user": {"username": username, **user}}
    else:
        _record_failure(ip)
        remaining = MAX_ATTEMPTS - len(_failed_attempts.get(ip, []))
        logger.warning(f"Failed login: {username} from {ip}")
        return {
            "success": False,
            "message": f"Invalid credentials. {max(0, remaining)} attempts remaining.",
            "user": None,
        }


def set_session(user: dict):
    session["logged_in"] = True
    session["username"] = user["username"]
    session["role"] = user["role"]
    session["name"] = user["name"]
    session["token"] = secrets.token_hex(16)


def clear_session():
    session.clear()


def is_authenticated() -> bool:
    return session.get("logged_in", False)


def get_current_user() -> dict:
    return {
        "username": session.get("username", ""),
        "role": session.get("role", "viewer"),
        "name": session.get("name", ""),
    }


# ─── Decorators ──────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_authenticated():
            flash("Please log in to access the dashboard.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_authenticated():
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated
