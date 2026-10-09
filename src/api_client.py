"""Client HTTP de l'API (utilisé par l'application Streamlit)."""
import os

import requests


def get_api_url():
    url = os.environ.get("API_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("API_URL")
        except Exception:
            url = None
    return (url or "http://localhost:8000").rstrip("/")


def api_health(url=None, timeout=20):
    """Renvoie (disponible, message)."""
    url = url or get_api_url()
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        r.raise_for_status()
        return True, r.json().get("status", "ok")
    except Exception as e:
        return False, type(e).__name__


def api_post(path, payload, url=None, timeout=120):
    url = url or get_api_url()
    r = requests.post(f"{url}{path}", json=payload, timeout=timeout)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"API {r.status_code} : {str(detail)[:400]}")
    return r.json()


def api_get(path, url=None, timeout=60):
    url = url or get_api_url()
    r = requests.get(f"{url}{path}", timeout=timeout)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"API {r.status_code} : {str(detail)[:400]}")
    return r.json()
