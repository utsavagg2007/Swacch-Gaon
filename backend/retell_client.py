import os
from typing import Optional, Dict, Any
import requests


RETELL_API_BASE = os.environ.get("RETELL_API_BASE", "https://api.retellai.com")


def _api_key() -> str:
    key = os.environ.get("RETELL_API_KEY")
    if not key:
        raise RetellError("RETELL_API_KEY is not configured")
    return key


class RetellError(Exception):
    pass


def _headers() -> Dict[str, str]:
    if not RETELL_API_KEY:
        raise RetellError("RETELL_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {RETELL_API_KEY}",
        "Content-Type": "application/json",
    }


def create_phone_call(
    *,
    agent_id: str,
    to_number: str,
    webhook_url: str,
    from_number: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    custom_headers: Optional[Dict[str, str]] = None,
    data_retention: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an outbound call via Retell v2 create-phone-call API.

    Ref: https://docs.retellai.com/api-references/create-phone-call

    Notes:
    - Retell uses Bearer token auth.
    - Field names can evolve; this wrapper is kept small and explicit.
    """

    url = f"{RETELL_API_BASE}/v2/create-phone-call"
    body: Dict[str, Any] = {
        "agent_id": agent_id,
        "to_number": to_number,
        "webhook_url": webhook_url,
    }
    if from_number:
        body["from_number"] = from_number
    if metadata:
        body["metadata"] = metadata
    if custom_headers:
        body["custom_headers"] = custom_headers
    if data_retention:
        body["data_retention"] = data_retention

    r = requests.post(url, headers=_headers(), json=body, timeout=30)
    if r.status_code >= 400:
        raise RetellError(f"Retell create_phone_call failed: {r.status_code} {r.text}")
    return r.json()
