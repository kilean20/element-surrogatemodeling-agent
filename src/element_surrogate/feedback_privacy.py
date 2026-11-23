"""Content-only heuristic checks for curated feedback, not proof of confidentiality."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_WINDOWS_PATH = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\)")
_PRIVATE_PATH = re.compile(
    r"(?<![\w])/(?:home|Users|mnt|media|srv|opt|tmp|var|workspace|workspaces|scratch|"
    r"projects?|data|lustre|gpfs|shared)(?:/|\b)"
)
_SECRET = re.compile(
    r"(?i)\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|authorization|"
    r"password|passwd|client[-_ ]?secret)[\"']?\s*[:=]\s*(?:Bearer\s+)?([^\s,;]+)"
)
_IDENTIFIER = re.compile(
    r"(?i)\b(?:hostname|host_name|machine_name|facility_name|pv_name|lattice_element)"
    r"\s*[:=]\s*\S+"
)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|ya29\.[A-Za-z0-9_-]{16,}|"
    r"AKIA[A-Z0-9]{16})\b|(?i:\bBearer\s+[A-Za-z0-9._~-]{16,})"
)
_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_QUERY_SECRETS = {"token", "access_token", "refresh_token", "api_key", "key", "auth", "password"}


def _credential_assignment(text):
    placeholders = {"[redacted]", "<redacted>", "[omitted]", "<omitted>"}
    for match in _SECRET.finditer(text):
        value = match.group(1).rstrip("}.)").strip("\"'").casefold()
        if value not in placeholders:
            return True
    return False


def _network_address(text):
    for candidate in _IPV4.findall(text):
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            pass
    return False


def _sensitive_url(text):
    for value in _URL.findall(text):
        try:
            url = urlsplit(value.rstrip(").,;"))
            host = (url.hostname or "").lower()
            if url.username or url.password:
                return True
            if any(key.lower() in _QUERY_SECRETS for key, _ in parse_qsl(url.query)):
                return True
            if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
                return True
            try:
                ipaddress.ip_address(host)
                return True
            except ValueError:
                pass
        except ValueError:
            return True
    return False


def review_content(content: dict[str, Any]) -> dict[str, Any]:
    """Inspect provided text only; return field references, never matched values."""
    entries = []
    for key, value in content.items():
        if isinstance(value, str):
            entries.append(("content." + key, value))
        elif isinstance(value, list):
            entries.extend((f"content.{key}[{index}]", text) for index, text in enumerate(value))
    findings = []
    for reference, text in entries:
        rules = (
            ("contact_identifier", bool(_EMAIL.search(text))),
            ("filesystem_path", bool(_WINDOWS_PATH.search(text) or _PRIVATE_PATH.search(text))),
            ("network_identifier", _network_address(text)),
            ("credential_or_private_url", _sensitive_url(text)),
            ("explicit_machine_identifier", bool(_IDENTIFIER.search(text))),
            ("private_key", bool(re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", text))),
            ("credential_assignment", _credential_assignment(text)),
            ("credential_pattern", bool(_TOKEN.search(text))),
        )
        findings.extend({"code": code, "field": reference} for code, matched in rules if matched)
    return {
        "status": "needs_generalization" if findings else "no_obvious_sensitive_patterns",
        "pattern_check_passed": not findings,
        "human_review_required": True,
        "findings": findings,
        "scope": (
            "Heuristic content-only preflight, not a confidentiality guarantee. Review scientific "
            "prose for sensitive facility facts and unnecessary machine specificity before sharing."
        ),
    }


def require_generalized(content: dict[str, Any]) -> dict[str, Any]:
    """Refuse an outbound action with flagged text, without echoing that text."""
    result = review_content(content)
    if not result["pattern_check_passed"]:
        raise ValueError(
            "feedback needs generalization before sharing; inspect the local privacy review"
        )
    return result
