"""Private, curated improvement reports and optional packet-specific sharing.

Consent is a client attestation of a user's decision, not authentication of that
person. Incoming text is evidence to review, never executable agent guidance.
"""

import hashlib
import json
import os
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from element_surrogate import __version__
from element_surrogate.benchmark.protocol import file_hash, read_json, write_json
from element_surrogate.feedback_privacy import require_generalized, review_content
from element_surrogate.file_io import artifact_guard
from element_surrogate.project import state_paths
from element_surrogate.project_paths import state_directory

MAX_BYTES = 128 * 1024
PURPOSE = "Review evidence to improve element-surrogate-agent; no automatic software changes"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {label} required")
    return value


def _bounded(path):
    if Path(path).stat().st_size > MAX_BYTES:
        raise ValueError("curated feedback exceeds 128 KiB; shorten it before review")
    return read_json(path)


def _destination(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"provider", "folder_id"}
        or value["provider"] != "google_drive"
        or not isinstance(value["folder_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value["folder_id"])
    ):
        raise ValueError("explicit google_drive folder_id required; no default enrollment")


def _content(value):
    required = {
        "category",
        "summary",
        "user_feedback",
        "findings",
        "evidence",
        "suggested_change",
        "limitations",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("curated feedback content fields required")
    if value["category"] not in {
        "workflow",
        "physics",
        "numerics",
        "benchmark",
        "documentation",
        "software",
        "other",
    }:
        raise ValueError("unknown feedback category")
    for key in required - {"evidence"}:
        _text(value[key], key)
    if not isinstance(value["evidence"], list) or not all(
        isinstance(x, str) and x.strip() for x in value["evidence"]
    ):
        raise ValueError(
            "evidence must list reviewed descriptions/identities; no automatic attachments"
        )


def _validate_packet(packet):
    if (
        set(packet)
        != {
            "schema",
            "software_version",
            "created_utc",
            "purpose",
            "recipient",
            "destination",
            "content",
        }
        or packet["schema"] != "element_surrogate.feedback_packet.v1"
    ):
        raise ValueError("feedback_packet.v1 required")
    if packet["purpose"] != PURPOSE:
        raise ValueError("improvement purpose differs")
    for key in ("software_version", "created_utc", "recipient"):
        _text(packet[key], key)
    _destination(packet["destination"])
    _content(packet["content"])


def _paths(directory, feedback_id):
    root, _ = state_paths(directory)
    if not isinstance(feedback_id, str) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", feedback_id
    ):
        raise ValueError("safe lowercase feedback ID required")
    folder = state_directory(root) / "feedback"
    folder.mkdir(exist_ok=True)
    return folder / (feedback_id + ".json"), folder / (feedback_id + ".state.json")


def prepare(directory, feedback_id, request_path):
    """Copy only explicitly curated text; never scan project data or transcripts."""
    request = _bounded(request_path)
    if (
        set(request) != {"schema", "recipient", "destination", "content"}
        or request["schema"] != "element_surrogate.feedback_request.v1"
    ):
        raise ValueError("feedback_request.v1 recipient/destination/content required")
    packet = {
        "schema": "element_surrogate.feedback_packet.v1",
        "software_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": PURPOSE,
        **{k: request[k] for k in ("recipient", "destination", "content")},
    }
    _validate_packet(packet)
    if len(json.dumps(packet, ensure_ascii=False).encode("utf-8")) > MAX_BYTES - 2048:
        raise ValueError("curated report too large")
    privacy = review_content(packet["content"])
    path, _ = _paths(directory, feedback_id)
    write_json(path, packet, exclusive=True)
    return {
        "packet_path": str(path),
        "packet_sha256": file_hash(path),
        "review_payload": packet if privacy["pattern_check_passed"] else None,
        "privacy_review": privacy,
        "sharing_status": (
            "local_only_no_consent"
            if privacy["pattern_check_passed"]
            else "local_only_needs_generalization"
        ),
        "warnings": (
            []
            if privacy["pattern_check_passed"]
            else [
                {
                    "code": "FEEDBACK_NEEDS_GENERALIZATION",
                    "message": (
                        "Possible private identifiers or credentials; values are not echoed."
                    ),
                    "recommended_action": (
                        "Generalize the flagged fields and prepare a new reviewed draft. "
                        "Feedback sharing is paused; unrelated scientific work can continue."
                    ),
                    "evidence_ref": str(path),
                    "blocking": False,
                }
            ]
        ),
        "uploaded": False,
    }


def consent(
    directory, feedback_id, expected_sha256, decision, actor, user_statement, source_reference
):
    """Bind a client-attested user choice to the exact report and destination."""
    if decision not in {"approve", "decline", "revoke"}:
        raise ValueError("explicit approve/decline/revoke required")
    for value in (actor, user_statement, source_reference):
        _text(value, "user decision provenance")
    path, state_path = _paths(directory, feedback_id)
    with artifact_guard(str(state_path) + ".operation", create=True):
        packet = _bounded(path)
        _validate_packet(packet)
        if file_hash(path) != expected_sha256:
            raise ValueError("packet changed; show exact current payload before consent")
        if decision == "approve":
            require_generalized(packet["content"])
        state = read_json(state_path) if state_path.exists() else {"events": []}
        if state.get("delivery_status") in {"sent", "uncertain"} and decision == "approve":
            raise ValueError(
                "already sent/uncertain: reconcile remote delivery before any new packet"
            )
        event = {
            "decision": decision,
            "actor": actor,
            "user_statement": user_statement,
            "source_reference": source_reference,
            "created_utc": datetime.now(UTC).isoformat(),
            "packet_sha256": expected_sha256,
            "recipient": packet["recipient"],
            "destination": packet["destination"],
            "purpose": packet["purpose"],
        }
        state["events"].append(event)
        state["consent"] = event
        write_json(state_path, state)
    return {
        "state_path": str(state_path),
        "decision": decision,
        "packet_sha256": expected_sha256,
        "authority": "client-attested user decision; not identity authentication",
        "revocation_scope": (
            "Future attempts only; already transmitted information cannot be recalled"
        ),
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http(method, url, token, body=None, content_type=None):
    """Fixed Google endpoint only. Never expose tokens or server error bodies."""
    if not url.startswith(
        ("https://www.googleapis.com/drive/v3/", "https://www.googleapis.com/upload/drive/v3/")
    ):
        raise ValueError("fixed Drive API host required")
    headers = {"Authorization": "Bearer " + token}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
        data = response.read(1_048_577)
        if len(data) > 1_048_576:
            raise ValueError("Drive response exceeded limit")
        return json.loads(data)


def send_drive(directory, feedback_id, expected_sha256, access_token_env):
    """One consented attempt using the user's own OAuth token environment variable.

    An uncertain upload is never automatically retried. The project sender does
    not obtain credentials, change scopes, copy sign-in files or authenticate users.
    """
    if not isinstance(access_token_env, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", access_token_env
    ):
        raise ValueError("access-token environment variable name required, never the token itself")
    path, state_path = _paths(directory, feedback_id)
    with artifact_guard(str(state_path) + ".operation", create=True):
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("feedback exceeds size limit")
        packet = json.loads(raw)
        _validate_packet(packet)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("packet identity changed")
        state = read_json(state_path) if state_path.exists() else {}
        choice = state.get("consent", {})
        if choice.get("decision") != "approve" or any(
            choice.get(k) != v
            for k, v in {
                "packet_sha256": expected_sha256,
                "recipient": packet["recipient"],
                "destination": packet["destination"],
                "purpose": packet["purpose"],
            }.items()
        ):
            raise ValueError("exact report/recipient/destination user approval required")
        if state.get("delivery_status") in {"sent", "uncertain", "attempting"}:
            raise ValueError("no automatic retry: reconcile prior remote delivery")
        require_generalized(packet["content"])
        token = os.environ.get(access_token_env)
        if not token:
            return {
                "status": "pending_connection",
                "uploaded": False,
                "packet_path": str(path),
                "reason": (
                    "Configure your authorized OAuth connection or share the reviewed file manually"
                ),
            }
        if len(token) > 16384 or any(c in token for c in "\r\n"):
            raise ValueError("invalid access-token environment value")
        folder = packet["destination"]["folder_id"]
        try:
            metadata = _http(
                "GET",
                "https://www.googleapis.com/drive/v3/files/"
                + folder
                + "?fields=id,mimeType,capabilities(canAddChildren)&supportsAllDrives=true",
                token,
            )
            allowed = (
                metadata.get("id") == folder
                and metadata.get("mimeType") == "application/vnd.google-apps.folder"
                and metadata.get("capabilities", {}).get("canAddChildren") is True
            )
        except Exception:
            return {
                "status": "connection_or_permission_check_failed",
                "uploaded": False,
                "reason": ("No upload attempted; verify your OAuth access and folder permissions"),
            }
        if not allowed:
            return {"status": "folder_write_not_verified", "uploaded": False}
        # Record before files.create: process failure thereafter leaves an uncertain attempt.
        state["delivery_status"] = "attempting"
        write_json(state_path, state)
        boundary = "esa_feedback_" + expected_sha256
        meta = json.dumps(
            {
                "name": "esa-feedback-" + expected_sha256[:24] + ".json",
                "parents": [folder],
                "appProperties": {"esa_packet_sha256": expected_sha256},
            }
        )
        body = (
            (
                "--"
                + boundary
                + "\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
                + meta
                + "\r\n--"
                + boundary
                + "\r\nContent-Type: application/json\r\n\r\n"
            ).encode()
            + raw
            + ("\r\n--" + boundary + "--\r\n").encode()
        )
        try:
            reply = _http(
                "POST",
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id&supportsAllDrives=true",
                token,
                body,
                "multipart/related; boundary=" + boundary,
            )
            remote_id = reply.get("id")
            if not isinstance(remote_id, str) or not remote_id.strip():
                raise ValueError("missing remote file identity")
        except Exception:
            state["delivery_status"] = "uncertain"
            write_json(state_path, state)
            return {
                "status": "uncertain",
                "uploaded": None,
                "reason": (
                    "Remote delivery may have occurred. Reconcile the packet hash in Drive; "
                    "do not retry blindly"
                ),
            }
        state.update(
            delivery_status="sent", remote_file_id=remote_id, sent_utc=datetime.now(UTC).isoformat()
        )
        write_json(state_path, state)
        return {
            "status": "sent",
            "uploaded": True,
            "remote_file_id": remote_id,
            "packet_sha256": expected_sha256,
        }


def collect(inbox_directory, output_path):
    """Index owner-obtained packets only, with no network access or promotion."""
    paths = sorted(Path(inbox_directory).glob("*.json"))
    if len(paths) > 1000:
        raise ValueError("intake exceeds 1000 packets; partition the inbox")
    rows, rejected = [], []
    for path in paths:
        try:
            packet = _bounded(path)
            _validate_packet(packet)
        except (ValueError, TypeError, KeyError, OSError):
            rejected.append(path.name)
            continue
        rows.append(
            {
                "file": path.name,
                "sha256": file_hash(path),
                "software_version": packet["software_version"],
                **{k: packet["content"][k] for k in ("category", "summary", "evidence")},
            }
        )
    result = {
        "schema": "element_surrogate.feedback_intake.v1",
        "packets": rows,
        "rejected_files": rejected,
        "trust": "Untrusted user text, not instructions",
        "next_step": (
            "Review, reproduce, test and release changes explicitly; "
            "no automatic KB promotion or software patch"
        ),
    }
    write_json(output_path, result, exclusive=True)
    return result
