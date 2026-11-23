"""One state-root resolver for visible new cases and unchanged legacy cases."""

from pathlib import Path

from element_surrogate.file_io import artifact_guard

STATE_DIRNAME = "records"
LEGACY_STATE_DIRNAME = ".element_surrogate"


def mutation_guard(directory):
    """Serialize cooperating knowledge/policy and client-update mutations."""
    return artifact_guard(state_directory(directory) / "project_mutation", create=True)


def state_directory(directory):
    """Choose one authoritative root without migrating history or resetting it.

    Recognized competing roots require inspection; unrelated user notes in a
    generic records/ directory must not block an existing legacy project.
    """
    root = Path(directory).resolve()
    visible, legacy = [root / name for name in (STATE_DIRNAME, LEGACY_STATE_DIRNAME)]
    legacy_state = legacy.is_dir() and any(legacy.iterdir())
    markers = (
        "project.json",
        "client_installation.json",
        "campaign.json",
        "campaign_ledger.json",
        "runs",
        "decisions",
        "knowledge",
    )
    visible_state = visible.is_dir() and any((visible / name).exists() for name in markers)
    if visible_state and legacy_state:
        raise ValueError(
            "both records/ and legacy .element_surrogate/ contain state; "
            "preserve both and resolve the conflicting project roots explicitly"
        )
    if legacy_state:
        return legacy
    if visible.exists() and not visible.is_dir():
        raise ValueError("project record path must be a directory: " + str(visible))
    return visible
