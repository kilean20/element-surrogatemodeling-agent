"""Inspect the declared ESA dependency closure without changing the environment."""

from importlib import metadata

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def dependency_snapshot(profile="analysis"):
    """Read each selected distribution once per call, never cache across reviews.

    Metadata consistency is separate from advisories, provenance and API behavior.
    The full profile records absent optional components for update identities;
    only a caller selecting that profile treats those components as required.
    """
    if profile not in {"analysis", "neural", "cards", "full"}:
        raise ValueError("unknown runtime profile")
    queue = [("element-surrogate", () if profile == "analysis" else (profile,))]
    seen, distributions, versions, missing, edges = set(), {}, {}, set(), set()
    environment = default_environment()
    while queue:
        name, extras = queue.pop()
        name = canonicalize_name(name)
        key = (name, tuple(sorted(extras)))
        if key in seen:
            continue
        seen.add(key)
        if name not in distributions:
            try:
                info = metadata.distribution(name).metadata
            except metadata.PackageNotFoundError:
                distributions[name] = None
                versions[name] = None
                missing.add(name)
                continue
            distributions[name] = (info["Version"], info.get_all("Requires-Dist") or [])
            versions[name] = info["Version"]
        if distributions[name] is None:
            continue
        for raw in distributions[name][1]:
            request = Requirement(raw)
            if request.marker and not any(
                request.marker.evaluate({**environment, "extra": extra}) for extra in ("", *extras)
            ):
                continue
            edges.add((name, str(request)))
            queue.append((canonicalize_name(request.name), tuple(request.extras)))
    conflicts = []
    for parent, raw in sorted(edges):
        request = Requirement(raw)
        installed = versions[canonicalize_name(request.name)]
        if installed is not None and installed not in request.specifier:
            conflicts.append({"required_by": parent, "requirement": raw, "installed": installed})
    return {
        "profile": profile,
        "versions": dict(sorted(versions.items())),
        "missing": sorted(missing),
        "incompatible": conflicts,
        "scope": "Declared active dependency metadata; not security or scientific qualification",
    }
