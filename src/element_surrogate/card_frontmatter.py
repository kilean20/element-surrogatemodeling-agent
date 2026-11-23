"""YAML metric composition around pinned Genesis renderers, without vendor edits."""

from collections import Counter

import yaml


class UniqueKeysLoader(yaml.SafeLoader):
    """Reject ambiguous mappings instead of silently retaining the last value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML frontmatter key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeysLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def split_card(text):
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("model card YAML frontmatter required")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("model card frontmatter is not closed")
    front = yaml.load("".join(lines[1:end]), Loader=UniqueKeysLoader)
    if not isinstance(front, dict):
        raise ValueError("model card frontmatter must be a mapping")
    return front, "".join(lines[end + 1 :])


def expected_metrics(bundle):
    return list(dict.fromkeys(f"{r['benchmark']}/{r['metric']}" for r in bundle["results"]))


def metric_coverage(text, bundle):
    front, _ = split_card(text)
    metrics = front.get("metrics", [])
    if not isinstance(metrics, list) or any(not isinstance(m, str) for m in metrics):
        raise ValueError("model metrics must be a YAML list of strings")
    expected = expected_metrics(bundle)
    missing = [m for m in expected if m not in metrics]
    duplicate = [m for m, n in Counter(metrics).items() if n > 1]
    return {
        "pass": not missing and not duplicate,
        "expected_metrics": expected,
        "missing_metrics": missing,
        "duplicate_metrics": duplicate,
        "scope": "duplicate-free YAML and exact bundle metric coverage; not registry conformance",
    }


def compose_metrics(text, bundle, supplied_metrics):
    front, body = split_card(text)
    if not isinstance(supplied_metrics, list) or any(
        not isinstance(m, str) for m in supplied_metrics
    ):
        raise ValueError("supplied model metrics must be a list of strings")
    coverage = metric_coverage(text, bundle)
    if not coverage["pass"]:
        raise ValueError("upstream model metrics missing or duplicated; inspect raw card")
    front["metrics"] = list(dict.fromkeys(supplied_metrics + front["metrics"]))
    return "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + body
