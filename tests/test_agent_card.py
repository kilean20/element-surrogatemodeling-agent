"""Local checks of the agent card. CI also runs AI-ModCon's own check_card.py on it."""

import re
from pathlib import Path

import yaml

CARD = Path(__file__).resolve().parent.parent / "cards" / "agent_card.md"
MEDIA_TYPE = re.compile(r"^[a-z]+/[a-zA-Z0-9][a-zA-Z0-9!#$&^_.+-]*$")
TEMPLATE_BODY_TEXT = (
    "Replace this body",
    "A paragraph or two.",
    "Replace this with what",
    "Describe one thing your agent can do",
)


def _split(text):
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    assert match, "the card must start with YAML front matter"
    return yaml.safe_load(match.group(1)), match.group(2)


def test_card_is_an_agent_xcard():
    card, _ = _split(CARD.read_text(encoding="utf-8"))
    assert card["xCard"] == {"card_type": "agent", "version": "0.1"}
    assert card["extension"]["kind"] == "agent"
    for key in ("id", "name", "description", "science", "keywords", "developed_by", "license"):
        assert card[key], key
    assert card["sensitivity"] in {"unreviewed", "general", "reviewed", "restricted"}


def test_io_contract_uses_media_types():
    card, _ = _split(CARD.read_text(encoding="utf-8"))
    for side in ("input_modes", "output_modes"):
        values = card["io_contract"][side]
        assert len(values) == len(set(values)), side
        for value in values:
            assert MEDIA_TYPE.match(value), value


def test_keywords_are_not_bare_numbers():
    card, _ = _split(CARD.read_text(encoding="utf-8"))
    assert not [k for k in card["keywords"] if str(k).isdigit()]


def test_body_is_not_the_template():
    _, body = _split(CARD.read_text(encoding="utf-8"))
    assert not [s for s in TEMPLATE_BODY_TEXT if s in body]
