"""Unmodified Genesis rendering; regression interpretation stays in card narrative.

The harness enum extension is recorded separately. This does not impersonate
an LLM evaluation, and empirical errors are not experimental uncertainties.
"""

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def install(skill):
    scripts = Path(skill) / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "render_card_sections", scripts / "render_card_sections.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Preserve pinned renderer prose as well as numbers. Rewriting generated
    # text otherwise causes UNTRACED in the independent stock validator.
    sys.modules["render_card_sections"] = module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["update", "validate"])
    args, remaining = p.parse_known_args()
    skill = Path(__file__).parent / "_vendor/genesis/card-eval-updater"
    install(skill)
    script = (
        skill / "scripts" / ("update_card.py" if args.mode == "update" else "validate_card_eval.py")
    )
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
