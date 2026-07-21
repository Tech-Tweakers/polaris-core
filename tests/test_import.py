import os
import sys

import pytest


def test_example_usage_is_importable():
    """The example usage script should be syntactically valid Python."""
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "example_usage.py"
    )
    with open(example_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), example_path, "exec")
    assert code is not None


def test_polaris_core_import_skips_without_binary():
    """polaris_core is a compiled extension; skip when not built."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)
    try:
        import polaris_core  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"compiled polaris_core not available: {exc}")
