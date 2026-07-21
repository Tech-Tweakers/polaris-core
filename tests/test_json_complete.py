"""Mirror of the C++ json_complete heuristic used for XCT early-stop.

These tests validate the parsing shape that Polaris Core expects before
stopping generation early on a JSON response.
"""

import pytest


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"done": true}', True),
        ('{"done": true', False),
        ('{"next_step": "foo"}', True),
        ('{"key": "\\"done\\""}', True),
        ('{"key": "\\"done\\"}', False),
        ('[{"done": true}]', True),
        ('{"nested": {"done": true}}', True),
        ("not json", False),
    ],
)
def test_json_complete(text, expected):
    braces = 0
    brackets = 0
    in_str = False
    esc = False
    started = False

    for c in text:
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if c == "{":
            braces += 1
            started = True
        elif c == "}":
            braces -= 1
        elif c == "[":
            brackets += 1
            started = True
        elif c == "]":
            brackets -= 1

    result = started and braces == 0 and brackets == 0 and not in_str
    assert result is expected
