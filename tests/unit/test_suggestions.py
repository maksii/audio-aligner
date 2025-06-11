import pytest

from audio_aligner.suggestions import get_suggestion, analyze_problems, generate_suggestions
from tests.fixtures.mock_media_info import mock_media_info

def test_codec_detection_logic():
    """Ensure that get_suggestion picks the right tool based on codec and problem.

    We sample a few known codecs and validate the mapping rules defined in
    ``SUGGESTIONS_DB``. The test purposefully covers both an exact codec match
    and the fallback to the *default* entry.
    """

    # Exact match – AAC should use ffmpeg for "Cut"
    assert get_suggestion("aac", "Cut")["tool"] == "ffmpeg"

    # Exact match – AC3 should use eac3to for positive delay ("Delay")
    assert get_suggestion("ac3", "Delay")["tool"] == "eac3to"

    # Unknown codec should fall back to the default mapping (ffmpeg)
    assert get_suggestion("foobar", "Cut")["tool"] == "ffmpeg"

def test_suggestion_ranking():
    """Verify that *analyze_problems* returns problems in the expected priority
    order: Delay/Cut → Drifting delay → Different FPS. This implicit order is
    used by the CLI when printing suggestions.
    """

    # Case with constant delay + FPS change – delay should appear first
    result = {
        "mode_delay_ms": -120,
        "average_delay_ms": -121,
        "delay_threshold": 42,
        "delay_groups": [{"count": 10}],
    }

    ref_info = mock_media_info()  # Default ~23.976 fps (24000/1001)

    # Secondary video with different FPS to trigger the FPS issue (25 fps)
    from fractions import Fraction

    sec_info = mock_media_info(fps=Fraction(25, 1))
    # Guarantee the expected 'num/den' string format for the analyzer
    sec_info["streams"][0]["avg_frame_rate"] = "25/1"

    problems = analyze_problems(result, ref_info, sec_info)

    # Extract only the problem types preserving order
    ordered_problem_types = [p[0] for p in problems]
    assert ordered_problem_types[0] in ("Delay", "Cut")
    if len(ordered_problem_types) > 1:
        assert ordered_problem_types[1] in {"Drifting delay", "Different FPS"}
    # Ensure FPS is captured as one of the problems
    assert "Different FPS" in ordered_problem_types

def test_command_generation_with_various_parameters():
    suggestion = get_suggestion("aac", "Cut")
    cmd = suggestion.get("command")
    assert cmd is not None
    formatted_cmd = cmd.format(
        delay_s="{delay_s}", sec_file="{sec_file}", output_file="{output_file}"
    )
    assert 'ffmpeg -ss {delay_s}' in formatted_cmd
    assert '-c:a copy' in formatted_cmd

    suggestion = get_suggestion("dts", "Delay")
    cmd = suggestion.get("command")
    assert cmd is not None
    formatted_cmd = cmd.format(
        delay_ms="{delay_ms}", sec_file="{sec_file}", output_file="{output_file}"
    )
    assert 'eac3to' in formatted_cmd
    assert '+{delay_ms}ms' in formatted_cmd

def test_error_handling_for_unknown_codecs():
    suggestion = get_suggestion("unknown_codec", "Delay")
    assert suggestion["tool"] == "ffmpeg"

def test_problem_analysis_constant_delay():
    result = {
        "mode_delay_ms": -100,
        "average_delay_ms": -101,
        "delay_threshold": 42,
        "delay_groups": [{"count": 5}],
    }
    ref_info = mock_media_info()
    sec_info = mock_media_info()
    problems = analyze_problems(result, ref_info, sec_info)
    assert ("Delay", -100) in problems

def test_problem_analysis_drift_detection():
    result = {
        "mode_delay_ms": 0,
        "average_delay_ms": 0,
        "delay_threshold": 42,
        "delay_groups": [{"count": 2}, {"count": 3}],  # more than 1 group indicates drift
    }
    ref_info = mock_media_info()
    sec_info = mock_media_info()
    problems = analyze_problems(result, ref_info, sec_info)
    assert ("Drifting delay", None) in problems 