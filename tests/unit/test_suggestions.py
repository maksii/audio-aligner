import pytest

from audio_aligner.suggestions import get_suggestion, analyze_problems, generate_suggestions
from tests.fixtures.mock_media_info import mock_media_info

def test_codec_detection_logic():
    pass

def test_suggestion_ranking():
    pass

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