from audio_aligner.post_processing import find_delay_groups, build_results


def test_delay_group_finding():
    chunk_delays = [
        {"start_time": 0, "delay": 10},
        {"start_time": 5, "delay": 11},
        {"start_time": 10, "delay": 9},
        {"start_time": 15, "delay": 100},
        {"start_time": 20, "delay": 101},
        {"start_time": 25, "delay": 99},
    ]
    groups = find_delay_groups(chunk_delays, grouping_threshold_ms=5, min_group_size=3)
    assert len(groups) == 2
    assert groups[0]["average_delay"] == 10
    assert groups[1]["average_delay"] == 100


def test_results_building():
    valid_delays = [(0, 10), (5, 12), (10, 10)]
    result = build_results(
        valid_delays,
        "align",
        chunk_duration=5,
        frame_duration=41.7,
        delay_threshold=42,
        ref=("ref.mkv", 0),
        sec=("sec.mkv", 1),
    )
    assert result["command"] == "align"
    assert result["mode_delay_ms"] == 10
    assert result["average_delay_ms"] == 10  # (10+12+10)/3
    assert result["reference_file"] == "ref.mkv"
    assert result["secondary_track"] == 1
    assert len(result["chunk_delays_ms"]) == 3 