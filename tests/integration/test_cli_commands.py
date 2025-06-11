import os
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from audio_aligner.main import cli


@patch("audio_aligner.main.run_align")
def test_align_command_with_mock_files(mock_run_align, tmp_path):
    mock_run_align.return_value = [(0, -50), (300, -52)]
    runner = CliRunner()
    ref_file = tmp_path / "ref.mkv"
    sec_file = tmp_path / "sec.mkv"
    ref_file.touch()
    sec_file.touch()

    result = runner.invoke(
        cli,
        ["align", str(ref_file), str(sec_file), "--output", "report.json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Comparing Ref Track 0 to Sec Track 0" in result.output
    assert "    [00:00:00] -50ms" in result.output
    assert "    [00:05:00] -52ms" in result.output
    mock_run_align.assert_called_once()


@patch("audio_aligner.main.run_align")
def test_intra_compare_command(mock_run_align, tmp_path):
    mock_run_align.return_value = [(0, 0)]
    runner = CliRunner()
    test_file = tmp_path / "test.mkv"
    test_file.touch()

    # Mock get_media_info to return 2 tracks
    with patch("audio_aligner.main.get_media_info") as mock_get_info:
        mock_get_info.return_value = (24, 2)
        result = runner.invoke(
            cli, ["intra-compare", str(test_file)], catch_exceptions=False
        )

    assert result.exit_code == 0
    assert "Will compare 1 pairs of tracks" in result.output
    assert "Comparing Track 0 vs Track 1" in result.output
    mock_run_align.assert_called_once()


@patch("audio_aligner.info.print_file_info")
def test_info_command(mock_print_info, tmp_path):
    runner = CliRunner()
    test_file = tmp_path / "test.mkv"
    test_file.touch()

    result = runner.invoke(cli, ["info", str(test_file)], catch_exceptions=False)

    assert result.exit_code == 0
    mock_print_info.assert_called_with(str(test_file)) 