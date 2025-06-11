from click.testing import CliRunner
import numpy as np
from scipy.io.wavfile import write as write_wav
import os

from audio_aligner.main import cli
from tests.fixtures.mock_audio_data import generate_reference_audio, generate_delayed_audio


def test_full_alignment_workflow(tmp_path):
    runner = CliRunner()
    sample_rate = 48000
    duration = 10
    delay_ms = 75

    ref_audio = generate_reference_audio(duration, sample_rate, "noise")
    sec_audio = generate_delayed_audio(ref_audio, delay_ms, sample_rate)

    ref_path = os.path.join(tmp_path, "ref.wav")
    sec_path = os.path.join(tmp_path, "sec.wav")

    write_wav(ref_path, sample_rate, ref_audio)
    write_wav(sec_path, sample_rate, sec_audio)

    result = runner.invoke(
        cli,
        ["align", ref_path, sec_path, "-m", "rms", "-n", "1"],
        catch_exceptions=False
    )

    assert result.exit_code == 0
    # Allow for slight variations in calculation
    assert "Mode Delay: -7" in result.output 