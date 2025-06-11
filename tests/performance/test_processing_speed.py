import pytest
from click.testing import CliRunner
from scipy.io.wavfile import write as write_wav
import os
import numpy as np

from audio_aligner.main import cli
from tests.fixtures.mock_audio_data import generate_reference_audio


@pytest.mark.benchmark(group="speed")
def test_processing_speed(benchmark, tmp_path):
    runner = CliRunner()
    sample_rate = 48000
    duration = 30  # 30 seconds of audio

    ref_audio = generate_reference_audio(duration, sample_rate, "noise")
    ref_path = os.path.join(tmp_path, "ref.wav")
    write_wav(ref_path, sample_rate, ref_audio)

    def run_align():
        runner.invoke(
            cli,
            ["align", ref_path, ref_path, "-m", "onset"],
            catch_exceptions=False
        )

    benchmark(run_align) 