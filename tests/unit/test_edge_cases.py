import numpy as np

from audio_aligner.processing import get_chunks, process_single_chunk, share_arrays
from tests.fixtures.mock_audio_data import generate_reference_audio


def test_silence_only_audio_returns_none():
    """When both reference and secondary audio are silence, the algorithm should
    not crash and will likely return ``None`` since standardisation divides by
    zero. The test confirms graceful failure (returning ``None``) rather than an
    uncaught exception.
    """

    sample_rate = 48_000
    duration = 3
    silence = np.zeros(duration * sample_rate, dtype=np.float32)

    share_arrays(silence, silence)
    chunks = get_chunks(silence, silence, chunk_duration=duration, sr=sample_rate)
    result = process_single_chunk((sample_rate, "rms", chunks[0]))

    # The implementation may either return ``None`` (if it raises internally)
    # or an arbitrary delay. We only assert that no unhandled exception
    # bubbles up and the function returns *something*.
    assert result is None or isinstance(result, tuple)


def test_variable_sample_rate_handling():
    """Verify that the pipeline works with an alternative sample rate (44.1 kHz)."""

    sample_rate = 44100
    duration = 4

    ref_audio = generate_reference_audio(duration, sample_rate, "sine")
    sec_audio = ref_audio.copy()

    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration=duration, sr=sample_rate)
    result = process_single_chunk((sample_rate, "rms", chunks[0]))

    # Zero delay expected; tolerance ±1 ms at lower sample rate
    assert result is not None
    _, detected_delay = result
    assert abs(detected_delay) <= 1 