import hypothesis.strategies as st
from hypothesis import given, settings

from audio_aligner.processing import get_chunks, process_single_chunk, share_arrays
from tests.fixtures.mock_audio_data import generate_reference_audio, generate_delayed_audio


@given(
    delay_ms=st.integers(min_value=-150, max_value=150),
    duration=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=50, deadline=None)
def test_detect_delay_property(delay_ms: int, duration: int):
    """Property-based test: for a random *delay_ms* within ±150 ms the algorithm
    should recover (approximately) the negative of that delay.
    """

    sample_rate = 48_000

    ref_audio = generate_reference_audio(duration, sample_rate, "onset")
    sec_audio = generate_delayed_audio(ref_audio, delay_ms, sample_rate)

    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration=duration, sr=sample_rate)

    result = process_single_chunk((sample_rate, "onset", chunks[0]))
    # The algorithm may fail for edge-case silence; ensure we got a result
    assert result is not None
    _, detected_delay = result

    # Remember: Positive *delay_ms* => secondary audio is late → detected delay
    # must be *negative* to realign, hence we expect ≈ -delay_ms.
    acceptable_error = 5  # milliseconds
    assert abs(detected_delay - (-delay_ms)) <= acceptable_error 