import numpy as np
import pytest

from audio_aligner.processing import (
    get_chunks,
    process_single_chunk,
    share_arrays,
    init_worker,
)
from tests.fixtures.mock_audio_data import (
    generate_delayed_audio,
    generate_onset_pattern,
    generate_reference_audio,
)


@pytest.fixture
def sample_rate():
    return 48000


def test_onset_detection_with_known_patterns(sample_rate):
    duration = 5
    onset_times = [1.0, 2.5, 4.0]
    ref_audio = generate_onset_pattern(duration, onset_times, sample_rate)
    sec_audio = ref_audio.copy()

    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration=5, sr=sample_rate)
    task_args = (sample_rate, "onset", chunks[0])
    result = process_single_chunk(task_args)

    assert result is not None
    _, detected_delay = result
    assert abs(detected_delay) <= 1  # Should be perfectly aligned


def test_rms_calculation_accuracy(sample_rate):
    duration = 5
    ref_audio = generate_reference_audio(duration, sample_rate, "sine")
    sec_audio = ref_audio.copy()

    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration=5, sr=sample_rate)
    task_args = (sample_rate, "rms", chunks[0])
    result = process_single_chunk(task_args)

    assert result is not None
    _, detected_delay = result
    assert abs(detected_delay) <= 1


def test_cross_correlation_with_synthetic_delay(sample_rate):
    duration = 10
    delay_ms = 50
    ref_audio = generate_reference_audio(duration, sample_rate, "onset")
    sec_audio = generate_delayed_audio(ref_audio, delay_ms, sample_rate)

    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration=5, sr=sample_rate)
    
    # We expect only one chunk since duration is short
    assert len(chunks) > 0
    
    task_args = (sample_rate, "onset", chunks[0])
    
    result = process_single_chunk(task_args)

    assert result is not None
    start_time, detected_delay = result

    assert start_time == 0
    # A positive delay_ms means the secondary audio is shifted *later*,
    # so the detected delay to align it should be *negative*.
    assert abs(detected_delay - (-delay_ms)) <= 2


def test_chunk_boundary_handling(sample_rate):
    chunk_duration = 5
    delay_ms = 100
    # Create audio that is longer than one chunk
    duration = chunk_duration * 2
    ref_audio = generate_reference_audio(duration, sample_rate, "noise")
    sec_audio = generate_delayed_audio(ref_audio, delay_ms, sample_rate)

    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration, sample_rate)
    assert len(chunks) == 2

    # Check first chunk
    task_args_1 = (sample_rate, "onset", chunks[0])
    result_1 = process_single_chunk(task_args_1)
    assert result_1 is not None
    _, detected_delay_1 = result_1
    assert abs(detected_delay_1 - (-delay_ms)) <= 25

    # Check second chunk
    task_args_2 = (sample_rate, "onset", chunks[1])
    result_2 = process_single_chunk(task_args_2)
    assert result_2 is not None
    _, detected_delay_2 = result_2
    assert abs(detected_delay_2 - (-delay_ms)) <= 25


def test_multiprocessing_consistency():
    """Ensure that running the alignment logic in parallel gives the same
    results as running it serially for the same inputs.

    The test keeps the audio duration short so that spawning the pool is fast
    and memory-friendly inside the CI environment. We use the same
    ``process_single_chunk`` worker that the CLI relies on and reproduce the
    Pool initialisation sequence from ``run_align``.
    """

    import ctypes
    import multiprocessing as mp

    # Generate a simple synthetic signal and a delayed copy
    sample_rate = 48_000
    duration = 5  # seconds
    delay_ms = 80
    ref_audio = generate_reference_audio(duration, sample_rate, "onset")
    sec_audio = generate_delayed_audio(ref_audio, delay_ms, sample_rate)

    # Prepare one chunk covering the full signal
    share_arrays(ref_audio, sec_audio)
    chunks = get_chunks(ref_audio, sec_audio, chunk_duration=duration, sr=sample_rate)
    assert len(chunks) == 1
    task_args = (sample_rate, "onset", chunks[0])

    # Serial execution
    serial_result = process_single_chunk(task_args)

    # Parallel execution with a pool of 2 workers
    shared_ref = mp.Array(ctypes.c_float, ref_audio.size)
    shared_sec = mp.Array(ctypes.c_float, sec_audio.size)
    import numpy as _np

    _np.copyto(_np.frombuffer(shared_ref.get_obj(), dtype=_np.float32), ref_audio)
    _np.copyto(_np.frombuffer(shared_sec.get_obj(), dtype=_np.float32), sec_audio)

    with mp.Pool(processes=2, initializer=init_worker, initargs=(shared_ref, shared_sec)) as pool:
        parallel_result = pool.apply(process_single_chunk, (task_args,))

    # Both approaches should yield approximately the same delay (±2 ms)
    assert serial_result is not None
    assert parallel_result is not None
    _, serial_delay = serial_result
    _, parallel_delay = parallel_result
    assert abs(serial_delay - parallel_delay) <= 2 