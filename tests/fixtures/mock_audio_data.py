import numpy as np


def generate_reference_audio(
    duration: int, sample_rate: int, pattern_type: str = "sine"
) -> np.ndarray:
    """Generate synthetic audio signals (sine waves, noise, onset patterns)."""
    t = np.linspace(0.0, duration, int(duration * sample_rate), endpoint=False)
    if pattern_type == "sine":
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)
    if pattern_type == "noise":
        return np.random.randn(len(t)).astype(np.float32)
    if pattern_type == "onset":
        return generate_onset_pattern(duration, [1.0, 2.0, 3.0], sample_rate)
    raise ValueError(f"Unknown pattern type: {pattern_type}")


def generate_delayed_audio(
    reference: np.ndarray, delay_ms: int, sample_rate: int, drift_function=None
) -> np.ndarray:
    """Create known delay patterns for validation."""
    delay_samples = int(delay_ms * sample_rate / 1000)
    if delay_samples >= 0:
        delayed_audio = np.pad(reference, (delay_samples, 0), "constant")
        return delayed_audio[: len(reference)]
    else:
        delayed_audio = reference[-delay_samples:]
        delayed_audio = np.pad(delayed_audio, (0, -delay_samples), "constant")
        return delayed_audio


def generate_onset_pattern(
    duration: int, onset_times: list[float], sample_rate: int
) -> np.ndarray:
    """Generate a simple signal with onsets."""
    length = int(duration * sample_rate)
    signal = np.zeros(length, dtype=np.float32)
    for onset_time in onset_times:
        onset_sample = int(onset_time * sample_rate)
        if 0 <= onset_sample < length:
            signal[onset_sample : onset_sample + 100] = np.hanning(100)
    return signal


def add_realistic_noise(audio_data: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise to a signal at a given SNR."""
    signal_power = np.mean(audio_data**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_data)).astype(
        np.float32
    )
    return audio_data + noise 