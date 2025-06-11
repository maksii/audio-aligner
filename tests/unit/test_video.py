from fractions import Fraction
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from audio_aligner.video import get_media_info, load_audio_track


def test_video_info_extraction(mocker):
    mock_video_stream = MagicMock()
    mock_video_stream.average_rate = Fraction(24, 1)
    mock_container = MagicMock()
    mock_container.__enter__.return_value = mock_container
    mock_container.streams.video = [mock_video_stream]
    mock_container.streams.audio = [MagicMock(), MagicMock()]
    mocker.patch("av.open", return_value=mock_container)

    fps, audio_tracks = get_media_info("dummy.mkv")
    assert fps == Fraction(24, 1)
    assert audio_tracks == 2


def test_audio_loading(mocker):
    # Mock av.open to simulate decoding
    mock_frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
    mock_frame.pts = 1
    mock_frame.time_base = Fraction(1, 48000)
    mock_container = MagicMock()
    mock_container.__enter__.return_value = mock_container
    mock_container.decode.return_value = [mock_frame]
    mock_audio_stream = MagicMock()
    mock_audio_stream.time_base = Fraction(1, 48000)
    mock_audio_stream.duration = 48000 * 5  # 5 seconds
    mock_audio_stream.metadata = {}
    mock_audio_stream.layout = av.AudioLayout("mono")  # Valid layout
    mock_container.streams.audio = [mock_audio_stream]

    # Mock the resampler
    mock_resampler = MagicMock()
    mock_resampler.resample.return_value = [mock_frame]
    mocker.patch("av.AudioResampler", return_value=mock_resampler)

    # Mock librosa.load since we are not dealing with a real wav buffer
    mocker.patch("librosa.load", return_value=(np.zeros(48000 * 5), 48000))
    mocker.patch("av.open", return_value=mock_container)
    mocker.patch("os.path.exists", return_value=False)  # Ensure no cache hit
    mocker.patch("os.stat", return_value=MagicMock(st_mtime=1))
    mocker.patch(
        "hashlib.md5",
        return_value=MagicMock(hexdigest=MagicMock(return_value="dummyhash")),
    )
    mocker.patch("numpy.save")

    audio = load_audio_track("dummy.mkv", 0, 48000, 5, Fraction(24, 1))
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 48000 * 5


def test_caching_logic(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("numpy.load", return_value=np.array([1, 2, 3]))
    mocker.patch("os.stat", return_value=MagicMock(st_mtime=1))
    mocker.patch(
        "hashlib.md5",
        return_value=MagicMock(hexdigest=MagicMock(return_value="dummyhash")),
    )

    audio = load_audio_track("dummy.mkv", 0, 48000, 5, Fraction(24, 1))
    np.load.assert_called_once()
    assert np.array_equal(audio, np.array([1, 2, 3])) 