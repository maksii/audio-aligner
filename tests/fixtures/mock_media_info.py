from fractions import Fraction
from typing import Any


def mock_media_info(
    fps: Fraction = Fraction(24000, 1001),
    duration: int = 600,
    audio_streams: list[dict[str, Any]] | None = None,
    video_streams: list[dict[str, Any]] | None = None,
) -> dict:
    """
    Mocks container properties (FPS, duration, streams).
    Simulates various codec configurations.
    Generates metadata scenarios.
    """
    if audio_streams is None:
        audio_streams = [
            {
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": 48000,
                "channels": 2,
                "channel_layout": "stereo",
                "metadata": {"language": "eng", "title": "Stereo"},
            }
        ]

    if video_streams is None:
        video_streams = [
            {
                "codec_name": "h264",
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": str(fps),
            }
        ]

    return {
        "format": {"duration": str(float(duration))},
        "streams": video_streams + audio_streams,
    } 