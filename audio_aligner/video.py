import io
import os
import hashlib
import tempfile
from fractions import Fraction

import av
import click
import librosa
import numpy as np

AV_TIME_BASE_Q = 1000000

DEFAULT_FPS = Fraction(24000, 1001)


def get_media_info(video_path: str) -> tuple[Fraction, int]:
    """Opens a media file once to get key information: FPS and audio track count."""
    fps = DEFAULT_FPS
    audio_track_count = 0
    try:
        with av.open(video_path, 'r') as container:
            audio_track_count = len(container.streams.audio)
            if container.streams.video:
                s = container.streams.video[0]
                if s.average_rate and s.average_rate.denominator != 0:
                    fps = s.average_rate
                elif s.guessed_rate and s.guessed_rate.denominator != 0:
                    fps = s.guessed_rate
                elif s.codec_context and s.codec_context.framerate and s.codec_context.framerate.denominator != 0:
                    fps = s.codec_context.framerate
    except Exception as e:
        click.echo(
            click.style(f'Could not fully inspect {video_path}, falling back to defaults. Error: {e}', fg='red'),
            err=True,
        )
    return fps, audio_track_count


def get_media_json(video_path: str) -> dict:
    """Opens a media file and returns the JSON representation of its streams."""
    try:
        with av.open(video_path, 'r') as container:
            streams_data = []
            for stream in container.streams:
                stream_data = {
                    'codec_name': stream.codec_context.name,
                    'codec_type': stream.type,
                }
                if stream.type == 'video':
                    stream_data['avg_frame_rate'] = str(stream.average_rate)
                streams_data.append(stream_data)
            return {"streams": streams_data}
    except Exception as e:
        click.echo(
            click.style(f'Could not read media information from {video_path}. Error: {e}', fg='red'),
            err=True,
        )
        return {}


def load_audio_track(
    video_path: str,
    audio_track: int,
    sample_rate: int,
    max_duration: int,
    target_fps: Fraction,
) -> np.ndarray:
    cache_dir = os.path.join(tempfile.gettempdir(), 'audio_aligner_cache')
    os.makedirs(cache_dir, exist_ok=True)

    try:
        file_stat = os.stat(video_path)
        cache_key_str = f"{video_path}{file_stat.st_mtime}{audio_track}{sample_rate}{max_duration}"
        cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_key_hash}.npy")

        if os.path.exists(cache_file):
            click.echo(f"Loading from cache: {cache_file}")
            return np.load(cache_file)
    except Exception as e:
        click.echo(click.style(f'Coud not check cache, got {e}', fg='red'), err=True)

    input_container = av.open(video_path, 'r')

    input_fps, _ = get_media_info(video_path)
    audio_speed_factor = float(target_fps / input_fps)

    input_audio_stream = input_container.streams.audio[audio_track]
    input_audio_stream.thread_type = 'AUTO'

    title = input_audio_stream.metadata.get('title')
    language = input_audio_stream.metadata.get('language')
    loading_text = 'Loading audio track number'
    if title:
        loading_text += f' {title}'
    if language:
        loading_text += f' ({language})'
    loading_text += f' from {audio_track}:{video_path}'
    click.echo(loading_text)

    if audio_speed_factor != 1.0:
        # TODO
        click.echo(
            click.style(
                f'FPS mismatch detected! Speed factor: {audio_speed_factor:.2f}',
                fg='yellow',
            ),
        )

    if (
        input_audio_stream.duration is not None
        and input_audio_stream.time_base is not None
        and input_audio_stream.time_base != 0
    ):
        stream_duration_seconds = int(
            input_audio_stream.duration * input_audio_stream.time_base,
        )
    else:
        stream_duration_seconds = int(input_container.duration / AV_TIME_BASE_Q)

    if not max_duration or max_duration > stream_duration_seconds:
        max_duration = stream_duration_seconds

    output_codec = 'pcm_s16le'
    output_layout = 'mono'
    output_format = 's16'

    wav_buffer = io.BytesIO()

    output_container = None
    try:
        output_container = av.open(wav_buffer, mode='w', format='wav')

        output_audio_stream = output_container.add_stream(
            output_codec,
            rate=sample_rate,
            layout=output_layout,
        )

        resampler = av.AudioResampler(
            format=output_format,
            layout=output_layout,
            rate=sample_rate,
        )

        with click.progressbar(length=max_duration, label='Converting audio to WAV') as bar:
            bar.update(0, current_item=0)
            for frame in input_container.decode(input_audio_stream):
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    for packet in output_audio_stream.encode(resampled_frame):
                        output_container.mux(packet)

                current_frame_time = float(frame.pts * input_audio_stream.time_base)
                bar.update(
                    current_frame_time - bar.current_item,
                    current_item=current_frame_time,
                )
                if max_duration and current_frame_time > max_duration:
                    bar.update(max_duration)
                    break

        resampled_frames = resampler.resample(None)
        for resampled_frame in resampled_frames:
            for packet in output_audio_stream.encode(resampled_frame):
                output_container.mux(packet)

        for packet in output_audio_stream.encode():
            output_container.mux(packet)

        wav_buffer.seek(0)
        y, _ = librosa.load(wav_buffer, sr=None)

        try:
            np.save(cache_file, y)
            click.echo(f"Saved to cache: {cache_file}")
        except Exception as e:
            click.echo(click.style(f'Coud not save to cache, got {e}', fg='red'), err=True)

        return y

    finally:
        input_container.close()
        if output_container:
            output_container.close()
        wav_buffer.close()
