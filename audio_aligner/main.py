import ctypes
import itertools
import multiprocessing
import sys
from fractions import Fraction
from pathlib import Path

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import TYPE_CHECKING, Any, NewType, TypedDict

import click
import numpy as np

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedArray

from audio_aligner.post_processing import build_results, print_results
from audio_aligner.processing import (
    get_chunks,
    init_worker,
    process_single_chunk,
    share_arrays,
)
from audio_aligner.reports import get_reporter
from audio_aligner.utils import validate_non_negative_integer, validate_positive_integer
from audio_aligner.video import get_media_info, load_audio_track

AlignmentResult = NewType('AlignmentResult', dict[str, Any])

VIDEO_EXTS = (
    '*.mkv',
    '*.mp4',
    '*.mov',
    '*.avi',
    '*.webm',
)
AUDIO_EXTS = (
    '*.mp3',
    '*.wav',
    '*.flac',
    '*.aac',
    '*.ogg',
    '*.m4a',
    '*.opus',
    '*.ac3',
    '*.eac3',
    '*.dts',
    '*.dtshd',
    '*.thd',
)
MEDIA_EXTS = VIDEO_EXTS + AUDIO_EXTS


class Alignment(TypedDict):
    reference_video: str
    secondary_video: str
    ref_audio_track: int
    sec_audio_track: int
    method: str
    max_duration: int
    chunk_duration: int
    sample_rate: int
    num_workers: int
    output: str | None
    output_format: Literal['json', 'csv']
    delay_threshold: int
    target_fps: Fraction | None = None


def run_align(
    reference_video: str,
    secondary_video: str,
    ref_audio_track: int,
    sec_audio_track: int,
    method: str,
    max_duration: int,
    chunk_duration: int,
    sample_rate: int,
    num_workers: int,
    target_fps: Fraction | None = None,
) -> list[tuple[int, int]] | None:
    if target_fps is None:
        target_fps, _ = get_media_info(reference_video)

    reference_y = load_audio_track(
        reference_video,
        ref_audio_track,
        sample_rate,
        max_duration,
        target_fps=target_fps,
    )
    secondary_y = load_audio_track(
        secondary_video,
        sec_audio_track,
        sample_rate,
        max_duration,
        target_fps=target_fps,
    )

    chunk_tasks = get_chunks(reference_y, secondary_y, chunk_duration, sample_rate)
    if not chunk_tasks:
        click.echo(
            click.style(
                'Error: No processable audio chunks found. Check durations.',
                fg='red',
            ),
            err=True,
        )
        return None

    shared_ref: np.ndarray | SynchronizedArray
    shared_sec: np.ndarray | SynchronizedArray

    if num_workers > 1:
        shared_ref = multiprocessing.Array(ctypes.c_float, reference_y.size)
        shared_sec = multiprocessing.Array(ctypes.c_float, secondary_y.size)

        np.copyto(np.frombuffer(shared_ref.get_obj(), dtype=np.float32), reference_y)
        np.copyto(np.frombuffer(shared_sec.get_obj(), dtype=np.float32), secondary_y)
    else:
        shared_ref = reference_y
        shared_sec = secondary_y

    share_arrays(shared_ref, shared_sec)

    worker_args = [(sample_rate, method, task_info) for task_info in chunk_tasks]

    chunk_delays_results = []
    bar_label = (
        f'Processing {len(chunk_tasks)} chunks ({min(num_workers, len(chunk_tasks))} workers)'
    )
    with click.progressbar(
        length=len(chunk_tasks),
        label=bar_label,
    ) as bar:
        if num_workers > 1:
            with multiprocessing.Pool(
                processes=min(num_workers, len(chunk_tasks)),
                initializer=init_worker,
                initargs=(shared_ref, shared_sec),
            ) as pool:
                result_iterator = pool.imap_unordered(process_single_chunk, worker_args)
                for result in result_iterator:
                    chunk_delays_results.append(result)
                    bar.update(1)
        else:
            for worker_arg in worker_args:
                result = process_single_chunk(worker_arg)
                chunk_delays_results.append(result)
                bar.update(1)

    valid_delays = [d for d in chunk_delays_results if d is not None]
    if not valid_delays:
        click.echo(
            click.style('Error: No valid delays calculated from any chunk.', fg='red'),
            err=True,
        )
        return None

    valid_delays.sort()
    return valid_delays


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli() -> None:
    pass


@cli.command()
@click.argument('videos', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--sec-folder',
    'sec_folder_path',
    type=click.Path(exists=True, file_okay=False),
    help='Path to a folder with secondary videos.',
)
@click.option(
    '-ra',
    '--ref-audio-track',
    'ref_audio_track',
    type=int,
    default=0,
    show_default=True,
    help='Audio track number (0-indexed) from the reference video.',
)
@click.option(
    '-sa',
    '--sec-audio-track',
    'sec_audio_track',
    type=int,
    default=0,
    show_default=True,
    help='Audio track number (0-indexed) from the secondary video.',
)
@click.option(
    '-a',
    '--all-tracks',
    'all_tracks',
    is_flag=True,
    default=False,
    help='Compare a reference tracks to all tracks in a secondary video.',
)
@click.option(
    '-m',
    '--method',
    type=click.Choice(['rms', 'onset'], case_sensitive=False),
    default='onset',
    show_default=True,
    help='Algorithm for feature extraction and comparison.',
)
@click.option(
    '-md',
    '--max-duration',
    'max_duration',
    type=int,
    default=0,
    show_default=True,
    callback=validate_non_negative_integer,
    help='Maximum duration (seconds) from the start of each video to process. 0 for full length.',
)
@click.option(
    '-cd',
    '--chunk-duration',
    'chunk_duration',
    type=float,
    default=300,
    show_default=True,
    callback=validate_positive_integer,
    help='Duration (seconds) of audio chunks for parallel processing.',
)
@click.option(
    '-sr',
    '--sample-rate',
    'sample_rate',
    type=int,
    default=48000,
    show_default=True,
    help='Target sampling rate (Hz) for audio processing.',
)
@click.option(
    '-n',
    '--num-workers',
    'num_workers',
    type=int,
    default=1,
    show_default=True,
    callback=validate_positive_integer,
    help='Number of parallel worker processes.',
)
@click.option(
    '--output',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Path to save the report file.',
)
@click.option(
    '--output-format',
    type=click.Choice(['json', 'csv'], case_sensitive=False),
    default='json',
    show_default=True,
    help='Output report format.',
)
@click.option(
    '--delay-threshold',
    type=int,
    default=42,
    show_default=True,
    help='Sets the delay threshold for highlighting issues.',
)
def align(
    videos: tuple[str, ...],
    sec_folder_path: str,
    ref_audio_track: int,
    sec_audio_track: int,
    all_tracks: bool,
    method: str,
    max_duration: int,
    chunk_duration: int,
    sample_rate: int,
    num_workers: int,
    output: str | None,
    output_format: Literal['json', 'csv'],
    delay_threshold: int,
) -> None:
    if not videos:
        raise click.UsageError('Missing video file arguments.')

    reference_video = videos[0]
    secondary_videos = videos[1:]

    all_results: list[AlignmentResult] = []

    files_to_process = list(secondary_videos)
    if sec_folder_path:
        folder = Path(sec_folder_path)
        for ext in MEDIA_EXTS:
            files_to_process.extend(str(p) for p in folder.glob(ext))

    if not files_to_process:
        raise click.UsageError(
            'No secondary videos found. Provide them as arguments or with --sec-folder.'
        )

    for secondary_video in files_to_process:
        click.echo('--- Processing pair ---')
        click.echo(f"  Ref: {click.style(reference_video, fg='cyan')}")
        click.echo(f"  Sec: {click.style(secondary_video, fg='cyan')}")
        try:
            target_fps, _ = get_media_info(reference_video)
            _, sec_track_count = get_media_info(secondary_video)

            if all_tracks:
                click.echo(
                    f'  Comparing Ref Track {ref_audio_track} to all {sec_track_count} tracks in secondary.'
                )
                for i in range(sec_track_count):
                    results = run_align(
                        reference_video,
                        secondary_video,
                        ref_audio_track,
                        i,
                        method,
                        max_duration,
                        chunk_duration,
                        sample_rate,
                        num_workers,
                        target_fps=target_fps,
                    )
                    if results:
                        all_results.append(
                            build_results(
                                results,
                                'align',
                                chunk_duration,
                                float(1000 / target_fps),
                                delay_threshold,
                                (reference_video, ref_audio_track),
                                (secondary_video, i),
                            )
                        )
            else:
                click.echo(f'  Comparing Ref Track {ref_audio_track} to Sec Track {sec_audio_track}.')
                results = run_align(
                    reference_video,
                    secondary_video,
                    ref_audio_track,
                    sec_audio_track,
                    method,
                    max_duration,
                    chunk_duration,
                    sample_rate,
                    num_workers,
                    target_fps=target_fps,
                )
                if results:
                    all_results.append(
                        build_results(
                            results,
                            'align',
                            chunk_duration,
                            float(1000 / target_fps),
                            delay_threshold,
                            (reference_video, ref_audio_track),
                            (secondary_video, sec_audio_track),
                        )
                    )
        except Exception as e:
            click.echo(click.style(f'  Error processing pair: {e}', fg='red'), err=True)

    print_results(all_results, delay_threshold, output, output_format)


@cli.command('intra-compare')
@click.argument('video_files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--folder',
    'folder_path',
    type=click.Path(exists=True, file_okay=False),
    help='Check all videos in a folder.',
)
@click.option(
    '-m',
    '--method',
    type=click.Choice(['rms', 'onset'], case_sensitive=False),
    default='onset',
    show_default=True,
    help='Algorithm for feature extraction and comparison.',
)
@click.option(
    '-md',
    '--max-duration',
    'max_duration',
    type=int,
    default=0,
    show_default=True,
    callback=validate_non_negative_integer,
    help='Maximum duration (seconds) from the start of each video to process. 0 for full length.',
)
@click.option(
    '-cd',
    '--chunk-duration',
    'chunk_duration',
    type=float,
    default=300,
    show_default=True,
    callback=validate_positive_integer,
    help='Duration (seconds) of audio chunks for parallel processing.',
)
@click.option(
    '-sr',
    '--sample-rate',
    'sample_rate',
    type=int,
    default=48000,
    show_default=True,
    help='Target sampling rate (Hz) for audio processing.',
)
@click.option(
    '-n',
    '--num-workers',
    'num_workers',
    type=int,
    default=1,
    show_default=True,
    callback=validate_positive_integer,
    help='Number of parallel worker processes.',
)
@click.option(
    '--output',
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help='Path to save the report file.',
)
@click.option(
    '--output-format',
    type=click.Choice(['json', 'csv'], case_sensitive=False),
    default='json',
    show_default=True,
    help='Output report format.',
)
@click.option(
    '--delay-threshold',
    type=int,
    default=42,
    show_default=True,
    help='Sets the delay threshold for highlighting issues.',
)
def intra_compare(
    video_files: tuple[str, ...],
    folder_path: str,
    method: str,
    max_duration: int,
    chunk_duration: int,
    sample_rate: int,
    num_workers: int,
    output: str | None,
    output_format: Literal['json', 'csv'],
    delay_threshold: int,
) -> None:
    files_to_process = list(video_files)
    if folder_path:
        folder = Path(folder_path)
        for ext in VIDEO_EXTS:
            files_to_process.extend(str(p) for p in folder.glob(ext))
    all_results: list[AlignmentResult] = []

    for file in files_to_process:
        click.echo(f"--- Processing file: {click.style(file, fg='yellow')} ---")
        try:
            click.echo('  Inspecting file for media info...')
            target_fps, track_count = get_media_info(file)
            click.echo(f'  Found {track_count} audio tracks.')
            click.echo(f'  Detected framerate: {target_fps}')

            if track_count < 2:
                click.echo('  Skipping: Not enough audio tracks to compare (requires at least 2).')
                continue

            track_combinations = list(itertools.combinations(range(track_count), 2))
            click.echo(f'  Will compare {len(track_combinations)} pairs of tracks.')

            for track1, track2 in track_combinations:
                click.echo(f'  - Comparing Track {track1} vs Track {track2}:')
                results = run_align(
                    file,
                    file,
                    track1,
                    track2,
                    method,
                    max_duration,
                    chunk_duration,
                    sample_rate,
                    num_workers,
                    target_fps=target_fps,
                )
                if results:
                    all_results.append(
                        build_results(
                            results,
                            'intra-compare',
                            chunk_duration,
                            float(1000 / target_fps),
                            delay_threshold,
                            (file, track1),
                            (file, track2),
                        )
                    )
        except Exception as e:
            click.echo(click.style(f'  Error processing file {file}: {e}', fg='red'), err=True)

    print_results(all_results, delay_threshold, output, output_format)


@cli.command('info')
@click.argument('video_file', type=click.Path(exists=True, dir_okay=False))
def info(video_file: str) -> None:
    from audio_aligner.info import print_file_info

    print_file_info(video_file)


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        cli(sys.argv[1:])
    else:
        cli()

