import os
from multiprocessing.sharedctypes import Array, SynchronizedArray
from typing import Any

import click
import numpy as np
from librosa import feature, onset  # type: ignore[attr-defined]
from scipy.signal import correlate

shared_ref: np.ndarray | SynchronizedArray | None = None
shared_sec: np.ndarray | SynchronizedArray | None = None

def init_worker(ref: Array, sec: Array) -> None:
    global shared_ref, shared_sec
    shared_ref = ref
    shared_sec = sec

def share_arrays(
    ref_: np.ndarray | SynchronizedArray,
    sec_: np.ndarray | SynchronizedArray,
) -> None:
    global shared_ref, shared_sec
    shared_ref = ref_
    shared_sec = sec_


def get_chunks(
    ref_y: np.ndarray, sec_y: np.ndarray, chunk_duration: float, sr: int
) -> list[dict[str, Any]]:
    ref_duration = len(ref_y) / sr
    sec_duration = len(sec_y) / sr
    min_duration = min(ref_duration, sec_duration)

    if min_duration < chunk_duration:
        return [
            {
                'id': 0,
                'start': 0,
                'ref_slice': (0, len(ref_y)),
                'sec_slice': (0, len(sec_y)),
            }
        ]

    chunk_size = int(chunk_duration * sr)
    ref_chunks_indices = [
        (i, i + chunk_size) for i in range(0, len(ref_y), chunk_size)
    ]
    sec_chunks_indices = [
        (i, i + chunk_size) for i in range(0, len(sec_y), chunk_size)
    ]

    min_chunks = min(len(ref_chunks_indices), len(sec_chunks_indices))

    tasks = []
    for i in range(min_chunks):
        ref_start, ref_end = ref_chunks_indices[i]
        sec_start, sec_end = sec_chunks_indices[i]
        tasks.append(
            {
                'id': i,
                'start': int(ref_start / sr),
                'ref_slice': (ref_start, ref_end),
                'sec_slice': (sec_start, sec_end),
            }
        )
    return tasks


def process_single_chunk(
    task_args: tuple[int, str, dict],
) -> tuple[int, int] | None:
    sr, method_name, task_info = task_args

    if shared_ref is None or shared_sec is None:
        raise RuntimeError('shared_ref and shared_sec must be set')

    if isinstance(shared_ref, np.ndarray):
        y_ref_full_data = shared_ref
    else:
        y_ref_full_data = np.frombuffer(shared_ref.get_obj(), dtype=np.float32)

    if isinstance(shared_sec, np.ndarray):
        y_sec_full_data = shared_sec
    else:
        y_sec_full_data = np.frombuffer(shared_sec.get_obj(), dtype=np.float32)

    ref_start, ref_end = task_info['ref_slice']
    sec_start, sec_end = task_info['sec_slice']

    y_ref_chunk = y_ref_full_data[ref_start:ref_end]
    y_sec_chunk = y_sec_full_data[sec_start:sec_end]

    try:
        hop_length = int(sr / 1000) + 1
        frame_length = hop_length * 12

        if method_name == 'rms':
            y_ref_feat = feature.rms(
                y=y_ref_chunk,
                frame_length=frame_length,
                hop_length=hop_length,
            )[0]
            y_sec_feat = feature.rms(
                y=y_sec_chunk,
                frame_length=frame_length,
                hop_length=hop_length,
            )[0]
        elif method_name == 'onset':
            y_ref_feat = onset.onset_strength(y=y_ref_chunk, sr=sr, hop_length=hop_length)
            y_sec_feat = onset.onset_strength(y=y_sec_chunk, sr=sr, hop_length=hop_length)
        else:
            raise ValueError("Unsupported method. Choose 'rms' or 'onset'.")

        y_ref_feat = (y_ref_feat - np.mean(y_ref_feat)) / np.std(y_ref_feat)
        y_sec_feat = (y_sec_feat - np.mean(y_sec_feat)) / np.std(y_sec_feat)

        correlation = correlate(y_ref_feat, y_sec_feat, mode='full')
        lags = np.arange(-len(y_sec_feat) + 1, len(y_ref_feat))

        delay_frames = lags[np.argmax(correlation)]
        delay_seconds = delay_frames * hop_length / sr
        return task_info['start'], int(delay_seconds * 1000)
    except Exception as e:
        click.echo(
            click.style(
                f"Worker (PID {os.getpid()}) Error in chunk {task_info['id']}: {e}",
                fg='red',
            ),
            err=True,
        )
        return None
