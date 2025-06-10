from typing import Any, Optional

import click
import numpy as np

from audio_aligner.reports import get_reporter


def find_delay_groups(
    chunk_delays_ms: list[dict[str, Any]],
    grouping_threshold_ms: int = 5,
    min_group_size: int = 3,
) -> list[dict[str, Any]]:
    """
    Identifies groups of consecutive chunks with stable delays.
    """
    if len(chunk_delays_ms) < min_group_size:
        return []

    groups = []
    current_group: list[dict[str, Any]] = []

    for chunk in chunk_delays_ms:
        if not current_group:
            current_group.append(chunk)
            continue

        avg_delay = np.mean([c['delay'] for c in current_group])

        if abs(chunk['delay'] - avg_delay) <= grouping_threshold_ms:
            current_group.append(chunk)
        else:
            if len(current_group) >= min_group_size:
                groups.append(list(current_group))
            current_group = [chunk]

    if len(current_group) >= min_group_size:
        groups.append(current_group)

    processed_groups = []
    for group in groups:
        delays = [d['delay'] for d in group]
        avg_delay = int(np.mean(delays))
        processed_groups.append(
            {
                "start_time": group[0]['start_time'],
                "end_time": group[-1]['start_time'],
                "count": len(group),
                "average_delay": avg_delay,
            }
        )

    return processed_groups


def build_results(
    valid_delays: list[tuple[int, int]],
    command: str,
    chunk_duration: int,
    frame_duration: float,
    delay_threshold: int,
    ref: tuple[str, int],
    sec: tuple[str, int],
) -> dict[str, Any]:
    integers_delays = [d[1] for d in valid_delays]
    mode_delay = max(set(integers_delays), key=integers_delays.count)
    average_delay = int(sum(integers_delays) / len(integers_delays))
    max_delay = max(integers_delays, key=lambda d: abs(d))
    min_delay = min(integers_delays, key=lambda d: abs(d))

    chunk_delays = [
        {'start_time': start, 'delay': delay} for start, delay in valid_delays
    ]

    delay_groups = find_delay_groups(chunk_delays)

    return {
        'command': command,
        'reference_file': ref[0],
        'reference_track': ref[1],
        'secondary_file': sec[0],
        'secondary_track': sec[1],
        'mode_delay_ms': mode_delay,
        'average_delay_ms': average_delay,
        'min_delay_ms': min_delay,
        'max_delay_ms': max_delay,
        'delay_threshold': delay_threshold,
        'frame_duration': frame_duration,
        'chunk_duration': chunk_duration,
        'chunk_delays_ms': chunk_delays,
        'delay_groups': delay_groups,
    }


def print_results(
    results: list[dict[str, Any]],
    delay_threshold: int,
    output_path: Optional[str],
    output_format: Optional[str],
) -> None:
    if not results:
        click.echo('No results to print.')
        return

    issues_found = 0
    problem_files = []
    for res in results:
        click.echo('-----------------------------------')
        click.echo(
            f'Processing: {res["reference_file"]} (Track {res["reference_track"]}) vs '
            f'{res["secondary_file"]} (Track {res["secondary_track"]})'
        )
        click.echo('  Delays per chunk:')
        for chunk in res['chunk_delays_ms']:
            start_time = chunk['start_time']
            hours = start_time // 3600
            minutes = (start_time % 3600) // 60
            secs = start_time % 60
            delay = chunk['delay']
            click.echo(
                click.style(
                    f'    [{hours:02}:{minutes:02}:{secs:02}] {delay}ms',
                    fg='red' if abs(delay) > delay_threshold else 'green',
                )
            )
        click.echo('  Summary:')
        mode_delay = res['mode_delay_ms']
        avg_delay = res['average_delay_ms']
        max_delay = res['max_delay_ms']
        is_issue = (
            abs(mode_delay) > delay_threshold
            or abs(avg_delay) > delay_threshold
            or abs(max_delay) > delay_threshold
        )
        click.echo(
            click.style(
                f"    Mode Delay: {mode_delay}ms{' (High Delay!)' if abs(mode_delay) > delay_threshold else ''}",
                fg='red' if abs(mode_delay) > delay_threshold else 'green',
            )
        )
        click.echo(
            click.style(
                f"    Average Delay: {avg_delay}ms{' (High Delay!)' if abs(avg_delay) > delay_threshold else ''}",
                fg='red' if abs(avg_delay) > delay_threshold else 'green',
            )
        )
        click.echo(
            click.style(
                f"    Peak Delay: {max_delay}ms{' (High Delay!)' if abs(max_delay) > delay_threshold else ''}",
                fg='red' if abs(max_delay) > delay_threshold else 'green',
            )
        )

        delay_groups = res.get('delay_groups', [])
        if len(delay_groups) > 1:
            is_issue = True
            click.echo(
                click.style(
                    '    Sync Drift Detected:',
                    fg='yellow',
                )
            )
            for group in delay_groups:
                start = group['start_time']
                end = group['end_time']
                count = group['count']
                avg = group['average_delay']

                sh = start // 3600
                sm = (start % 3600) // 60
                ss = start % 60

                eh = end // 3600
                em = (end % 3600) // 60
                es = end % 60

                click.echo(
                    click.style(
                        f"      - From [{sh:02}:{sm:02}:{ss:02}] to [{eh:02}:{em:02}:{es:02}] ({count} chunks): average delay of {avg}ms",
                        fg='yellow',
                    )
                )

        if is_issue:
            issues_found += 1
            problem_files.append(res)

    click.echo('====================')
    click.echo('Alignment Check Complete')
    click.echo('====================')
    click.echo(f'Track Pairs Compared: {len(results)}')
    click.echo(
        click.style(
            f'Issues Found (delay > {delay_threshold}ms or sync drift): {issues_found}',
            fg="red" if issues_found > 0 else "green",
        )
    )

    if issues_found > 0:
        click.echo('Problem Files:')
        for res in problem_files:
            mode_delay = res['mode_delay_ms']
            avg_delay = res['average_delay_ms']
            max_delay = res['max_delay_ms']
            click.echo(
                click.style(
                    f' - {res["secondary_file"]} (Track {res["reference_track"]} vs {res["secondary_track"]}: '
                    f'mode: {mode_delay}ms, avg: {avg_delay}ms, peak: {max_delay}ms)',
                    fg='red',
                )
            )

    if output_path and output_format:
        reporter = get_reporter(output_path, output_format)
        reporter.write(results)
        click.echo(f'Full report saved to: {output_path}') 