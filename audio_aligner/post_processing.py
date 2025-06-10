from typing import Any, Optional

import click

from audio_aligner.reports import get_reporter


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
        'chunk_delays_ms': [
            {'start_time': int(i * chunk_duration), 'delay': d} for i, d in valid_delays
        ],
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
        if (
            abs(mode_delay) > delay_threshold
            or abs(avg_delay) > delay_threshold
            or abs(max_delay) > delay_threshold
        ):
            issues_found += 1
        click.echo(
            click.style(
                f"    Mode Delay: {mode_delay}ms {'(High Delay!)' if abs(mode_delay) > delay_threshold else '(OK)'}",
                fg='red' if abs(mode_delay) > delay_threshold else 'green',
            )
        )
        click.echo(
            click.style(
                f"    Average Delay: {avg_delay}ms {'(High Delay!)' if abs(avg_delay) > delay_threshold else '(OK)'}",
                fg='red' if abs(avg_delay) > delay_threshold else 'green',
            )
        )
        click.echo(
            click.style(
                f"    Peak Delay: {max_delay}ms {'(High Delay!)' if abs(max_delay) > delay_threshold else '(OK)'}",
                fg='red' if abs(max_delay) > delay_threshold else 'green',
            )
        )

    click.echo('====================')
    click.echo('Alignment Check Complete')
    click.echo('====================')
    click.echo(f'Track Pairs Compared: {len(results)}')
    click.echo(
        click.style(
            f'Issues Found (delay > {delay_threshold}ms): {issues_found}',
            fg="red" if issues_found > 0 else "green",
        )
    )

    if issues_found > 0:
        click.echo('Problem Files:')
        for res in results:
            mode_delay = res['mode_delay_ms']
            avg_delay = res['average_delay_ms']
            max_delay = res['max_delay_ms']
            if (
                abs(mode_delay) > delay_threshold
                or abs(avg_delay) > delay_threshold
                or abs(max_delay) > delay_threshold
            ):
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