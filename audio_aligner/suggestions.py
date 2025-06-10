from typing import Any, Dict, List, Tuple
import click
from audio_aligner.video import get_media_info, get_media_json


SUGGESTIONS_DB = {
    "ac3": {
        "Delay": {
            "tool": "eac3to",
            "action": "Apply a positive delay to the audio track (add silence).",
            "command": 'eac3to "{sec_file}" "{output_file}" +{delay_ms}ms',
            "note": "eac3to is recommended for AC3 tracks as it can apply delays without re-encoding.",
        },
        "Cut": {
            "tool": "eac3to",
            "action": "Apply a negative delay to the audio track (cut from start).",
            "command": 'eac3to "{sec_file}" "{output_file}" -{delay_ms}ms',
            "note": "eac3to is recommended for AC3 tracks as it can apply delays without re-encoding.",
        },
    },
    "eac3": {
        "Delay": {
            "tool": "eac3to",
            "action": "Apply a positive delay to the audio track (add silence).",
            "command": 'eac3to "{sec_file}" "{output_file}" +{delay_ms}ms',
            "note": "eac3to is recommended for E-AC3 tracks as it can apply delays without re-encoding.",
        },
        "Cut": {
            "tool": "eac3to",
            "action": "Apply a negative delay to the audio track (cut from start).",
            "command": 'eac3to "{sec_file}" "{output_file}" -{delay_ms}ms',
            "note": "eac3to is recommended for E-AC3 tracks as it can apply delays without re-encoding.",
        },
    },
    "dts": {
        "Delay": {
            "tool": "eac3to",
            "action": "Apply a positive delay to the audio track (add silence).",
            "command": 'eac3to "{sec_file}" "{output_file}" +{delay_ms}ms',
            "note": "eac3to is recommended for DTS tracks as it can apply delays without re-encoding.",
        },
        "Cut": {
            "tool": "eac3to",
            "action": "Apply a negative delay to the audio track (cut from start).",
            "command": 'eac3to "{sec_file}" "{output_file}" -{delay_ms}ms',
            "note": "eac3to is recommended for DTS tracks as it can apply delays without re-encoding.",
        },
    },
    "truehd": {
        "Delay": {
            "tool": "eac3to",
            "action": "Apply a positive delay to the audio track (add silence).",
            "command": 'eac3to "{sec_file}" "{output_file}" +{delay_ms}ms',
            "note": "eac3to is recommended for TrueHD tracks as it can apply delays without re-encoding.",
        },
        "Cut": {
            "tool": "eac3to",
            "action": "Apply a negative delay to the audio track (cut from start).",
            "command": 'eac3to "{sec_file}" "{output_file}" -{delay_ms}ms',
            "note": "eac3to is recommended for TrueHD tracks as it can apply delays without re-encoding.",
        },
    },
    "aac": {
        "Delay": {
            "tool": "ffmpeg",
            "action": "Add silence to the beginning of the audio track.",
            "command": 'ffmpeg -i "{sec_file}" -af "adelay={delay_ms}|{delay_ms}" -c:a aac "{output_file}"',
            "note": "ffmpeg must re-encode to add silence to an AAC track. This may cause a slight quality loss.",
        },
        "Cut": {
            "tool": "ffmpeg",
            "action": "Cut from the beginning of the audio track.",
            "command": 'ffmpeg -ss {delay_s} -i "{sec_file}" -c:a copy "{output_file}"',
            "note": "ffmpeg can cut from an AAC track without re-encoding.",
        },
    },
    "default": {
        "Delay": {
            "tool": "ffmpeg",
            "action": "Add silence to the beginning of the audio track.",
            "command": 'ffmpeg -i "{sec_file}" -af "adelay={delay_ms}|{delay_ms}" "{output_file}"',
            "note": "For this codec, re-encoding is likely necessary. To avoid quality loss, consider decoding to WAV, "
            "applying the delay, and then re-encoding with a high-quality encoder.",
        },
        "Cut": {
            "tool": "ffmpeg",
            "action": "Cut from the beginning of the audio track.",
            "command": 'ffmpeg -ss {delay_s} -i "{sec_file}" -c:a copy "{output_file}"',
            "note": "This command attempts to cut without re-encoding. If it fails, re-encoding might be necessary.",
        },
        "Drifting delay": {
            "tool": "Audio Editor (e.g., Audacity, Adobe Audition)",
            "action": "Manually stretch or shrink audio parts to match the reference.",
            "command": None,
            "note": "Drifting delay requires advanced editing. No simple command can fix this automatically.",
        },
        "Different FPS": {
            "tool": "sox",
            "action": "Change audio speed to match the new framerate (preserves pitch).",
            "command": 'sox "{sec_file}" "{output_file}" speed {fps_ratio}',
            "note": "This adjusts the audio length to match the video length based on the FPS change. "
            "The fps_ratio should be calculated as new_fps/old_fps.",
        },
    },
}


def get_suggestion(codec: str, problem: str) -> Dict[str, Any]:
    """Get a suggestion from the database."""
    codec_suggestions = SUGGESTIONS_DB.get(codec, SUGGESTIONS_DB["default"])
    return codec_suggestions.get(problem, SUGGESTIONS_DB["default"].get(problem, {}))


def generate_suggestions(alignment_results: List[Dict[str, Any]]):
    """
    Generate and print suggestions based on alignment results.
    """
    suggestions_made = False
    for result in alignment_results:
        ref_file = result["reference_file"]
        sec_file = result["secondary_file"]

        ref_info = get_media_json(ref_file)
        sec_info = get_media_json(sec_file)

        problems = analyze_problems(result, ref_info, sec_info)

        if not problems:
            continue

        if not suggestions_made:
            click.echo(click.style("\n--- Audio Alignment Suggestions ---", fg="cyan", bold=True))
            suggestions_made = True

        sec_track = result["secondary_track"]

        sec_audio_streams = [s for s in sec_info.get("streams", []) if s.get("codec_type") == "audio"]
        codec = sec_audio_streams[sec_track].get("codec_name", "unknown") if sec_track < len(sec_audio_streams) else "unknown"

        click.echo(click.style(f"\nSuggestions for: {sec_file} (Track {sec_track})", fg="white", bold=True))
        click.echo(click.style(f"  Codec: {codec}", fg="green"))

        for problem, data in problems:
            suggestion = get_suggestion(codec, problem)
            if not suggestion:
                continue

            click.echo(click.style(f"  - Problem: {problem}", fg="yellow"))
            click.echo(click.style(f"    Action: {suggestion['action']}", fg="blue"))

            if "note" in suggestion:
                click.echo(click.style(f"    Note: {suggestion['note']}", dim=True))

            if cmd := suggestion.get("command"):
                delay_ms = result.get("mode_delay_ms", 0)
                
                output_file = f"output.{codec}"  # Placeholder
                fps_ratio = data if problem == "Different FPS" else "N/A"

                formatted_cmd = cmd.format(
                    sec_file=sec_file,
                    output_file=output_file,
                    delay_ms=abs(delay_ms),
                    delay_s=f"{abs(delay_ms) / 1000.0:.3f}",
                    ref_file=ref_file,
                    fps_ratio=fps_ratio,
                )
                click.echo(click.style(f"    Suggested command ({suggestion['tool']}):", fg="magenta"))
                click.echo(click.style(f"      {formatted_cmd}", fg="magenta", bold=True))


def analyze_problems(result: Dict[str, Any], ref_info: Dict, sec_info: Dict) -> List[Tuple[str, Any]]:
    """
    Analyze alignment results and identify problems.
    """
    problems = []
    delay_threshold = result.get("delay_threshold", 42)

    # Problem: Constant Delay
    is_delay_issue = (
        abs(result.get("mode_delay_ms", 0)) > delay_threshold
        or abs(result.get("average_delay_ms", 0)) > delay_threshold
    )
    if is_delay_issue:
        delay_ms = result.get("mode_delay_ms", 0)
        # If delay is negative, secondary is ahead, so we need to add delay (positive offset)
        # If delay is positive, secondary is behind, so we need to cut (negative offset)
        problem_type = "Delay" if delay_ms < 0 else "Cut"
        problems.append((problem_type, result.get("mode_delay_ms")))

    # Problem: Drifting Delay
    if len(result.get("delay_groups", [])) > 1:
        problems.append(("Drifting delay", None))

    # Problem: Different FPS
    ref_video_stream = next((s for s in ref_info.get("streams", []) if s.get("codec_type") == "video"), None)
    sec_video_stream = next((s for s in sec_info.get("streams", []) if s.get("codec_type") == "video"), None)

    if ref_video_stream and sec_video_stream:
        ref_fps_str = ref_video_stream.get("avg_frame_rate", "0/1")
        sec_fps_str = sec_video_stream.get("avg_frame_rate", "0/1")

        ref_num, ref_den = map(int, ref_fps_str.split('/'))
        sec_num, sec_den = map(int, sec_fps_str.split('/'))

        if ref_den > 0 and sec_den > 0 and ref_fps_str != sec_fps_str:
            ref_fps = ref_num / ref_den
            sec_fps = sec_num / sec_den
            problems.append(("Different FPS", f"{ref_fps / sec_fps:.6f}"))

    return problems 