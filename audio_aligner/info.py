import click


def print_file_info(video_path: str) -> None:
    import av

    with av.open(video_path, 'r') as container:
        click.echo(f'File: {video_path}')
        click.echo(f'Duration: {container.duration / 1000000}s')
        click.echo('Streams:')
        for i, stream in enumerate(container.streams):
            click.echo(f'  - Stream {i}:')
            click.echo(f'    Type: {stream.type}')
            if stream.type == 'video':
                click.echo(f'    Codec: {stream.codec_context.name}')
                click.echo(f'    Resolution: {stream.width}x{stream.height}')
                click.echo(f'    FPS: {stream.average_rate}')
            elif stream.type == 'audio':
                click.echo(f'    Codec: {stream.codec_context.name}')
                click.echo(f'    Layout: {stream.layout.name}')
                click.echo(f'    Sample Rate: {stream.sample_rate}')
                if stream.metadata.get('language'):
                    click.echo(f'    Language: {stream.metadata.get("language")}')
                if stream.metadata.get('title'):
                    click.echo(f'    Title: {stream.metadata.get("title")}') 