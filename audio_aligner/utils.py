import click


def validate_positive_integer(ctx: click.Context, param: dict, value: int) -> int:
    if value < 1:
        raise click.BadParameter('Should be a positive integer.')
    return value


def validate_non_negative_integer(ctx: click.Context, param: dict, value: int) -> int:
    if value < 0:
        raise click.BadParameter('Should be a non-negative integer.')
    return value 