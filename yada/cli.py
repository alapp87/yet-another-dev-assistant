import sys
import click
from uuid import uuid4

from yada.yada_cli import YadaCli


@click.command()
@click.option("-V", "--version", "version", is_flag=True, help="Show version")
@click.option(
    "-t", "--thread-id", "thread_id", default=str(uuid4()), help="Agent graph thread ID"
)
@click.option("-D", "--debug", "debug", is_flag=True, help="Debug mode")
@click.argument("command", nargs=-1, required=False)
def run(version: str, thread_id: str, debug: bool, command: tuple[str]):
    if version:
        _print_version()
        sys.exit(0)

    yada_cli = YadaCli(thread_id=thread_id, debug=debug)

    if command:
        command = " ".join(command)
        yada_cli.yada_command(command)
    else:
        yada_cli.yada_chat()


def _print_version():
    from importlib.metadata import version, PackageNotFoundError

    try:
        print(version("yada"))
    except PackageNotFoundError:
        print("0.0.0")  # Default version if package is not installed


if __name__ == "__main__":
    run()
