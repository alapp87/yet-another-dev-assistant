import sys
import click
from uuid import uuid4

from yada import utils
from yada.config import _config as yada_config, set_api_key
from yada.yada_cli import YadaCli


@click.command()
@click.option("-V", "--version", "version", is_flag=True, help="Show version")
@click.option(
    "-t", "--thread-id", "thread_id", default=str(uuid4()), help="Agent graph thread ID"
)
@click.option("-D", "--debug", "debug", is_flag=True, help="Debug mode")
@click.argument("command", nargs=-1, required=False)
def run(version: bool, thread_id: str, debug: bool, command: tuple[str]):
    if version:
        _print_version()
        sys.exit(0)

    _check_api_key()

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


def _check_api_key() -> None:
    if not yada_config.api_key:
        utils.agent_response(
            "Looks like you haven't set up your OpenAI API key. Please provide it to me and I'll update the configuration."
        )
        while True:
            api_key = utils.user_input("API KEY (q to quit): ")
            if not api_key:
                continue
            elif api_key.strip().lower() in ["q", "exit", "quit"]:
                utils.say_goodbye()
                sys.exit(0)
            else:
                break

        set_api_key(api_key)


if __name__ == "__main__":
    run()
