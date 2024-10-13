import sys
import click
from uuid import uuid4

from yada import utils
from yada.config import (
    _config as yada_config,
    config_selections,
    set_api_key,
)
from yada.yada_cli import YadaCli


@click.command()
@click.option("-V", "--version", "version", is_flag=True, help="Show version")
@click.option("--config", is_flag=True, help="Configure YADA")
@click.option(
    "-t", "--thread-id", "thread_id", default=str(uuid4()), help="Agent graph thread ID"
)
@click.option("-D", "--debug", "debug", is_flag=True, help="Debug mode")
@click.argument("command", nargs=-1, required=False)
def run(version: bool, config: bool, thread_id: str, debug: bool, command: tuple[str]):
    if version:
        _print_version()
        sys.exit(0)
    elif config:
        _configure_yada()
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


def _configure_yada() -> None:
    try:
        while True:
            _print_config_selection(config_selections)
            selection = utils.user_input("Your selection (q to quit): ")
            if not selection:
                continue
            elif utils.is_exit_response(selection):
                break

            selection_int = int(selection) - 1
            if selection_int not in range(len(config_selections)):
                utils.print_markdown(
                    "Invalid selection. Please try again.", style="red"
                )
                continue

            value = utils.user_input("New value: ")

            name = config_selections[selection_int]["name"]
            update_func = config_selections[selection_int]["update_func"]
            update_func(value)
            utils.print_markdown(
                f"{name} updated successfully.",
                style="bold blue",
            )
    except KeyboardInterrupt:
        pass


def _print_config_selection(config_selections: list[dict]) -> None:
    message = "Please select the configuration value you would like to update:\n"
    for idx, s in enumerate(config_selections):
        message += f"{idx+1}. {s['name']}\n"

    utils.print_markdown(
        message,
        prepend_text="\n",
        style="blue",
    )


def _check_api_key() -> None:
    if not yada_config.api_key:
        utils.agent_response(
            "Looks like you haven't set your OpenAI API key. Please provide it to me and I'll update the configuration."
        )
        while True:
            api_key = utils.user_input("API KEY (q to quit): ")
            if not api_key:
                continue
            elif utils.is_exit_response(api_key):
                utils.say_goodbye()
                sys.exit(0)
            else:
                break

        set_api_key(api_key)


if __name__ == "__main__":
    run()
