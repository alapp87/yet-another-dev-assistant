import platform
import subprocess

from langchain.tools import tool
from yada.tools import safe_tool, sensitive_tool


@safe_tool
@tool
def get_system_operating_system() -> str:
    """
    Get the operating system of the system.
    """
    return platform.system()


@safe_tool
@tool
def get_system_chip_architecture() -> str:
    """
    Get the chip architecture (i.e. Mac, Linux, Windows and amd64 arm64) of the system.
    """
    return platform.machine()


@sensitive_tool
@tool
def execute_shell_command(command: str) -> str:
    """
    Execute a shell command.

    Args:
        command (str): The command to execute.
    """
    command_output = subprocess.run(
        command, shell=True, capture_output=True
    ).stdout.decode("utf-8")

    return f"""
    Command Output
    ```
    {command_output}
    ```
    """
