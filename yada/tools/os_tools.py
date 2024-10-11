import platform

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
