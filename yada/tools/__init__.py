import json

from langchain.tools import tool

_tool_registry = {}


def get_tool_registry() -> dict:
    return _tool_registry


def safe_tool(*args, **kwargs):
    structured_tool = args[0]
    get_tool_registry()[structured_tool.name] = True
    return structured_tool


def sensitive_tool(*args, **kwargs):
    structured_tool = args[0]
    get_tool_registry()[structured_tool.name] = False
    return structured_tool


def json2str(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda obj: str(obj))


from yada.tools import (
    docker_tools,
    filesystem_tools,
    github_tools,
    homebrew_tools,
    os_tools,
    web_browser_tools,
)


@safe_tool
@tool
def list_capabilities() -> str:
    """
    List the capabilities of the tools available in YADA.
    """
    return json2str(get_tool_registry().keys())
