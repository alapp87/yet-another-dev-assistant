import os
import shutil
import pathlib

from langchain.tools import tool

from yada.tools import safe_tool, sensitive_tool, json2str


@safe_tool
@tool
def list_directory(directory: str = ".") -> list[str]:
    """
    List the contents of a directory.

    Args:
        directory (str): The directory to list, default ".".
    """
    return json2str([str(p) for p in pathlib.Path(directory).iterdir()])


@safe_tool
@tool
def create_directory(directory: str) -> str:
    """
    Create a directory.

    Args:
        directory (str): The directory to create.
    """
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return f"Created directory {directory}."


@sensitive_tool
@tool
def delete_directory(directory: str) -> str:
    """
    Delete a directory.

    Args:
        directory (str): The directory to delete.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        return f"Deleted directory and its contents: {directory}"
    else:
        return f"Directory does not exist: {directory}"
