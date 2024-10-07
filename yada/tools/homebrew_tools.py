import subprocess

from langchain.tools import tool

from yada.tools import safe_tool, sensitive_tool


@sensitive_tool
@tool
def install_homebrew():
    """
    Install Homebrew package manager.
    """
    try:
        subprocess.run(
            [
                "/bin/bash",
                "-c",
                "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)",
            ],
            check=True,
        )
        return "Homebrew installation complete."
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"


@safe_tool
@tool
def list_homebrew_packages():
    """
    List the installed Homebrew packages.
    """
    try:
        result = subprocess.run(
            ["brew", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"


@safe_tool
@tool
def install_homebrew_package(package: str):
    """
    Install a Homebrew package.

    Args:
        package (str): The package to install.
    """
    try:
        subprocess.run(
            ["brew", "install", package],
            check=True,
        )
        return f"Installed Homebrew package: {package}"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"


@sensitive_tool
@tool
def uninstall_homebrew_package(package: str):
    """
    Uninstall a Homebrew package.

    Args:
        package (str): The package to uninstall.
    """
    try:
        subprocess.run(
            ["brew", "uninstall", package],
            check=True,
        )
        return f"Uninstalled Homebrew package: {package}"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"
