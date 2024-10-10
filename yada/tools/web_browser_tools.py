import webbrowser

from langchain.tools import tool

from yada.tools import safe_tool, sensitive_tool


@safe_tool
@tool
def open_url_in_browser(url: str) -> str:
    """
    Open a URL in the default web browser.

    Args:
        url (str): The URL to open.
    """
    webbrowser.open(url)
    return f"Opened {url} in the default web browser."
