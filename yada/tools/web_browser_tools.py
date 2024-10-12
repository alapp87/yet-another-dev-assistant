import webbrowser

from langchain.tools import tool

from yada.tools import safe_tool, sensitive_tool


@safe_tool
@tool
def open_url_in_browser(
    url: str, new_window: bool = False, new_tab: bool = False
) -> str:
    """
    Open a URL in the default web browser.

    Args:
        url (str): The URL to open.
        new_window (bool): Optional, Open the URL in a new window, default False.
        new_tab (bool): Optional, Open the URL in a new tab, default False.
    """
    if new_window:
        webbrowser.open_new(url)
    elif new_tab:
        webbrowser.open_new_tab(url)
    else:
        webbrowser.open(url)
    return f"Opened {url} in the default web browser."
