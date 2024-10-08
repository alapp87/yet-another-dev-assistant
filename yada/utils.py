from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt.tool_node import ToolNode

USER_TEXT = Text("YOU: ", style="bold green")
AGENT_TEXT = Text("YADA: ", style="bold blue")


def print_text(text: str, style: str = None) -> None:
    console = Console()
    console.print(text, style=style)


def print_markdown(markdown: str, prepend_text: Text = None, style: str = None) -> None:
    console = Console()
    if prepend_text:
        console.print(prepend_text, end="")
    console.print(Markdown(markdown.strip()), style=style)


def agent_response(text: str) -> None:
    print_markdown(text, prepend_text=AGENT_TEXT)


def user_response(text: str) -> None:
    print_markdown(text, prepend_text=USER_TEXT)


def user_input(message: str = "YOU: ") -> str:
    console = Console()
    return console.input(Text(message, style="bold green"))


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
