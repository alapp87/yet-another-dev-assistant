import sys
import click
from uuid import uuid4
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from yada import utils, model
from yada.config import _config as yada_config, set_api_key
from yada.tool_loader import ToolLoader
from yada.agent import YadaAgent


@click.command()
@click.option(
    "-t", "--thread-id", "thread_id", default=str(uuid4()), help="Agent graph thread ID"
)
@click.option("-D", "--debug", "debug", is_flag=True, help="Debug mode")
def run(thread_id: str, debug: bool):
    _printed = set()
    config = {"configurable": {"thread_id": thread_id}}

    _print_title()
    _check_api_key()

    agent = _new_agent(debug=debug)

    utils.agent_response("Hello! How can I help you?")

    while True:
        try:
            user_prompt = utils.user_input()
            if not user_prompt:
                continue
            elif user_prompt.lower() in ["q", "exit", "quit"]:
                utils.agent_response("Goodbye!")
                break

            events = agent.stream(
                {"messages": [user_prompt]},
                config=config,
            )

            for event in events:
                _print_event(event, _printed, agent.is_sensitive_tool_call_exist)

            snapshot = agent.get_state(config)
            while snapshot.next:
                while True:
                    user_prompt = utils.user_input("YOU (y/N): ")
                    if not user_prompt:
                        continue
                    else:
                        break

                if user_prompt.strip().lower() == "y":
                    result = agent.invoke(None, config)
                else:
                    if user_prompt.strip().lower() == "n":
                        user_prompt = "No, I don't want to execute those tools."

                    result = agent.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=event["messages"][-1].tool_calls[0][
                                        "id"
                                    ],
                                    content=f"Tool call denied by user. Reasoning: '{user_prompt}'. Continue assisting, accounting for the user's input.",
                                )
                            ]
                        },
                        config,
                    )

                _print_event(result, _printed)

                snapshot = agent.get_state(config)
        except KeyboardInterrupt:
            utils.agent_response("Goodbye!")
            break


def _print_title() -> None:
    utils.print_text(
        """
 __  _____   ___  ___ 
 \\ \\/ / _ | / _ \\/ _ |
  \\  / __ |/ // / __ |
  /_/_/ |_/____/_/ |_|
Yet Another Dev Assistant
    """,
        style="blue",
    )
    utils.print_markdown(
        """
**Examples of what to ask me**
- What can you do?
- Create the dir "foo"
- Install Homebrew
- Install the python@3.12 package using Homebrew 
- Clone the Github repo "Git URL" to path "foo"
        """.strip()
    )
    print("")  # newline


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
                utils.agent_response("Goodbye!")
                sys.exit(0)
            else:
                break

        set_api_key(api_key)


def _new_agent(debug: bool = False) -> YadaAgent:
    tool_loader = ToolLoader()
    tool_loader.load()

    return YadaAgent(
        model=model(),
        safe_tools=tool_loader.safe_tools,
        sensitive_tools=tool_loader.sensitive_tools,
        checkpointer=MemorySaver(),
        debug=debug,
    )


def _print_event(
    event: dict,
    printed_events: set,
    sensitive_tool_check_func: Callable | None = None,
    print_user_events: bool = False,
) -> None:
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in:", current_state[-1])

    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]

        message_id = message.id
        if message_id not in printed_events:
            if isinstance(message, HumanMessage) and print_user_events:
                utils.user_response(message.content)
            elif isinstance(message, AIMessage):
                tool_calls = message.tool_calls
                if tool_calls:
                    if sensitive_tool_check_func and sensitive_tool_check_func(
                        tool_calls
                    ):
                        tool_call_msg = (
                            "I want to execute the following tools. Reply 'y' to continue or 'n' to cancel. Otherwise you can explain your requested changes."
                            "\n\n**Calling tool(s)**\n"
                        )

                        for tc in tool_calls:
                            tool_call_msg += f"- **Tool:** {tc['name']}\n\t- **Args**\n"

                            for arg in tc['args']:
                                tool_call_msg += f"\t\t- {arg}={tc['args'][arg]}\n"

                        utils.agent_response(tool_call_msg)
                else:
                    utils.agent_response(message.content)

            printed_events.add(message_id)


if __name__ == "__main__":
    run()
