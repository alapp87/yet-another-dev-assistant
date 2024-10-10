import sys
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from yada import utils, model
from yada.config import _config as yada_config, set_api_key
from yada.tool_loader import ToolLoader
from yada.agent import YadaAgent


class YadaCli:
    def __init__(self, thread_id: str, debug: bool = False) -> None:
        self._printed = set()
        self.config = {"configurable": {"thread_id": thread_id}}
        self.debug = debug

    def yada_command(self, command: str) -> None:
        agent = self._new_agent()

        result = agent.invoke(
            {"messages": [command]},
            config=self.config,
        )
        self._print_event(result, self._printed, agent.is_sensitive_tool_call_exist)
        self._handle_tool_calls(result, agent, self.config)

    def yada_chat(self) -> None:
        self._print_title()
        self._check_api_key()

        agent = self._new_agent()

        utils.agent_response("Hello! How can I help you?")

        while True:
            try:
                user_prompt = utils.user_input()
                if not user_prompt:
                    continue
                elif user_prompt.lower() in ["q", "exit", "quit"]:
                    self._say_goodbye()
                    break

                events = agent.stream(
                    {"messages": [user_prompt]},
                    config=self.config,
                )

                for event in events:
                    self._print_event(
                        event, self._printed, agent.is_sensitive_tool_call_exist
                    )

                self._handle_tool_calls(event, agent, self.config)
            except KeyboardInterrupt:
                self._say_goodbye()
                break

    def _print_title(self) -> None:
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

    def _check_api_key(self) -> None:
        if not yada_config.api_key:
            utils.agent_response(
                "Looks like you haven't set up your OpenAI API key. Please provide it to me and I'll update the configuration."
            )
            while True:
                api_key = utils.user_input("API KEY (q to quit): ")
                if not api_key:
                    continue
                elif api_key.strip().lower() in ["q", "exit", "quit"]:
                    self._say_goodbye()
                    sys.exit(0)
                else:
                    break

            set_api_key(api_key)

    def _new_agent(self) -> YadaAgent:
        tool_loader = ToolLoader()
        tool_loader.load()

        return YadaAgent(
            model=model(),
            safe_tools=tool_loader.safe_tools,
            sensitive_tools=tool_loader.sensitive_tools,
            checkpointer=MemorySaver(),
            debug=self.debug,
        )

    def _print_event(
        self,
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
                                tool_call_msg += (
                                    f"- **Tool:** {tc['name']}\n\t- **Args**\n"
                                )

                                for arg in tc["args"]:
                                    tool_call_msg += f"\t\t- {arg}={tc['args'][arg]}\n"

                            utils.agent_response(tool_call_msg)
                    else:
                        utils.agent_response(message.content)

                printed_events.add(message_id)

    def _handle_tool_calls(self, event: dict, agent: YadaAgent, config: dict) -> None:
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
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"Tool call denied by user. Reasoning: '{user_prompt}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )

            self._print_event(result, self._printed)

            snapshot = agent.get_state(config)

    def _say_goodbye(self) -> None:
        utils.agent_response("Goodbye!")
