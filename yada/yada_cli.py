from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from yada import utils, model
from yada.tool_loader import ToolLoader
from yada.agent import YadaAgent


class YadaCli:
    def __init__(self, thread_id: str, debug: bool = False) -> None:
        self._printed = set()
        self.config = {"configurable": {"thread_id": thread_id}}
        self.debug = debug
        self.agent = self._new_agent()

    def yada_command(self, command: str) -> None:
        result = self.agent.invoke(
            {"messages": [command]},
            config=self.config,
        )
        self._handle_event(result)
        self._handle_tool_calls(result)

    def yada_chat(self) -> None:
        self._print_title()

        utils.agent_response("Hello! How can I help you?")

        while True:
            try:
                user_prompt = utils.user_input()
                if not user_prompt:
                    continue
                elif utils.is_exit_response(user_prompt):
                    utils.say_goodbye()
                    break

                utils.print_thinking()

                events = self.agent.stream(
                    {"messages": [user_prompt]},
                    config=self.config,
                )

                for event in events:
                    self._handle_event(event)

                self._handle_tool_calls(event)
            except KeyboardInterrupt:
                utils.say_goodbye()
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
**Examples of what you can ask me**
- What can you do?
- What are all the inputs to run a Docker container?
- Create the dir "foo"
- Install Homebrew
- Install python@3.12 using Homebrew
- Clone "Git URL" to path "bar"
""".strip()
        )
        print("")  # newline

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

    def _handle_event(
        self,
        event: dict,
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
            if message_id not in self._printed:
                if print_user_events and isinstance(message, HumanMessage):
                    utils.user_response(message.content)
                elif isinstance(message, AIMessage):
                    self._handle_ai_message(message)

                self._printed.add(message_id)

    def _handle_tool_calls(self, event: dict) -> None:
        snapshot = self.agent.get_state(self.config)
        while snapshot.next:
            while True:
                user_prompt = utils.user_input("YOU (y/N): ")
                if not user_prompt:
                    continue
                else:
                    break

            result = self._handle_user_response_to_sensitive_tool_call(
                user_prompt, event
            )

            last_message = result.get("messages")[-1]
            self._handle_ai_message(last_message)

            snapshot = self.agent.get_state(self.config)

    def _handle_user_response_to_sensitive_tool_call(
        self, user_prompt: str, event: dict
    ) -> dict:
        if user_prompt.strip().lower() == "y":
            utils.print_working()
            return self.agent.invoke(None, self.config)
        else:
            if user_prompt.strip().lower() == "n":
                user_prompt = "No, I don't want to execute those tools."

            utils.print_thinking()

            last_event_message: AIMessage = event["messages"][-1]
            tool_calls = last_event_message.tool_calls

            return self.agent.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_calls[0]["id"],
                            content=(
                                f"Tool call denied by user. Reasoning: '{user_prompt}'. "
                                "Continue assisting, accounting for the user's input."
                            ),
                        )
                    ]
                },
                self.config,
            )

    def _handle_ai_message(
        self,
        message: AIMessage,
    ) -> None:
        tool_calls = message.tool_calls
        if tool_calls:
            if self.agent.is_sensitive_tool_call_exist(tool_calls):
                self._print_tool_calls_message(tool_calls)
            else:
                # safe tool call
                utils.print_working()
        else:
            utils.agent_response(message.content)

    def _print_tool_calls_message(self, tool_calls: list[dict]) -> None:
        tool_call_msg = (
            "I want to execute the following tools. "
            "Reply 'y' to continue or 'n' to cancel. "
            "Otherwise you can explain your requested changes."
            "\n\n**Calling tool(s)**\n"
        )

        for tc in tool_calls:
            tool_call_msg += f"- **Tool:** {tc['name']}\n\t- **Args**\n"

            for arg in tc["args"]:
                tool_call_msg += f"\t\t- {arg}={tc['args'][arg]}\n"

        utils.agent_response(tool_call_msg)
