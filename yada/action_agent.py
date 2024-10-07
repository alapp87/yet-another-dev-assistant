import json
import operator
from typing import Annotated, List, Tuple, Any, Union, Literal
from typing_extensions import TypedDict

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END

from yada.assistant import Assistant
from yada.tool_loader import ToolLoader


class ActionAgent(Assistant):
    def __init__(self, model: ChatOpenAI, tool_loader: ToolLoader):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You're name is YADA. You are a helpful AI assistant for developers.
                    If you use a tool, provide useful information back to the user to 
                    help them understand what the tool did.
                    If asked what capabilities you have, ensure you list or describe 
                    ALL tools you have access to.
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        runnable = prompt | model.bind_tools(
            tool_loader.safe_tools + tool_loader.sensitive_tools
        )
        super().__init__(runnable)


class PlannerAgent:
    def __init__(self, model: ChatOpenAI, tools: list[BaseTool]) -> None:
        description_of_tools = json.dumps([repr(tool) for tool in tools], indent=2)

        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You're name is YADA. You are a helpful AI assistant for developers.
                    If you use a tool, provide useful information back to the user to 
                    help them understand what the tool did.
                    If asked what capabilities you have, ensure you list or describe 
                    ALL tools you have access to.
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        self.planner = planner_prompt | model.with_structured_output(Plan)

    def invoke(self, inputs: dict) -> Plan:
        return self.planner.invoke(inputs)


if __name__ == "__main__":
    from yada import model
    from yada.tool_loader import ToolLoader

    tl = ToolLoader()
    tl.load()

    planner = PlannerAgent(model(), tools=tl.safe_tools + tl.sensitive_tools)
    while True:
        user_prompt = input("YOU: ")
        if not user_prompt:
            continue
        elif user_prompt.lower() in ["q", "exit", "quit"]:
            print("Goodbye!")
            break

        response = planner.invoke({"messages": [("user", user_prompt)]})
        print(response.steps)
