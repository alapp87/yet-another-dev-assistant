import json
import operator
from typing import Annotated, List, Tuple, Any, Union, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    steps: List[str] = Field(
        description="Different steps to follow, should be in sorted order."
    )


class PlannerAgent:
    def __init__(self, model: ChatOpenAI, tools: list[BaseTool]) -> None:
        description_of_tools = json.dumps([repr(tool) for tool in tools], indent=2)

        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

You have the following tools available to you:
{description_of_tools}""",
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
