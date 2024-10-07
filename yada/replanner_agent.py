import json
import operator
from typing import Annotated, List, Tuple, Any, Union, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END

from yada.planner_agent import Plan, PlanExecute


class Response(BaseModel):
    response: str


class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to the user, use Response. If you need to further use tools to get the answer, use Plan."
    )


class ReplannerAgent:
    def __init__(self, model: ChatOpenAI):
        replanner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
        )

        self.agent = replanner_prompt | model.with_structured_output(Act)

    def invoke(self, state: PlanExecute) -> Union[Response, Plan]:
        return self.agent.invoke(state)
