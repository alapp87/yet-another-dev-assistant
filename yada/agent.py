from typing import Any, Annotated, Literal, Sequence, TypedDict, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

from langchain_openai import ChatOpenAI

from yada.sync_tool_node import SyncToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep


class YadaAgent:
    STATE_MODIFIER_RUNNABLE_NAME = "StateModifier"

    def __init__(
        self,
        model: ChatOpenAI,
        safe_tools: list[BaseTool],
        sensitive_tools: list[BaseTool],
        interrupt_before: list[str] = ["sensitive_tools"],
        checkpointer=None,
        debug: bool = False,
    ) -> None:
        tool_classes = safe_tools + sensitive_tools
        self.sensitive_tool_names = [tool.name for tool in sensitive_tools]
        safe_tool_node = SyncToolNode(safe_tools, all_tools=tool_classes)
        sensitive_tool_node = SyncToolNode(sensitive_tools, all_tools=tool_classes)
        model = model.bind_tools(tool_classes)

        state_modifier_runnable = RunnableLambda(
            lambda state: [
                SystemMessage(
                    """
                    You're name is YADA. You are a helpful AI assistant for developers.
                    If you use a tool, provide useful information back to the user to 
                    help them understand what the tool did.
                    If asked what capabilities you have, ensure you list or describe 
                    ALL tools you have access to.
                    """.strip()
                )
            ]
            + state["messages"],
            name=self.STATE_MODIFIER_RUNNABLE_NAME,
        )

        self.model_runnable = state_modifier_runnable | model

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", RunnableLambda(self._call_model))
        workflow.add_node("safe_tools", safe_tool_node)
        workflow.add_node("sensitive_tools", sensitive_tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "agent": "agent",
                "safe": "safe_tools",
                "sensitive": "sensitive_tools",
                "end": END,
            },
        )

        workflow.add_edge("safe_tools", "agent")
        workflow.add_edge("sensitive_tools", "agent")

        self.workflow = workflow.compile(
            checkpointer=checkpointer, interrupt_before=interrupt_before, debug=debug
        )

    def _call_model(self, state: AgentState, config: RunnableConfig):
        response = self.model_runnable.invoke(state, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }

        return {"messages": [response]}

    def _should_continue(
        self, state: AgentState
    ) -> Literal["agent", "safe", "sensitive", "end"]:
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, HumanMessage):
            return "agent"
        elif not last_message.tool_calls:
            return "end"
        elif self.is_sensitive_tool_call_exist(last_message.tool_calls):
            return "sensitive"
        else:
            return "safe"

    def invoke(
        self, input: dict[str, Any], config: RunnableConfig
    ) -> Iterator[dict[str, Any]]:
        return self.workflow.invoke(input, config)

    def stream(
        self, input: dict[str, Any], config: RunnableConfig
    ) -> Iterator[dict[str, Any]]:
        return self.workflow.stream(input, config, stream_mode="values")

    def get_state(self, config: RunnableConfig):
        return self.workflow.get_state(config)

    def is_sensitive_tool_call_exist(self, tool_calls: list[BaseTool]) -> bool:
        for tool_call in tool_calls:
            if self.is_sensitive_tool(tool_call["name"]):
                return True
        return False

    def is_sensitive_tool(self, tool_name: str) -> bool:
        return tool_name in self.sensitive_tool_names


if __name__ == "__main__":
    from yada import model

    yada = YadaAgent(model=model())
    g = yada.workflow.get_graph()
    print(g.draw_ascii())
    print(g.draw_mermaid())
