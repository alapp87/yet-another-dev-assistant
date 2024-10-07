from typing import Literal

from langchain_core.runnables import RunnableConfig, RunnableLambda

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from yada.tool_loader import ToolLoader
from yada.action_agent import ActionAgent
from yada.planner_agent import PlannerAgent, PlanExecute
from yada.replanner_agent import ReplannerAgent, Response


class YadaWorkflow:
    def __init__(
        self,
        config: RunnableConfig,
        action_agent: ActionAgent,
        planner_agent: PlannerAgent,
        replanner_agent: ReplannerAgent,
        tool_loader: ToolLoader,
        debug: bool = False,
    ) -> None:
        self.action_agent = action_agent
        self.planner_agent = planner_agent
        self.replanner_agent = replanner_agent
        self.config = config

        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("planner", self.plan_step)

        # Add the execution step
        workflow.add_node("agent", self.execute_step)

        # Add a replan node
        workflow.add_node("replan", self.replan_step)

        workflow.add_edge(START, "planner")

        # From plan we go to agent
        workflow.add_edge("planner", "agent")

        # From agent, we replan
        workflow.add_edge("agent", "replan")

        workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            self.should_end,
            ["agent", END],
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.app = workflow.compile(checkpointer=MemorySaver(), debug=debug)

    def execute_step(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        agent_response = self.action_agent.invoke(
            {"messages": [("user", task_formatted)]}, self.config
        )
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    def plan_step(self, state: PlanExecute):
        plan = self.planner_agent.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    def replan_step(self, state: PlanExecute):
        output = self.replanner_agent.invoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    def should_end(self, state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "agent"

    def invoke(self, inputs: dict, config: RunnableConfig) -> PlanExecute:
        return self.app.invoke(inputs, config)

    def stream(self, inputs: dict, config: RunnableConfig):
        return self.app.stream(inputs, config, stream_mode="values")


if __name__ == "__main__":
    from uuid import uuid4
    from yada import model
    from yada.tool_loader import ToolLoader

    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 20}
    tl = ToolLoader()
    tl.load()
    agent = YadaAgent(model(), tl.safe_tools, tl.sensitive_tools)
    planner = PlannerAgent(model(), tl.safe_tools + tl.sensitive_tools)
    replanner = ReplannerAgent(model())

    w = YadaWorkflow(
        config=config,
        action_agent=agent,
        planner_agent=planner,
        replanner_agent=replanner,
        tool_loader=tl,
    )
    g = w.app.get_graph(config)
    print(g.draw_ascii())
    print(g.draw_mermaid(with_styles=False))
