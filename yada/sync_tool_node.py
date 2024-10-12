from typing import (
    Any,
    Union,
    cast,
)

from langchain_core.messages import (
    AnyMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import (
    get_config_list,
)
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool

from langgraph.store.base import BaseStore

from pydantic import BaseModel

from langgraph.prebuilt.tool_node import ToolNode, _get_state_args, _get_store_arg


class SyncToolNode(ToolNode):
    """
    SyncToolNode is a ToolNode that runs tools synchronously.
    """

    def __init__(self, tools: list, all_tools: list) -> None:
        self.all_tools = all_tools
        super().__init__(tools)

        # add missing tools
        for tool_ in all_tools:
            if not isinstance(tool_, BaseTool):
                tool_ = cast(BaseTool, create_tool(tool_))
            if tool_.name not in self.tools_by_name:
                self.tools_by_name[tool_.name] = tool_
                self.tool_to_state_args[tool_.name] = _get_state_args(tool_)
                self.tool_to_store_arg[tool_.name] = _get_store_arg(tool_)

    def _func(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> Any:
        tool_calls, output_type = self._parse_input(input, store)
        config_list = get_config_list(config, len(tool_calls))
        # with get_executor_for_config(config) as executor:
        #     outputs = [*executor.map(self._run_one, tool_calls, config_list)]
        outputs = []
        for tool_call, tool_config in zip(tool_calls, config_list):
            outputs.append(self._run_one(tool_call, tool_config))
        # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
        return outputs if output_type == "list" else {"messages": outputs}
