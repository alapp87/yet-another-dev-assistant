import os
import importlib
import importlib.util
import inspect
import pathlib
from types import ModuleType

from langchain.tools import BaseTool

from yada import tools, custom_tools_dir


class ToolLoader:
    def __init__(self):
        self.safe_tools = []
        self.sensitive_tools = []

    def load(self) -> tuple[list[BaseTool], list[BaseTool]]:
        safe_tools, sensitive_tools = self._categorize_tools(
            tools, tools.get_tool_registry()
        )
        custom_safe_tools, custom_sensitive_tools = self._load_custom_tools(
            os.environ.get("YADA_CUSTOM_TOOLS_DIR", custom_tools_dir())
        )

        self.safe_tools.extend(safe_tools + custom_safe_tools)
        self.sensitive_tools.extend(sensitive_tools + custom_sensitive_tools)

    def _categorize_tools(
        self, module: object, registry: dict
    ) -> tuple[list[BaseTool], list[BaseTool]]:
        safe_tools = []
        sensitive_tools = []

        for name, obj in inspect.getmembers(module):
            if isinstance(obj, ModuleType) and obj.__name__.endswith("tools"):
                safe, sensitive = self._categorize_tools(obj, registry)
                safe_tools.extend(safe)
                sensitive_tools.extend(sensitive)
            elif isinstance(obj, BaseTool):
                if registry.get(obj.name):
                    safe_tools.append(obj)
                else:
                    sensitive_tools.append(obj)

        return safe_tools, sensitive_tools

    def _load_custom_tools(
        self, directory: str
    ) -> tuple[list[BaseTool], list[BaseTool]]:
        safe_tools = []
        sensitive_tools = []

        if directory:

            def load_user_functions(file_path):
                spec_name = pathlib.Path(file_path).stem
                spec = importlib.util.spec_from_file_location(spec_name, file_path)
                custom_tools = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_tools)
                return custom_tools

            tool_files = [
                f
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and f.endswith("_tools.py")
            ]

            for tool_file in tool_files:
                custom_tools_module = load_user_functions(
                    os.path.join(directory, tool_file)
                )
                safe, sensitive = self._categorize_tools(
                    custom_tools_module, tools.get_tool_registry()
                )
                safe_tools.extend(safe)
                sensitive_tools.extend(sensitive)

        return safe_tools, sensitive_tools
