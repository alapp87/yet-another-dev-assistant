# Yet Another Dev Assistant (YADA)

Yet Another Dev Assistant is a command-line AI agent to help augment developers by performing actions through natural language conversation. YADA has many builtin capabilities and can easily be extended with additional capabilities specific to a developer's needs.

![YADA Demo](https://github.com/alapp87/yet-another-dev-assistant/blob/main/examples/demo01.png)

## Getting Started

### Limitations

1. Works only for Linux/Mac systems and has not been tested with Windows systems.
2. OpenAI models only at the moment.


### Prerequisites

1. An OpenAI API key

### Configuration

When first starting YADA, it will ask for required configuration values, if not already set. You can provide them then and the config will automatically created. The config file can be found at `~/.config/yada/yada.config` and you can set any additional configuration values you want.

| Name             | Description                       | Required | Default | Example       |
|------------------|-----------------------------------|----------|---------|---------------|
| api_key          | OpenAI API key                    | Y        |         |               |
| llm_model_name   | OpenAI model name                 | N        | gpt-4o  |               |
| custom_tools_dir | Directory containing custom tools | N        |         | /custom/tools |


### Installation

The easiest way to install YADA is using pipx (https://pipx.pypa.io/stable/installation/).

```bash
pipx install "git+https://github.com/alapp87/yet-another-dev-assistant.git"
```

### Usage

Once installed, the `yada` command will be available in your environment. Run the command to start the assistant.

```bash
# start chat
yada
```

```bash
# one off command
yada create the dir "test"
```

```bash
yada --help
Usage: yada [OPTIONS] [COMMAND]...

Options:
  -V, --version         Show version
  --config              Configure YADA
  -t, --thread-id TEXT  Agent graph thread ID
  -D, --debug           Debug mode
  --help                Show this message and exit.
```

## Add Custom Tools

YADA allows developers to add their own tools. Create a python file(s) and write tool functions in them. The file names must end in `_tools.py`.

Set the configuration `custom_tools_dir` value to the directory containing the python tool files and when YADA starts up, it will automatically load them into it's capabilities.

Dependencies used in custom tools must be installed globally or alrady a part of the YADA dependencies.

### Example Tool File

`custom/tools/my_custom_tools.py`

```python
from langchain.tools import tool
from yada.tools import safe_tool, sensitive_tool

@safe_tool
@tool
def tell_a_joke() -> str:
    """
    Tell a joke.
    """
    return "Why don't we ever finish talking to you?\n\nBecause it's always \"YADA, YADA, YADA...\""

@safe_tool
@tool
def say_hello(name: str) -> str:
    """
    Say hello.

    Args:
        name (str): Name to say hello to.
    """
    return f"Hello {name}!"

@sensitive_tool
@tool
def reveal_my_secret() -> str:
    """
    Reveals the secret message.
    """
    return "Spaghetti is my favorite food."

```

Set `custom_tools_dir` to `custom/tools` and YADA will load the tools into it's capabilities.
