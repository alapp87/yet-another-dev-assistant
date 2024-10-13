import configparser
import pathlib
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

YADA_CONFIG_FILE_PATH = pathlib.Path.home() / ".config/yada/yada.config"
_SECTION_NAME = "default"


def get_or_create_yada_config_file() -> str:
    if not YADA_CONFIG_FILE_PATH.is_file():
        YADA_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(YADA_CONFIG_FILE_PATH, "w") as f:
            f.write("")

    return str(YADA_CONFIG_FILE_PATH)


class Config(BaseSettings):
    api_key: Optional[str] = ""
    llm_model_name: Optional[str] = "gpt-4o"
    custom_tools_dir: Optional[str] = ""
    model_config = SettingsConfigDict(
        env_file=get_or_create_yada_config_file(),
        arbitrary_types_allowed=False,
        extra="allow",
    )


_config = Config()


def get_config() -> Config:
    return _config


def reload_config() -> Config:
    global _config
    _config = Config()
    return _config


def _read_config_file() -> dict:
    # Read the key=value file
    config = configparser.ConfigParser()

    yada_config_file = get_or_create_yada_config_file()

    # We have to fake a section since configparser requires sections
    with open(yada_config_file, "r") as file:
        content = "[{}]\n".format(_SECTION_NAME) + file.read()

    config.read_string(content)

    return config


def _write_config_and_reload(config: configparser.ConfigParser) -> None:
    # Write changes back to the file (remove section header)
    with open(get_or_create_yada_config_file(), "w") as file:
        for key, value in config.items(_SECTION_NAME):
            file.write(f"{key}={value}\n")

    reload_config()


def set_api_key(api_key: str) -> None:
    config = _read_config_file()
    config[_SECTION_NAME]["api_key"] = api_key
    _write_config_and_reload(config)


def set_llm_model_name(llm_model_name: str) -> None:
    config = _read_config_file()
    config[_SECTION_NAME]["llm_model_name"] = llm_model_name
    _write_config_and_reload(config)


def set_custom_tools_dir(custom_tools_dir: str) -> None:
    config = _read_config_file()
    config[_SECTION_NAME]["custom_tools_dir"] = custom_tools_dir
    _write_config_and_reload(config)


config_selections = [
    {
        "name": "API Key",
        "update_func": set_api_key,
    },
    {
        "name": "LLM Model Name",
        "update_func": set_llm_model_name,
    },
    {
        "name": "Custom Tools Directory",
        "update_func": set_custom_tools_dir,
    },
]
