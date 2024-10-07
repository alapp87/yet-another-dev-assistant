from langchain_openai import ChatOpenAI

from yada.config import get_config


def model() -> ChatOpenAI:
    return ChatOpenAI(model=get_config().llm_model_name, api_key=get_config().api_key)


def custom_tools_dir() -> str:
    return get_config().custom_tools_dir
