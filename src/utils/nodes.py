from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from utils.tools import tools
from langgraph.prebuilt import ToolNode

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20241022")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


# Node - llm_with_tools
def chatbot(state, config):
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    return {"messages": [model.invoke(state["messages"])]}


tool_node = ToolNode(tools)