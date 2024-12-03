from utils.tools import tools
from langgraph.prebuilt import ToolNode

from services.llmprovider import LLMProvider
model = LLMProvider("anthropic").bind_tools(tools) # "openai""anthropic" or "ollama"


def chatbot(state):
    return {"messages": [model.invoke(state["messages"])]}

tool_node = ToolNode(tools)