
from langchain_core.messages import SystemMessage
from utils.tools import tools
from langgraph.prebuilt import ToolNode


from services.llmprovider import LLMProvider
llm = LLMProvider("openai").bind_tools(tools) 


system_message = SystemMessage(content=
    '''You are the oracle, the great AI decision-maker.
    Given the user's query, you must decide what to do with it based on the
    list of tools provided to you.

    If you see that a tool has been used with a particular
    query, do NOT use that same tool with the same query again. Also, do NOT use
    any tool more than twice.

    You should aim to collect information from a diverse range of sources before
    providing the answer to the user. Once you have collected plenty of information
    to answer the user's question, use the final_answer tool.'''
)


def oracle(state):
    return {"messages": [llm.invoke([system_message] + state["messages"])]}


tool_node = ToolNode(tools)