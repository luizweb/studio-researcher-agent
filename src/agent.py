from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from utils.state import State
from utils.nodes import oracle, tool_node

# Initialize the state graph with AgentState to manage the workflow.
workflow = StateGraph(State)
workflow.set_entry_point("oracle")
workflow.add_node("oracle", oracle)
workflow.add_node("tools", tool_node)

workflow.add_conditional_edges("oracle", tools_condition)
workflow.add_edge("tools", "oracle")

# Compile the workflow to make it executable.
graph = workflow.compile()