from langgraph.graph import StateGraph, END

from utils.state import AgentState
from utils.nodes import run_oracle, run_tool, router
from utils.tools import tools

# Initialize the state graph with AgentState to manage the workflow.
workflow = StateGraph(AgentState)

workflow.add_node('oracle', run_oracle)
workflow.add_node('rag_search_filter', run_tool)
workflow.add_node('rag_search', run_tool)
workflow.add_node('fetch_arxiv', run_tool)
workflow.add_node('web_search', run_tool)
workflow.add_node('final_answer', run_tool)

# Set the entry point to 'oracle'.
workflow.set_entry_point('oracle')

# Add conditional edges to determine the next step using the router function.
workflow.add_conditional_edges(source='oracle', path=router)

# Add edges from each tool back to 'oracle', except 'final_answer', which leads to 'END'.
for tool_obj in tools:
    if tool_obj.name != 'final_answer':
        workflow.add_edge(tool_obj.name, 'oracle')

workflow.add_edge('final_answer', END)

# Compile the workflow to make it executable.
graph = workflow.compile()