from langgraph.graph import StateGraph, END
from agent.state import ParentState
from agent.nodes import initial_retrieve, generate_answer
from agent.subgraph_refine import refinement_app

# Initialize the graph with the high-level ParentState
workflow = StateGraph(ParentState)

# Add the standard function nodes
workflow.add_node("initial_retrieve", initial_retrieve)
workflow.add_node("generate_answer", generate_answer)

# We add the compiled subgraph directly as a node!
# LangGraph will automatically map the matching keys (documents, original_question) 
# between ParentState and SubgraphState.
workflow.add_node("refine_docs", refinement_app)

# The parent flow is strictly linear. All the complex looping is hidden 
# inside the "refine_docs" subgraph node.
workflow.set_entry_point("initial_retrieve")
workflow.add_edge("initial_retrieve", "refine_docs")
workflow.add_edge("refine_docs", "generate_answer")
workflow.add_edge("generate_answer", END)