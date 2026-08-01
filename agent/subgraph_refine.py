from langgraph.graph import StateGraph, END
from agent.state import SubgraphState
from agent.nodes import grade_documents, transform_query, re_retrieve

def check_relevance(state: SubgraphState):
    """
    Decides whether to keep looping or exit the subgraph.
    """
    print("--- [SUBGRAPH] CHECKING RELEVANCE ---")
    
    # 1. Safety valve for local hardware limits
    if state.get("loop_count", 0) >= 3:
        print("--- [SUBGRAPH] MAX LOOPS REACHED. EXITING SUBGRAPH. ---")
        return "end"
        
    # 2. Check if we found good documents
    if len(state["documents"]) > 0:
        print("--- [SUBGRAPH] RELEVANT DOCS FOUND. EXITING SUBGRAPH. ---")
        return "end"
        
    # 3. If no good documents, transform the query and try again
    print("--- [SUBGRAPH] NO RELEVANT DOCS. REWRITING QUERY. ---")
    return "transform"

# Initialize the graph with the Subgraph-specific state
workflow = StateGraph(SubgraphState)

# Add our nodes
workflow.add_node("grade", grade_documents)
workflow.add_node("transform", transform_query)
workflow.add_node("re_retrieve", re_retrieve)

# Define the flow
workflow.set_entry_point("grade")

# The conditional routing edge
workflow.add_conditional_edges(
    "grade",
    check_relevance,
    {
        "end": END,
        "transform": "transform"
    }
)

# Complete the loop
workflow.add_edge("transform", "re_retrieve")
workflow.add_edge("re_retrieve", "grade")

# Compile the subgraph into a runnable application
refinement_app = workflow.compile()