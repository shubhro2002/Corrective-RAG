from typing import Optional, TypedDict, List

class ParentState(TypedDict):
    """
    The high-level state managed by the main execution graph.
    This contains only the final inputs and outputs needed for the user.
    """
    # Input from the user
    original_question: str
    
    # The payload of retrieved context. 
    # This gets passed down to the subgraph and overwritten with refined documents.
    documents: List[str]
    
    # The final synthesized answer generated after the subgraph finishes
    final_answer: Optional[str]


class SubgraphState(TypedDict):
    """
    The internal state managed exclusively by the refinement subgraph.
    It inherits the communication keys from the parent and adds its own 
    internal tracking variables.
    """
    original_question: str      # Read-only for the subgraph (used to rewrite queries)
    documents: List[str]        # Read & Write: Graded and re-retrieved here
    
    # --- INTERNAL KEYS (Only exist inside the Subgraph) ---
    current_query: str          # The actively transformed query (starts as original_question)
    loop_count: int