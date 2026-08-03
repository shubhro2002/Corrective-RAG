import os
import sys
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.runnables import RunnableConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.graph_parent import workflow
from agent.state import ParentState

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(root_dir, '.env'))

def main():
    print("--- INITIALIZING CRAG PIPELINE ---")
    
    # 1. Setup MongoDB Connection
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        print("Error: MONGODB_URI not found in .env file.")
        return
        
    mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    
    # Configure the Checkpointer (Memory)
    print("--- COMPILING GRAPH WITH MONGODB MEMORY ---\n")
    
    # We pass the MongoDB connection to LangGraph so it can save conversation states
    checkpointer = MongoDBSaver(mongo_client)
    
    # Compile the graph with the checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    # Define the Thread (Conversation ID)
    # In a real app, this would be a unique UUID per user session
    config: RunnableConfig = {"configurable": {"thread_id": "project_user_1"}}
    
    # Execute the Graph
    user_query = "How much did the software division of Acme Corp grow?"
    print(f"User Query: {user_query}\n")
    
    # Explicitly instantiate the TypedDict to satisfy static type checkers
    initial_input = ParentState(
        original_question=user_query,
        documents=[],
        final_answer=None
    )
    
    # Run the pipeline
    final_state = app.invoke(
        initial_input,
        config=config
    )
    
    # 5. Output the Result
    print("\n========================================")
    print("FINAL ANSWER:")
    print("========================================")
    print(final_state.get("final_answer", "No answer generated."))
    print("========================================")

if __name__ == "__main__":
    main()