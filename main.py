import os
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from agent.graph_parent import workflow
#from python_dotenv import load_dotenv

#load_dotenv()  # Load environment variables from .env file

def main():
    print("--- INITIALIZING CRAG PIPELINE ---")
    
    # Connect to your local MongoDB instance
    # Ensure you have MongoDB running locally on default port 27017
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongo_client = MongoClient(MONGODB_URI)
    
    # MongoDBSaver automatically creates a "langgraph_checkpoints" database

    checkpointer = MongoDBSaver(mongo_client)

    # Compile the workflow WITH the checkpointer
    print("--- COMPILING GRAPH WITH MONGODB MEMORY ---")
    app = workflow.compile(checkpointer=checkpointer)
        
    # The thread_id acts as the unique identifier for a user's session.
    #  LangGraph will use this ID to fetch previous state from MongoDB.
    config = {"configurable": {"thread_id": "local_dev_session_1"}}
        
    user_query = "What is the capital of France?"
    print(f"\nUser Query: {user_query}\n")
        
    initial_state = {"original_question": user_query}
        
    # Stream the execution so we can watch the agentic loop in real-time
    for output in app.stream(initial_state, config, stream_mode="updates"):
        # Print which node just finished executing
        for node_name, state_update in output.items():
            print(f"\n✅ Completed Node: {node_name}")

            # If we have reached the final answer, display it
            if "final_answer" in state_update:
                print("\n" + "="*40)
                print("FINAL ANSWER:")
                print("="*40)
                print(state_update["final_answer"])
                print("="*40 + "\n")

if __name__ == "__main__":
    main()