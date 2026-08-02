import sys
import os

from pymongo import MongoClient
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.node_parser import TokenTextSplitter
from langgraph.checkpoint.mongodb import MongoDBSaver

# Import our custom graph
from agent.graph_parent import workflow

# Load environment variables
load_dotenv()

def main():
    print("--- INITIALIZING CRAG PIPELINE ---")
    
    # 1. Setup MongoDB Connection
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI not found in .env file.")
        return
        
    mongo_client = MongoClient(MONGODB_URI)
    
    # 2. Configure the Checkpointer (Memory)
    print("--- COMPILING GRAPH WITH MONGODB MEMORY ---\n")
    
    # We pass the MongoDB connection to LangGraph so it can save conversation states
    checkpointer = MongoDBSaver(mongo_client)
    
    # Compile the graph with the checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    # 3. Define the Thread (Conversation ID)
    # In a real app, this would be a unique UUID per user session
    config = {"configurable": {"thread_id": "resume_project_user_1"}}
    
    # 4. Execute the Graph
    user_query = "How much did the software division of Acme Corp grow?"
    print(f"User Query: {user_query}\n")
    
    # Run the pipeline
    final_state = app.invoke(
        {"original_question": user_query},
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