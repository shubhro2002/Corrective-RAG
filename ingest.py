import sys
import os

from pymongo import MongoClient
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.node_parser import TokenTextSplitter

# Load environment variables from .env file
load_dotenv()

def ingest_documents():
    print("--- STARTING DATA INGESTION ---")
    
    # 1. Setup the data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' directory. Please drop some PDFs or TXT files inside and run again.")
        return

    # Load documents using LlamaIndex's built-in directory parser
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        print(f"No documents found in '{data_dir}'. Add some files and try again!")
        return
        
    print(f"Loaded {len(documents)} document chunks/pages.")

    # 2. Initialize the Local Embedding Model
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # 3. Setup MongoDB Connection
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI not found in .env file.")
        return
        
    mongo_client = MongoClient(MONGODB_URI)
    
    DB_NAME = "crag_database"
    COLLECTION_NAME = "vector_store"
    
    # Connect LlamaIndex to MongoDB
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=mongo_client,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        vector_index_name="vector_index"
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Chunking Configuration
    # Switching to TokenTextSplitter completely bypasses the NLTK Windows bug 
    # and provides more accurate token limits for our 3B model.
    parser = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

    # 5. Process and Store
    print("Chunking, generating embeddings, and saving to MongoDB... This might take a minute.")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[parser],
        show_progress=True
    )
    
    print("--- INGESTION COMPLETE ---")
    print(f"Documents vectorized and stored in MongoDB database: '{DB_NAME}'")

if __name__ == "__main__":
    ingest_documents()