import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

def clear_database():
    print("--- CONNECTING TO MONGODB ---")
    load_dotenv()
    
    # 1. Connect to MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI")
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    
    # 2. Select Database and Collection
    db = client["crag_database"]
    collection = db["vector_store"]
    
    # 3. Delete all documents to remove the corrupted PDF gibberish
    result = collection.delete_many({})
    
    print(f"SUCCESS: Cleared {result.deleted_count} corrupted chunks from MongoDB.")
    print("You can now safely run ingest.py to load the clean text data.")

if __name__ == "__main__":
    clear_database()