import os
import time
import certifi
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(root_dir, '.env'))

def setup_database():
    print("--- CONFIGURING MONGODB ATLAS INFRASTRUCTURE ---")
    
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        print("❌ Error: MONGODB_URI not found in .env file.")
        return

    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    
    db_name = "crag_database"
    collection_name = "vector_store"
    index_name = "vector_index"
    
    db = client[db_name]
    
    # 1. Create the collection if it doesn't exist (Index needs a collection to attach to)
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"✅ Created empty collection: '{collection_name}' in '{db_name}'")
    else:
        print(f"✅ Collection '{collection_name}' already exists.")
        
    collection = db[collection_name]
    
    # 2. Check if the index already exists
    existing_indexes = list(collection.list_search_indexes())
    if any(idx.get("name") == index_name for idx in existing_indexes):
        print(f"✅ Vector search index '{index_name}' already exists. You are good to go!")
        return

    print(f"⏳ Creating vector search index '{index_name}'...")
    
    # 3. Define the Vector Search Index Model
    # nomic-embed-text outputs 768 dimensions. We use cosine similarity for comparison.
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 768,
                    "path": "embedding",
                    "similarity": "cosine"
                }
            ]
        },
        name=index_name,
        type="vectorSearch"
    )
    
    # 4. Push the index configuration to Atlas
    collection.create_search_index(model=search_index_model)
    
    print("⏳ Index creation initiated in the cloud. This usually takes 1-2 minutes.")
    print("⏳ Waiting for index to become READY...")
    
    # 5. Wait for the index to finish building
    while True:
        # Fetch the specific index status
        indexes = list(collection.list_search_indexes(index_name))
        if indexes and indexes[0].get("status") == "READY":
            print("\n🚀 SUCCESS: Index is READY!")
            print("You can now safely run 'python ingest.py' to add your documents.")
            break
            
        print("Still building... checking again in 10 seconds.")
        time.sleep(10)

if __name__ == "__main__":
    setup_database()