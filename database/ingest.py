import os
import certifi
import pdfplumber
from pymongo import MongoClient
from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.readers.base import BaseReader

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(root_dir, '.env'))

class PDFPlumberReader(BaseReader):
    """Custom parser to extract text cleanly, preserving tabular structures."""
    def load_data(self, file, extra_info=None):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True) # attempts to preserve visual spacing (like tables)
                if page_text:
                    text += page_text + "\n\n"
        
        return [Document(text=text, extra_info=extra_info or {})]

def ingest_documents():
    print("--- STARTING DATA INGESTION ---")
    
    # 1. Setup the data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' directory. Please drop some PDFs or TXT files inside and run again.")
        return

    file_extractor: dict[str, BaseReader] = {".pdf": PDFPlumberReader()}

    # Load documents using LlamaIndex's built-in directory parser
    documents = SimpleDirectoryReader(
        data_dir, 
        file_extractor=file_extractor,
    ).load_data()

    if not documents:
        print(f"No documents found in '{data_dir}'. Add some files and try again!")
        return
        
    print(f"Loaded {len(documents)} document chunks/pages.")

    # 2. Initialize the Local Embedding Model
    embed_model = OpenAIEmbedding(
        model_name="openai/text-embedding-3-small", 
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        embed_batch_size=100
    )

    # 3. Setup MongoDB Connection
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        print("Error: MONGODB_URI not found in .env file.")
        return
        
    mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    
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