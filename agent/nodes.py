import os
import certifi
from pymongo import MongoClient
from pymongo import MongoClient
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from agent.state import ParentState, SubgraphState

# Load environment variables from .env file
load_dotenv()

# ==========================================
# 1. Initialize Local Models & Vector Store
# ==========================================
llm = Ollama(model="llama3.2:3b", request_timeout=120.0, temperature=0.2)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Connect to MongoDB
MONGODB_URI = os.environ.get("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())

vector_store = MongoDBAtlasVectorSearch(
    mongodb_client=mongo_client,
    db_name="crag_database",
    collection_name="vector_store",
    vector_index_name="vector_index" 
)

# Load the index directly from the database
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

# Retrieve the top 2 documents
retriever = index.as_retriever(similarity_top_k=2)

# ==========================================
# 2. Parent Graph Nodes
# ==========================================

def initial_retrieve(state: ParentState):
    """Fetches the first batch of documents based on the original user question."""
    question = state["original_question"]
    print(f"--- [PARENT] INITIAL RETRIEVAL for: {question} ---")
    
    nodes = retriever.retrieve(question)
    docs = [node.get_content() for node in nodes]
    
    # 🔍 ADDED VISIBILITY: Let's see what MongoDB is actually returning
    print(f"🔍 [DEBUG] Retrieved {len(docs)} chunks from MongoDB.")
    for i, doc in enumerate(docs):
        # Print the first 150 characters of each chunk
        print(f"   Snippet {i+1}: {doc[:150].replace(chr(10), ' ')}...")
    
    return {"documents": docs}

def generate_answer(state: ParentState):
    """Synthesizes the final answer using the refined documents."""
    question = state["original_question"]
    docs = state["documents"]
    print("--- [PARENT] GENERATING FINAL ANSWER ---")
    
    context = "\n\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely based only on the context:"
    
    response = llm.complete(prompt)
    return {"final_answer": str(response)}

# ==========================================
# 3. Subgraph Nodes (Refinement Loop)
# ==========================================

def grade_documents(state: SubgraphState):
    """Evaluates the relevance of retrieved documents."""
    print("--- [SUBGRAPH] GRADING DOCUMENTS ---")
    query = state.get("current_query", state["original_question"])
    docs = state["documents"]
    
    relevant_docs = []
    
    for doc in docs:
        # 🧠 PROMPT TUNING: Made the grader slightly more lenient for the 3B model
        prompt = (
            f"You are a grader evaluating document relevance.\n"
            f"Question: {query}\n"
            f"Document: {doc}\n"
            f"If the document contains ANY numbers, names, or keywords related to the question, answer 'yes'. "
            f"Otherwise, answer 'no'. Answer ONLY 'yes' or 'no'."
        )
        response = llm.complete(prompt).text.strip().lower()
        
        # 🔍 ADDED VISIBILITY: See exactly what the LLM decided
        print(f"⚖️ [DEBUG] Grader LLM responded: '{response}'")
        
        if "yes" in response:
            relevant_docs.append(doc)
            
    current_loop = state.get("loop_count", 0)
            
    return {"documents": relevant_docs, "loop_count": current_loop + 1}

def transform_query(state: SubgraphState):
    """Rewrites the query if the previous documents were graded as irrelevant."""
    print("--- [SUBGRAPH] TRANSFORMING QUERY ---")
    question = state["original_question"]
    
    # 🧠 PROMPT TUNING: Forbid the LLM from hallucinating dates/facts
    prompt = (
        f"You are an AI assistant tasked with query optimization.\n"
        f"The user asked: {question}\n"
        f"Rewrite this question to be more specific and optimized for a vector database search. "
        f"Do NOT add new information, dates, or facts that were not in the original question. "
        f"Provide ONLY the new query text."
    )
    
    response = llm.complete(prompt)
    new_query = str(response).strip()
    
    print(f"--- [SUBGRAPH] NEW QUERY: {new_query} ---")
    return {"current_query": new_query}

def re_retrieve(state: SubgraphState):
    """Fetches a new batch of documents using the transformed query."""
    print("--- [SUBGRAPH] RE-RETRIEVING ---")
    query = state["current_query"]
    
    nodes = retriever.retrieve(query)
    docs = [node.get_content() for node in nodes]
    
    print(f"🔍 [DEBUG] Re-retrieved {len(docs)} chunks from MongoDB.")
    
    return {"documents": docs}