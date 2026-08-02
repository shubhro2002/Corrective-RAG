from typing import Any, List
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Document

from agent.state import ParentState, SubgraphState

llm = Ollama(model="llama3.2", request_timeout=120.0, temperature=0.3)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

dummy_index = VectorStoreIndex.from_documents(
    [Document(text="This is a placeholder document about AI.")], 
    embed_model=embed_model
)
retriever = dummy_index.as_retriever(similarity_top_k=3)

def initial_retrieve(state: ParentState):
    """Fetches the first batch of documents based on the original user question."""
    question = state["original_question"]
    print(f"--- [PARENT] INITIAL RETRIEVAL for: {question} ---")
    
    nodes = retriever.retrieve(question)
    docs = [node.get_content() for node in nodes]
    
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

def grade_documents(state: SubgraphState):
    """Evaluates the relevance of retrieved documents."""
    print("--- [SUBGRAPH] GRADING DOCUMENTS ---")
    query = state.get("current_query", state["original_question"])
    docs = state["documents"]
    
    relevant_docs = []
    
    for doc in docs:
        prompt = (
            f"You are a strict grader evaluating document relevance.\n"
            f"Question: {query}\n"
            f"Document: {doc}\n"
            f"Is this document relevant to the question? Answer ONLY 'yes' or 'no'."
        )
        response = llm.complete(prompt).text.strip().lower()
        
        if "yes" in response:
            relevant_docs.append(doc)
            
    # Initialize loop count if it doesn't exist
    current_loop = state.get("loop_count", 0)
            
    return {"documents": relevant_docs, "loop_count": current_loop + 1}

def transform_query(state: SubgraphState):
    """Rewrites the query if the previous documents were graded as irrelevant."""
    print("--- [SUBGRAPH] TRANSFORMING QUERY ---")
    question = state["original_question"]
    
    prompt = (
        f"You are an AI assistant tasked with query optimization.\n"
        f"The user asked: {question}\n"
        f"Rewrite this question to be more specific and optimized for a vector database search. "
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
    
    return {"documents": docs}