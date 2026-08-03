import streamlit as st
import os
import sys
import uuid
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.runnables import RunnableConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.graph_parent import workflow
from agent.state import ParentState

st.set_page_config(page_title="Acme Corp Knowledge Base", page_icon="🧠", layout="wide")

# --- INITIALIZATION ---
# We use @st.cache_resource so the database connection and graph compilation 
# only happen once, instead of every time the user clicks a button.
@st.cache_resource
def load_agent():
    load_dotenv()
    MONGODB_URI = os.environ.get("MONGODB_URI")
    if not MONGODB_URI:
        st.error("❌ MONGODB_URI not found in .env file.")
        st.stop()
        
    mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    checkpointer = MongoDBSaver(mongo_client)
    return workflow.compile(checkpointer=checkpointer)

app = load_agent()

# Initialize the chat log
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Acme Corp Knowledge Agent. Ask me anything about our policies, tech stack, or projects."}]

# We generate a unique Thread ID for the session to utilize MongoDB memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.title("🧠 Agentic CRAG")
    st.markdown("A localized, multi-agent Corrective RAG system running on a 3B LLM.")
    st.divider()
    st.markdown("**Session Memory ID:**")
    st.code(st.session_state.thread_id)
    st.markdown("*This ID allows the agent to remember your conversation context across turns.*")
    
    # Button to wipe memory
    if st.button("Clear History & Start Fresh"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the Acme Corp Knowledge Agent. Ask me anything about our policies, tech stack, or projects."}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.title("Acme Corp RAG Portal")

# Draw the existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input (This waits for the user to hit 'Enter')
if user_query := st.chat_input("Ask a question about Acme Corp..."):
    # 1. Display user message in UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Run the agent
    with st.chat_message("assistant"):
        # We use st.status to create a nice expandable "thinking" spinner
        status = st.status("Agent is querying database and grading documents...", expanded=True)
        with status:
            st.write("Initializing retrieval subgraph...")
            st.write("*(Check your terminal window for detailed LLM debug logs!)*")
            
            config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_input = ParentState(
                original_question=user_query,
                documents=[],
                final_answer=None
            )
            
            # Execute the LangGraph workflow
            final_state = app.invoke(initial_input, config=config)
            
            final_answer = final_state.get("final_answer", "No answer generated.")
            status.update(label="Agent finished processing!", state="complete", expanded=False)
            
        # 3. Display the final answer in the UI
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})