import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def setup_agent():
    loader = PyPDFLoader("RabbiyaRiaz__AI engineer.pdf")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    web_search = DuckDuckGoSearchRun()

    @tool
    def search_document(query: str) -> str:
        """Search the document for relevant information."""
        results = retriever.invoke(query)
        return "\n\n".join(r.page_content for r in results)

    @tool
    def search_web(query: str) -> str:
        """Search the web for current information."""
        try:
            return web_search.run(query)
        except Exception as e:
            return f"Web search failed: {str(e)}. Try rephrasing."

    agent = create_agent(
        model=model,
        tools=[search_document, search_web],
        system_prompt="You are a helpful assistant. Use search_document for CV questions, search_web for general questions."
    )
    return agent

st.title("📄 AI Document Agent")
st.caption("Ask me anything about the document or the web!")

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = setup_agent()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get agent response
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages": st.session_state.chat_history})
    ai_message = response["messages"][-1]
    st.session_state.chat_history.append(ai_message)

    # Show agent response
    st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
    with st.chat_message("assistant"):
        st.write(ai_message.content)