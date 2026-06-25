from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("RabbiyaRiaz__AI engineer.pdf")
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
web_search = DuckDuckGoSearchRun()


@tool
def search_web(query: str) -> str:
    """Search the web for current information not found in the document."""
    return web_search.run(query)


@tool
def search_document(query: str) -> str:
    """Search the document for relevant information based on a query."""
    results = retriever.invoke(query)
    return "\n\n".join(r.page_content for r in results)

# --- Agent ---
agent = create_agent(
    model=model,
    tools=[search_document, search_web],
    system_prompt="""You are a helpful assistant with access to two tools:
    1. search_document - use this for questions about the person's CV and background
    2. search_web - use this for current information, news, or anything not in the document
    Always choose the right tool based on the question.""" )

# --- Memory loop ---
chat_history = []

def chat(user_input: str):
    chat_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages": chat_history})
    
    # Show thinking process
    print("\n--- AGENT THINKING PROCESS ---")
    for msg in response["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            print(f"👤 Human: {msg.content}")
        elif msg_type == "AIMessage":
            if msg.tool_calls:
                print(f"🤖 Agent decided to use tool: {msg.tool_calls[0]['name']}")
                print(f"   With input: {msg.tool_calls[0]['args']}")
            else:
                print(f"🤖 Agent final answer: {msg.content}")
        elif msg_type == "ToolMessage":
            print(f"🔧 Tool returned: {msg.content[:200]}...")
    print("--- END ---\n")
    
    ai_message = response["messages"][-1]
    chat_history.append(ai_message)
    print(f"AI: {ai_message.content}\n")

# def chat(user_input: str):
#     chat_history.append(HumanMessage(content=user_input))
#     response = agent.invoke({"messages": chat_history})
#     ai_message = response["messages"][-1]
#     chat_history.append(ai_message)
#     print(f"AI: {ai_message.content}\n")

print("Document Agent Ready! Ask me anything about the document.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    chat(user_input)