from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

agent = create_agent(model=model, tools=[get_word_length],
                     system_prompt="You are a helpful assistant.")

# Memory — just a list of messages
chat_history = []

def chat(user_input: str):
    chat_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages": chat_history})
    ai_message = response["messages"][-1]
    chat_history.append(ai_message)
    
    print(f"AI: {ai_message.content}\n")

chat("My name is Rabbiya.")
chat("What is the length of my name?")
chat("What was the first thing I told you?")