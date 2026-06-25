from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

agent = create_agent(
    model=model,
    tools=[get_word_length],
    system_prompt="You are a helpful assistant."
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you tell me the length of the word 'LangChain'?"
            }
        ]
    }
)

print(response["messages"][-1].content)