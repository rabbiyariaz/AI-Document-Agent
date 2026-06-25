from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a technical explainer. Be concise."),
    ("human", "Explain {topic} in exactly 3 lines.")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate to Urdu."),
    ("human", "{explanation}")
])


chain1=prompt1 | model | StrOutputParser()
chain2=prompt2 | model | StrOutputParser()
explanation = chain1.invoke({"topic": "quantum computing"})
translation = chain2.invoke({"explanation": explanation})
print("Explanation:", explanation)
print("Translation:", translation)
