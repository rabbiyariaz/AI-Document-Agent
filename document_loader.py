from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings




# Load
loader = PyPDFLoader("RabbiyaRiaz__AI engineer.pdf")
docs = loader.load()

print(f"Total pages: {len(docs)}")
print(f"First page preview: {docs[0].page_content[:200]}")
print(f"Metadata: {docs[0].metadata}")
print(len(docs[0].page_content))  # total characters on page 1

# # Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")
print(f"First chunk: {chunks[0].page_content}")
# Embed & store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)



# Search by meaning
results = vector_store.similarity_search("What are Rabbiya's technical skills?", k=2)
for r in results:
    print(len(r.page_content))
    print("---")


