
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

print("1. Script started")

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"

PDF_PATH = "islr.pdf"

# Check PDF path
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}. "
        "Place islr.pdf in the project directory."
    )

# Check Groq API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

# 1. Load PDF
print("2. Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

if not docs:
    raise ValueError("The PDF contains no readable pages.")

print("PDF pages loaded:", len(docs))

# 2. Split documents into chunks
print("3. Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
splits = splitter.split_documents(docs)

if not splits:
    raise ValueError("No text chunks were created from the PDF.")

print("Document chunks:", len(splits))

# 3. Create embeddings
print("4. Loading embedding model...")
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create FAISS vector store
print("5. Creating FAISS index...")
vs = FAISS.from_documents(splits, emb)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

print("FAISS vector store created successfully")

# Test retrieval before creating the chain
test_query = "What is statistical learning?"
results = retriever.invoke(test_query)

print("Retrieved documents:", len(results))
if results:
    print("First retrieved chunk:")
    print(results[0].page_content[:300])

# 5. Define prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer only from the provided context. "
        "If the answer is not found, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    ),
])

# 6. Configure Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=500,
)

# Format retrieved documents
def format_docs(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )

# 7. Build RAG chain
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

print("\n6. RAG chain created successfully!")
print("PDF RAG ready. Ask a question (or type 'exit').")

# 8. Ask multiple questions
while True:
    try:
        question = input("\nQ: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        answer = chain.invoke(question)
        print("\nA:", answer)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
