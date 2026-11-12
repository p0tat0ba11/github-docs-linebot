from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from github_loader import load_github_documents

# 1. Load and split documents
documents = load_github_documents()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunks = splitter.split_documents(documents)

# 2. Build FAISS vectorstore and retriever
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vectorstore = FAISS.from_documents(document_chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# 4. Build QA Chain using LCEL
prompt = ChatPromptTemplate.from_template(
    """Answer the question based strictly on the following context. 
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Invoke the chain
example_query = "What is the main topic of the documents?"
result = qa_chain.invoke(example_query)

print(f"Query: {example_query}\nAnswer: {result}")