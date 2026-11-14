import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from github_loader import load_github_documents

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

SYSTEM_PROMPT = """You are the official AI assistant for p0tat0ba11's blog (https://p0tat0ba11.github.io/). 
Your job is to help visitors explore articles, understand concepts, and find specific information written on the blog.

Guidelines:
1. Strict Grounding: Base your answers strictly on the provided Context below. Do not invent facts, assume details, or bring in outside knowledge not supported by the articles.
2. Helpful Fallbacks: If the Context does not contain the answer, politely let the user know that the blog hasn't covered that specific topic yet, and briefly mention what related topics ARE in the context (if any).
3. LINE-Friendly Formatting: Users are reading on mobile screens. Keep answers concise, clear, and well-structured using short paragraphs and bullet points. Avoid heavy Markdown tables or overly complex formatting that looks messy in chat.
4. Language Matching: Reply in the exact same language the user used to ask their question (e.g., if they ask in Traditional Chinese, answer in Traditional Chinese based on the context; if in English, answer in English).
5. Conversational & Welcoming: If the user simply greets you (e.g., "Hi", "你好"), warmly welcome them and briefly explain that you can help them search and summarize articles from p0tat0ba11's blog.

Context:
{context}

Question: {question}
Answer:"""

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def split_documents(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_retriever(
        documents: List[Document],
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        search_kwargs: Optional[dict] = {}
) -> FAISS:
    document_chunks = split_documents(documents, chunk_size, chunk_overlap)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def build_qa_chain(
        retriever,
        llm_model: str = LLM_MODEL,
        prompt_template: str = SYSTEM_PROMPT
):
    llm = ChatGoogleGenerativeAI(model=llm_model)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def initialize_rag_pipeline(
        documents: List[Document] = None,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        search_kwargs: Optional[dict] = {},
        prompt_template: str = SYSTEM_PROMPT
):
    docs = documents if documents else load_github_documents()
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    retriever = create_retriever(chunks, embedding_model, chunk_size, chunk_overlap, search_kwargs)
    return build_qa_chain(retriever, llm_model, prompt_template)

if __name__ == "__main__":
    qa_chain = initialize_rag_pipeline(search_kwargs={"k": 3})
    question = "What is the main topic of the documents?"
    result = qa_chain.invoke(question)
    print(f"Query: {question}\nAnswer: {result}")