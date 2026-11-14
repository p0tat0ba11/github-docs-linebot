import os
from typing import Any, Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from github_loader import load_github_documents

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
MAX_HISTORY_LENGTH = 20

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question which might reference context "
    "in the chat history, formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just reformulate it if "
    "needed and otherwise return it as is."
)

SYSTEM_PROMPT = """You are the official AI assistant for p0tat0ba11's blog (https://p0tat0ba11.github.io/). 
Your job is to help visitors explore articles, understand concepts, and find specific information written on the blog.

Guidelines:
1. Strict Grounding: Base your answers strictly on the provided Context below. Do not invent facts, assume details, or bring in outside knowledge not supported by the articles.
2. Helpful Fallbacks: If the Context does not contain the answer, politely let the user know that the blog hasn't covered that specific topic yet, and briefly mention what related topics ARE in the context (if any).
3. LINE-Friendly Formatting: Users are reading on mobile screens. Keep answers concise, clear, and well-structured using short paragraphs and bullet points. Avoid heavy Markdown tables or overly complex formatting that looks messy in chat.
4. Language Matching: Reply in the exact same language the user used to ask their question (e.g., if they ask in Traditional Chinese, answer in Traditional Chinese based on the context; if in English, answer in English).
5. Conversational & Welcoming: If the user simply greets you (e.g., "Hi", "你好"), warmly welcome them and briefly explain that you can help them search and summarize articles from p0tat0ba11's blog.

Context:
{context}"""

session: Dict[str, ChatMessageHistory] = {}


def get_user_history(user_id: str, max_length: int = MAX_HISTORY_LENGTH) -> ChatMessageHistory:
    if user_id not in session:
        session[user_id] = ChatMessageHistory()
    history = session[user_id]
    if len(history.messages) > max_length:
        history.messages = history.messages[-max_length:]
    return history


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_retriever(
    chunks: List[Document],
    embedding_model: str = EMBEDDING_MODEL,
    search_kwargs: Optional[Dict[str, Any]] = None,
) -> VectorStoreRetriever:
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    effective_kwargs = search_kwargs if search_kwargs else {"k": 3}
    return vectorstore.as_retriever(search_kwargs=effective_kwargs)


def build_qa_chain(
    retriever: VectorStoreRetriever,
    llm_model: str = LLM_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> RunnableWithMessageHistory:
    llm = ChatGoogleGenerativeAI(model=llm_model)

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    def get_standalone_question(inputs: Dict[str, Any]) -> str:
        if inputs.get("chat_history"):
            return contextualize_chain.invoke(inputs)
        return inputs["input"]

    # Fixed: .from_messages() instead of .from_template(), and "chat_history" placeholder
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(get_standalone_question(x)))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_user_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def initialize_rag_pipeline(
    documents: Optional[List[Document]] = None,
    embedding_model: str = EMBEDDING_MODEL,
    llm_model: str = LLM_MODEL,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    search_kwargs: Optional[Dict[str, Any]] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> RunnableWithMessageHistory:
    docs = documents if documents else load_github_documents()
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = create_retriever(
        chunks,
        embedding_model=embedding_model,
        search_kwargs=search_kwargs,
    )
    return build_qa_chain(retriever, llm_model, system_prompt)


if __name__ == "__main__":
    qa_chain = initialize_rag_pipeline(search_kwargs={"k": 3})
    question = "What is the main topic of the documents?"
    result = qa_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "test_user"}},
    )
    print(f"Query: {question}\nAnswer: {result}")