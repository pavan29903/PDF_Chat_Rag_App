import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

TEMP_PDF_DIR = "temp_uploaded_pdfs"
os.makedirs(TEMP_PDF_DIR, exist_ok=True) 


@st.cache_resource
def get_vectorstore_from_pdfs(file_paths):
    """
    Loads multiple PDFs, splits them into chunks, and creates a Chroma vector store.
    This function is cached to avoid re-processing PDFs on every rerun.
    """
    if not file_paths:
        return None

    all_documents = []
    with st.spinner("Processing PDF(s) and creating knowledge base... This might take a moment."):
        for file_path in file_paths:
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
                st.info(f"Loaded {len(documents)} pages from '{os.path.basename(file_path)}'.")
            except Exception as e:
                st.error(f"Error loading '{os.path.basename(file_path)}': {e}")
                continue 

        if not all_documents:
            st.warning("No documents could be loaded from the provided PDF(s). Please ensure they are valid PDF files.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
        splits = text_splitter.split_documents(all_documents)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        st.success("Knowledge base created successfully from uploaded PDF(s)! You can now ask questions.")
        return vectorstore

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a chat history for a given session ID.
    This ensures chat history persists across reruns for a given session.
    """
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def setup_rag_chain(retriever):
    """
    Sets up the RAG chain, including a history-aware retriever and a
    question-answering chain with a general AI assistant prompt.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

    contextualize_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_system_prompt = (
        "You are a helpful AI assistant. Your task is to answer questions based solely on the provided context. "
        "Keep your answers concise and directly relevant to the question. "
        "if the answer is not in the provided context then just say \"the answer is not in the context provided\", don't provide the wrong answer or other information."
        "Do not make up information or provide answers from outside the given documents."
        "\n\nContext: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_chat_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )
    return conversational_rag_chain


def main():
    st.set_page_config(page_title="Chat With Your PDF(s) 💬", layout="wide")
    st.title("Chat With Your PDF(s) 💬") 

    with st.sidebar:
        st.header("Configuration")
        session_id = st.text_input(
            "Session ID", 
            value="default_session", 
            help="Enter a unique ID for your chat session to maintain history."
        )
        st.write("---") 

        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files", 
            type="pdf", 
            accept_multiple_files=True, 
            help="Upload the documents you want the AI to analyze. Supports multiple files."
        )
        
        pdf_paths = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(TEMP_PDF_DIR, uploaded_file.name)
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                pdf_paths.append(file_path)
            st.sidebar.success(f"Successfully staged {len(pdf_paths)} PDF(s) for processing.")
        else:
            st.sidebar.info("Please upload one or more PDFs to enable the chat functionality.")
        
        st.write("---") 
        if st.button("Clear Chat History", help="Click to clear all messages and start a new conversation."):
            if "store" in st.session_state:
                st.session_state.store[session_id] = ChatMessageHistory()
            st.success("Chat history cleared!")
            st.rerun() 
    vectorstore = None
    if pdf_paths: 
        vectorstore = get_vectorstore_from_pdfs(pdf_paths)
    
    if not vectorstore:
        st.info("Upload PDF documents and wait for the knowledge base to be created to begin interaction.")
        st.stop()
        
    retriever = vectorstore.as_retriever()
    conversational_rag_chain = setup_rag_chain(retriever)

    chat_history = get_chat_history(session_id).messages
    for msg in chat_history:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if user_input := st.chat_input("Ask a question about the documents..."):
        with st.chat_message("human"):
            st.markdown(user_input)
            
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.markdown(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    st.warning("Please try again or re-upload the documents if the issue persists.")


if __name__ == "__main__":
    main()