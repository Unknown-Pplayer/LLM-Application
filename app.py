from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
import tempfile
import os


# Takes a PDF file and extracts the text content


def extract_text_from_pdf(pdf_file):
    text = ""
    # Save the uploaded PDF file object to a temporary location and load it by PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    for page in pages:
        text += page.page_content
    os.remove(tmp_file_path)

    return text

# Splits the text into chunks of 1000 characters with 150 characters overlap


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
    text_chunks = text_splitter.split_text(text)

    return text_chunks

# Creates a vector store from the text chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

# Creates a conversation chain from the vector store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

# Handles the user input and displays the conversation history and the AI response


def handle_userinput(user_input):
    if st.session_state.conversation is None:
        st.warning(
            "Please upload a PDF file and click Submit to start a conversation.")
        return

    response = st.session_state.conversation({'question': user_input})
    chat_history = response['chat_history']
    answer = response['answer']

    # Format the chat history
    formatted_history = ""
    for i, message in enumerate(chat_history):
        role = "User" if i % 2 == 0 else "AI"
        formatted_history += f"\n{role}: {message.content}\n"

    # Display the formatted response
    st.write(formatted_history)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Assistant", page_icon="📄")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("PDF Assistant")
    user_input = st.text_input("Ask a question about the document")

    if user_input:
        handle_userinput(user_input)

    with st.sidebar:
        st.subheader("Upload Your File")
        pdf_doc = st.file_uploader("Upload a PDF file")
        if st.button("Submit"):
            if pdf_doc is not None:
                with st.spinner("Processing"):
                    # Extract text from the PDF
                    text = extract_text_from_pdf(pdf_doc)
                    # Split the text into chunks
                    text_chunks = get_text_chunks(text)
                    # Create a vector store from the text chunks
                    vectorstore = get_vectorstore(text_chunks)
                    # Create a conversation chain from the vector store
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
            else:
                st.warning("Please upload a PDF file before clicking Submit.")


if __name__ == "__main__":
    main()
