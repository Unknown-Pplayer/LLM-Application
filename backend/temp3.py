import os
from dotenv import load_dotenv
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough


class LangchainModel:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize the ChatOpenAI model
        self.chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

        # Initialize the document loader
        self.loader = WebBaseLoader(
            "https://docs.smith.langchain.com/overview")

        # Load data
        self.data = self.loader.load()

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0)

        # Split documents into chunks
        self.all_splits = self.text_splitter.split_documents(self.data)

        # Initialize the vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.all_splits, embedding=OpenAIEmbeddings())

        # Set up the retriever
        self.retriever = self.vectorstore.as_retriever(k=4)

        # Set up the question answering prompt
        self.question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Answer the user's questions based on the below context. 
                    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

                    <context>
                    {context}
                    </context>
                    """,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Create the document chain
        self.document_chain = create_stuff_documents_chain(
            self.chat, self.question_answering_prompt)

        # Create the retrieval chain
        self.retrieval_chain = self.create_retrieval_chain()

    def create_retrieval_chain(self):
        # Define the function to parse the retriever input
        def parse_retriever_input(params: Dict):
            return params["messages"][-1].content

        # Create and return the retrieval chain
        return RunnablePassthrough.assign(
            context=parse_retriever_input | self.retriever,
        ).assign(
            answer=self.document_chain,
        )

    def run_retrieval_chain(self, messages):
        # Run the retrieval chain with the provided messages
        response = self.retrieval_chain.invoke({"messages": messages})
        return response


if __name__ == "__main__":
    # Create an instance of the LangchainModel class
    model = LangchainModel()

    # Define the messages for the query
    messages = [
        HumanMessage(content="Can LangSmith help test my LLM applications?")
    ]

    # Run the retrieval chain and print the response
    response = model.run_retrieval_chain(messages)
    print("Response:", response)
