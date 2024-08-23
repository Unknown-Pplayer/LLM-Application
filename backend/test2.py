from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import os
from json import dumps, loads
import tempfile
import sqlite3
import time
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, List
from typing import Union
from datetime import datetime
from langchain.chains import ConversationChain

from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
import json
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from chatmodeltocalltool import router


load_dotenv()

# Set up SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

app = FastAPI()


class Question(BaseModel):
    question: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_qa_chain = None
global_conversation_chain = None
global_vectorstore = None

database_path = ".langchain.db"
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS cache (
                    question TEXT PRIMARY KEY,
                    answer TEXT NOT NULL
                  )""")
conn.commit()


def extract_text_from_pdf(pdf_file: UploadFile):
    text_pages = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    for i, page in enumerate(pages, start=1):
        text_pages.append((page.page_content, i))

    os.remove(tmp_file_path)

    return text_pages


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore,
        memory=memory,
    )
    return conversation_chain


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_vectorstore, global_conversation_chain, global_qa_chain

    try:
        print("Received file:", file.filename)
        file_name = file.filename
        persist_directory = f'./db/{file_name}'

        if os.path.exists(persist_directory):
            print(f"Loading existing embeddings for file: {file.filename}")
            global_vectorstore = Chroma(persist_directory=persist_directory,
                                        embedding_function=OpenAIEmbeddings())
        else:
            print(f"Processing new file: {file.filename}")
            text_pages = extract_text_from_pdf(file)

            documents = []
            for text, page_num in text_pages:
                chunks = get_text_chunks(text)
                documents.extend([Document(page_content=chunk, metadata={'page_number': page_num})
                                  for chunk in chunks])

            embeddings = OpenAIEmbeddings()
            global_vectorstore = Chroma.from_documents(
                documents, embedding=embeddings, persist_directory=persist_directory)

        retriever = global_vectorstore.as_retriever()
        global_qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                                      chain_type="stuff",
                                                      retriever=retriever,
                                                      return_source_documents=True)

        global_conversation_chain = get_conversation_chain(retriever)

        return {"status": "success"}
    except Exception as e:
        print(f"Error in upload_pdf: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/ask")
async def ask_question(question_data: Question):
    global global_vectorstore

    try:
        if global_vectorstore is None:
            return {"status": "error", "message": "No PDF has been processed yet."}

        original_question = question_data.question
        print("Received question:", original_question)

        # Before interacting with the LLM or checking cache, clean up the question.
        cleaned_prompt = original_question.strip()  # Adjust cleansing as needed

        # Check cache first
        start_time = time.time()
        with get_openai_callback() as cb:
            cursor.execute(
                "SELECT answer FROM cache WHERE question = ?", (cleaned_prompt,))
            result = cursor.fetchone()

        if result:
            print("Cache hit")
            return {"answer": result[0],
                    "time": time.time() - start_time,
                    "token": cb.total_tokens
                    }
        else:
            llm = OpenAI()
            start_time = time.time()
            with get_openai_callback() as cb:
                standard_qa = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=global_vectorstore.as_retriever())
                standard_result = standard_qa(original_question)
            standard_time = time.time() - start_time
            standard_tokens = cb.total_tokens
            # Update cache with only the cleaned question
            cursor.execute("INSERT INTO cache (question, answer) VALUES (?, ?)",
                           (cleaned_prompt, standard_result['result']))
            conn.commit()
        standard_time = time.time() - start_time

        response_data = {
            "standard_rag": {
                "answer": standard_result['result'],
                "time": standard_time,
                "tokens": standard_tokens
            }
        }
        return response_data
    except Exception as e:
        print("Error processing question:", e)
        return {"status": "error", "message": str(e)}


@app.get("/pdfs")
async def get_pdfs():
    pdfs = []
    for file in os.listdir('./db'):
        if file.endswith(".pdf"):
            pdfs.append(file)
    return pdfs


@app.post("/changeQuery")
async def change_query(question_data: Question):
    original_question = question_data.question.strip()  # Cleanse as needed
    temp_question = "temp question text here"  # Example of new text to use

    pattern = f"%{original_question}%"  # Use cleaned question for matching
    updated_prompt = "\n\n" + temp_question  # Prepare update as needed

    cursor.execute("UPDATE cache SET prompt = ? WHERE prompt LIKE ?",
                   (updated_prompt, pattern))
    conn.commit()

    return {"message": "Database updated successfully."}


@app.post("/deleteQuery")
async def delete_query(question_data: Question):
    original_question = question_data.question
    cursor.execute("DELETE FROM cache WHERE question = ?",
                   (original_question,))
    conn.commit()
    return {"message": "Query deleted successfully."}


class QnA(BaseModel):
    question: str
    answer: str


@app.post("/prefeed/")
async def prefeed_qna(qna: QnA):
    cursor.execute(
        "INSERT OR REPLACE INTO cache (question, answer) VALUES (?, ?)", (qna.question, qna.answer))
    conn.commit()
    return {"message": "Question and Answer pre-fed successfully."}


def fetch_qa_pairs_from_db(question):
    conn = sqlite3.connect(".langchain.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM cache")
    rows = cursor.fetchall()
    conn.close()
    return rows


def create_documents_from_qa_pairs(qa_pairs):
    docs = []
    for question, answer in qa_pairs:
        doc = Document(page_content=question, metadata={"answer": answer})
        docs.append(doc)
    return docs


def generate_queries(original_query: str, llm: OpenAI) -> list[str]:
    prompt = PromptTemplate(
        input_variables=["original_query"],
        template="Generate 5 similar queries based on the following question:\n\n{original_query}\n\n1.",
    )
    with get_openai_callback() as cb:
        response = llm(prompt.format(original_query=original_query))
        queries = [original_query] + [q.strip()
                                      for q in response.split("\n") if q.strip()]
        print("token count:", cb.total_tokens)
    return queries[:5]


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps({"content": doc.page_content,
                            "metadata": doc.metadata})
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


@app.post("/test")
async def test_api(question_data: Question):
    llm = OpenAI()
    question = question_data.question
    qa_pair = fetch_qa_pairs_from_db(question)
    docs = create_documents_from_qa_pairs(qa_pair)
    with get_openai_callback() as cb:
        vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever)
        response = qa_chain(question)
        print("Rag Token: ", cb.total_tokens)
        # return {"response": response}

    # Rag fusion
    queries = generate_queries(question, OpenAI())
    all_results = []
    with get_openai_callback() as cb:
        for query in queries:
            docs = vectorstore.similarity_search(query)
            all_results.append(docs)

        reranked_results = reciprocal_rank_fusion(all_results)
        # top_docs = [doc['content'] for doc, _ in reranked_results[:3]]
        # context = "\n\n".join(top_docs)
        top_docs_and_answers = [
            (doc['content'], doc['metadata']['answer']) for doc, _ in reranked_results[:3]
        ]
        context = "\n\n".join([
            f"Question: {content}\nAnswer: {answer}" for content, answer in top_docs_and_answers
        ])
        print("context: ", context)
        fusion_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer only the question and only on the following context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        fusion_result = OpenAI()(fusion_prompt.format(context=context, question=question))
        print("Fusion Token: ", cb.total_tokens)
        return ({"response": response, "fusion_result": fusion_result})
    # unique_responses = {}
    # for doc in response:
    #     # Adjusted data access pattern:
    #     question_content = doc.metadata.get('question')  # Assuming 'question' is stored in metadata
    #     if question_content not in unique_responses:
    #         unique_responses[question_content] = doc.metadata.get('answer')

    # # Convert back to a list but only include answers in this example
    # final_response = list(unique_responses.values())

    # return {"response": final_response}

# conclusion it checks function name first, if no function name matches it check docstring.

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
