from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
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
import os
from json import dumps, loads
import time
import tempfile

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

def extract_text_from_pdf(pdf_file: UploadFile):
    text_pages = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    for i, page in enumerate(pages, start=1):
        text_pages.append((page.page_content, i))
    
    os.remove(tmp_file_path)  # Clean up the temporary file

    return text_pages

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def clean_text(text):
    return ' '.join(text.split())

def generate_queries(original_query: str, llm: OpenAI) -> list[str]:
    prompt = PromptTemplate(
        input_variables=["original_query"],
        template="Generate 5 similar queries based on the following question:\n\n{original_query}\n\n1.",
    )
    with get_openai_callback() as cb:
        response = llm(prompt.format(original_query=original_query))
        queries = [original_query] + [q.strip() for q in response.split("\n") if q.strip()]
        print("token count:", cb.total_tokens)
    return queries[:5]

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps({"content": doc.page_content, "metadata": doc.metadata})
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

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

        llm = OpenAI()
        
        # Standard RAG
        start_time = time.time()
        with get_openai_callback() as cb:
            standard_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=global_vectorstore.as_retriever())
            standard_result = standard_qa(original_question)
        standard_time = time.time() - start_time
        standard_tokens = cb.total_tokens
        print("Standard Token :", cb.prompt_tokens, cb.completion_tokens, cb.total_tokens)

        # RAG Fusion
        start_time = time.time()
        queries = generate_queries(original_question, llm)
        # print("Generated queries:", queries)

        all_results = []
        with get_openai_callback() as cb:
            for query in queries:
                docs = global_vectorstore.similarity_search(query)
                all_results.append(docs)

            reranked_results = reciprocal_rank_fusion(all_results)
            
            top_docs = [doc['content'] for doc, _ in reranked_results[:3]]
            context = "\n\n".join(top_docs)
            
            fusion_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="Answer the question based on the following context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
            
            fusion_result = llm(fusion_prompt.format(context=context, question=original_question))

        fusion_time = time.time() - start_time
        fusion_tokens = cb.total_tokens
        print("Fusion Token :", cb.prompt_tokens, cb.completion_tokens, cb.total_tokens)


        response_data = {
            "standard_rag": {
                "answer": standard_result['result'],
                "time": standard_time,
                "tokens": standard_tokens
            },
            "rag_fusion": {
                "answer": fusion_result,
                "time": fusion_time,
                "tokens": fusion_tokens,
                "queries": queries
            }
        }
        # print("Response data:", response_data)
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