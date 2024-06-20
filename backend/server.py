import openai
import os
import re
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function


class Question(BaseModel):
    question: str


class ambQuery(BaseModel):
    input: str


class GptQuery(BaseModel):
    input: str
    temperature: float


class TextQuery(BaseModel):
    input: str
    option: str
    language: str
    tone: str


class addInfo(BaseModel):
    input: str
    additionalInfo: str


class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")


# Define global variables
global_vectorstore = None
global_conversation_chain = None
global_qa_chain = None
llm = OpenAI()
global_chat_chain = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
global_ambiguous = []
global_text_string = ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(pdf_file):
    text_pages = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    for i, page in enumerate(pages, start=1):
        text_pages.append((page.page_content, i))
    os.remove(tmp_file_path)

    return text_pages


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


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


def clean_text(text):
    cleaned_text = text.replace('\xa0', ' ')
    return cleaned_text


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_summary(text):
    prompt = f"""
    Summarize the text delimited by triple backticks \
    as a list bullet points in the following format:

    - <Possible Title>
    - <The most important point in the text, around 2 sentence.>
    - <The most interesting action in the text.>
    - <A list of most important characters in JSON format.>

    ```{text}```
    """
    return get_completion(prompt)


def get_translation(text, language):
    prompt = f"""
    Translate the text delimited by triple backticks text to {language}:
    Your response should strictly adhere to the translation of the
    original user message.
    ```{text}```
    """
    return get_completion(prompt)


def find_placeholders(text):
    """
    Finds all placeholders within the given text that are enclosed in square brackets,
    maintaining their original order in the text.

    Args:
    - text (str): The text to search for placeholders.

    Returns:
    - list: A list of placeholders found within the text, in their original order.
    """
    # Regex to find words within square brackets
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, text)

    return matches


def get_transformation(text, tone):
    prompt = f"""
    Transform the text delimited by triple backticks
    in to a email with {tone} tone in the following format:
    - Subject: <Subject of the email>
    - Body: <Body of the email, include a greeting at the beginning.>
    ```{text}```
    """
    return get_completion(prompt)


def get_correction(text):
    prompt = f"""
    Proofread and correct the grammar of the text
    {text}
    """
    return get_completion(prompt)


def get_addition(text, additional_text):
    prompt = f"""
    Add the additional information to the text by replacing the placeholders
    with the apporpriate {additional_text} in exact order without square brackets and single quotation mark
    each placeholder should be replaced with one {additional_text} that is relevant to the context.
    return the final text after all the replacement.

    {additional_text}
    {text}
    """
    return get_completion(prompt)


def get_sentiment(text):
    prompt = f"""
    Identify the following items from the text:
    - Sentiment (positive or negative)
    - Is the reviewer expressing anger? (true or false)
    - 5 key topics discussed in the text

    Format your response as a JSON object with \
    "Sentiment", "Anger" and "topics" as the keys.
    If the information isn't present, use "unknown" \
    as the value.
    Make your response as short as possible.
    Format the Anger value as a boolean.
    {text}
    """
    return get_completion(prompt)


def determine_negative(sentiment):
    prompt = f"""
    Determine if the sentiment is negative or not.
    return true if the sentiment is negative and false if it is not.
    {sentiment}
    """
    return get_completion(prompt)


def generate_ai_response(text):
    prompt = f"""
    You are a customer service AI assistant.
    Your task is to generate a response based on the text.
    If the sentiment is negative, apologize and suggest that \
    they can reach out to customer service.
    Make sure to use specific details from the review.
    Write in a concise and professional tone.
    Sign the email as `AI customer agent`.
    Customer text: ```{text}```
    """
    return get_completion(prompt)


def get_intention(entire_story, current_intention):
    prompt = f"""
    Determine the intention of the text in the following format:
    {{
        "entire intention": "The intention of the text",
        "current intention": "The intention of the current sentence"
    }}

    example:
    text: "code is poetry"
    intention: "To convey the idea that code is like poetry"
    text: "every line tells a story"
    intention: "To convey the idea that every line of code tells a story"
    text: "error could be anywhere"
    intention: "To convey the idea that errors could be anywhere in the code"
    text: "stopping the program execution"
    intention: "To convey the idea that error code could stop the program execution"

    {entire_story, current_intention}
    """
    return get_completion(prompt)


def get_break_down(text):
    prompt = f"""
    break down the text into the following format:
    if any of the following parts are not present in the text, use "unknown" as the value.
    {{
        "sentence": "The sentence",
        "structure": {{
            "interrogative": "The interrogative in the sentence",
            "subject": "The subject in the sentence",
            "object": "The object in the sentence",
            "auxiliary_verb": "The auxiliary verb in the sentence",
            "main_verb": "The main verb in the sentence",
            "prepositional_phrase": "in the month of May"ß
        }}
    }}
    only return the key value pairs that are not unknown.
    {text}
    """
    return get_completion(prompt)


@ app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_conversation_chain
    global global_qa_chain

    print("Received file:", file.filename)
    file_name = file.filename
    persist_directory = f'./db/{file_name}'

    if os.path.exists(persist_directory):
        print(f"Loading existing embeddings for file: {file.filename}")
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=OpenAIEmbeddings())
    else:
        print(f"Processing new file: {file.filename}")
        text_pages = extract_text_from_pdf(file.file)

        # Create Document objects from text chunks
        documents = []
        for text, page_num in text_pages:
            # print("page_num:", page_num)
            # print("text:", text)
            chunks = get_text_chunks(text)
            documents.extend([Document(page_content=chunk, metadata={'page_number': page_num})
                              for chunk in chunks])

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents, embedding=embeddings, persist_directory=persist_directory)

    retriever = vectordb.as_retriever()
    global_qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                                  chain_type="stuff",
                                                  retriever=retriever,
                                                  return_source_documents=True)

    global_conversation_chain = get_conversation_chain(retriever)
    return {"status": "success"}


@ app.post("/ask")
async def ask_question(question_data: Question):
    global_conversation_chain
    global_qa_chain
    try:
        if global_conversation_chain is None:
            return {"status": "error", "message": "No PDF has been processed yet."}

        question = question_data.question
        print("Received question:", question)

        input_tokens = 0
        output_tokens = 0

        with get_openai_callback() as cb:
            llm_response = global_qa_chain(question)
            # print("Prompt Tokens:", cb.prompt_tokens)
            # print("Completion Tokens:", cb.completion_tokens)
            input_tokens = cb.prompt_tokens
            output_tokens = cb.completion_tokens

        source_documents_info = []
        for doc in llm_response['source_documents']:
            clean_content = clean_text(doc.page_content)
            page_number = doc.metadata.get('page_number')
            source_documents_info.append({
                "quote": clean_content,
                "page": page_number
            })
        # print("Source documents with page numbers:", source_documents_info)

        answer = global_conversation_chain.invoke(question)

        response_data = {
            "answer": answer,
            "source_documents_info": source_documents_info,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

        # print("Response data:", response_data)
        return response_data
    except Exception as e:
        print("Error processing question:", e)
        return {"status": "error", "message": str(e)}


@ app.get("/pdfs")
async def get_pdfs():
    pdfs = []
    for file in os.listdir('./db'):
        if file.endswith(".pdf"):
            pdfs.append(file)
    return pdfs


@ app.post("/api/gpt")
async def gpt_query(query: GptQuery):
    global global_chat_chain
    input_text = query.input
    temperature = query.temperature
    print("Received GPT query:", input_text, temperature)

    try:
        response = global_chat_chain.predict(input=input_text)
        print("GPT response:", response)
        return {"question": input_text, "answer": response}
    except Exception as e:
        print("Error processing GPT query:", e)


@ app.post("/api/bot")
async def process_text(query: TextQuery):
    text = query.input
    option = query.option

    if option == '1':
        language = query.language
        return get_translation(text, language)
    elif option == '2':
        return get_summary(text)
    elif option == '3':
        tone = query.tone
        print("Tone:", tone)
        transformation_response = get_transformation(text, tone)
        print("Transformation response:", transformation_response)
        placeholders = find_placeholders(transformation_response)
        print("Placeholders:", placeholders)
        if placeholders:
            return {"message": "Need more information", "placeholders": placeholders, "transformation": transformation_response}
        else:
            return {"transformation": transformation_response}
    elif option == '4':
        return get_correction(text)
    elif option == '5':
        sentiment = get_sentiment(text)
        negative = determine_negative(sentiment)
        if negative:
            return {"sentiment": sentiment, "response": generate_ai_response(text)}
        else:
            return sentiment


@ app.post("/api/info")
async def add_info(query: addInfo):
    input_text = query.input
    # additional_text = query.additionalInfo
    additional_text = query.additionalInfo.split(', ')
    print("additional_text:", additional_text)
    response = get_addition(input_text, additional_text)
    # print("Input text:", input_text)
    print("Response:", response)
    return get_addition(input_text, additional_text)


@ app.post("/api/ambiguous")
async def ambiguous_text(query: ambQuery):
    global global_ambiguous
    global global_text_string
    input_text = query.input

    if input_text == "reset":
        global_ambiguous = []
        global_text_string = ""
        return

    global_text_string += (input_text + " ")
    intention = get_intention(global_text_string, input_text)
    breakdown = get_break_down(input_text)
    global_ambiguous.extend([
    {"type": "user", "content": input_text},
    {"type": "ai", "content": f"Breakdown: {breakdown + intention}"}
    ])

    return global_ambiguous
