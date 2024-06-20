import os
import json
import getpass
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserQuery(BaseModel):
    input_text: str


class NvidiaLangChainClient:
    def __init__(self, model: str = None):
        """
        Initialize the client with the specific model.
        """
        nvapi_key = os.getenv("NVIDIA_API_KEY")
        model = model or os.getenv("NVIDIA_MODEL")
        if not nvapi_key or not nvapi_key.startswith("nvapi-"):
            raise ValueError(
                "Invalid or missing NVIDIA_API_KEY in the environment.")
        if not model:
            raise ValueError(
                "Model is not specified and no default model found in environment variables.")

        self.llm = ChatNVIDIA(model=model)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI language model. Please respond to the user's queries.",
                ),
                ("user", "{input}"),
            ]
        )

    def query(self, input_text: str):
        """
        Send a query to the NVIDIA model and return the response.
        """
        chain = self.prompt_template | ChatNVIDIA(
            model=self.llm.model) | StrOutputParser()
        response = ""
        for txt in chain.stream({"input": input_text}):
            response += txt
        return response

    def handle_errors(self, function_to_wrap):
        """
        functions for error handling.
        """
        def wrapper(*args, **kwargs):
            try:
                return function_to_wrap(*args, **kwargs)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return "An error occurred while processing your request."
        return wrapper


def save_to_json(filename: str, data: dict):
    file_path = f"{filename}.json"
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    else:
        with open(file_path, "r") as file:
            existing_data = json.load(file)
            should_update = False
            for key, value in data.items():
                if key not in existing_data:
                    existing_data[key] = value
                    should_update = True
            if should_update:
                with open(file_path, "w") as file:
                    json.dump(existing_data, file, indent=4)
            else:
                print(
                    "File exists and contains the keys to be added. No changes were made.")


nvidia_client = NvidiaLangChainClient(model=os.getenv("NVIDIA_MODEL"))


@app.post("/query")
def process_query(query: UserQuery):
    response = nvidia_client.query(query.input_text)
    return {"response": response}


@app.post("/scraping")
def process_web(query: UserQuery):
    page = requests.get(query.input_text)
    soup = BeautifulSoup(page.content, 'html.parser')
    title = soup.title.text
    title = title.replace("-", "").replace(" ", "_")

    article_text = " ".join([p.text for p in soup.find_all('p')])
    prompt = """You are an expert IT instructor. Now I will give you a complete article,
        Read and analyze the article, Then I want you to create a step-by-step guide on \
        how to complete the task described in the article. 

        ## Article Content:
        """ + article_text

    filename = title[:10]
    response = nvidia_client.query(prompt)
    steps = re.findall(r"(Step \d+:.*?)(?=Step \d+:|$)", response, re.DOTALL)
    steps_list = [step.strip() for step in steps]
    save_to_json(filename, {"instructions": steps_list})
    # print(response)
    return {"instructions": response}


@app.post("/summary")
def process_summary(query: UserQuery):
    page = requests.get(query.input_text)
    soup = BeautifulSoup(page.content, 'html.parser')
    title = soup.title.text
    title = title.replace("-", "").replace(" ", "_")

    article_text = " ".join([p.text for p in soup.find_all('p')])
    prompt = """You are an AI language model. Please summarize the following text \
    in no more than one paragraph.

    ## Article Content:
    """ + article_text

    filename = title[:10]
    response = nvidia_client.query(prompt)
    # print(response)
    save_to_json(filename, {"summary": response})
    return {"summary": response}
