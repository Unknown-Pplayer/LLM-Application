import openai
import os
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ambQuery(BaseModel):
    input: str

global_ambiguous = []
global_text_string = ""

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content
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

@app.post("/api/ambiguous")
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
