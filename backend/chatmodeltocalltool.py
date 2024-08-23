from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory


class Question(BaseModel):
    question: str


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        description="How funny the joke is, from 1 to 10")


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(
        description="A conversational response to the user's query")


class AddEvent(BaseModel):
    """Add an event to the calendar with enhanced recognition and details."""
    event_name: str = Field(description="The name of the event")
    event_date: Optional[datetime] = Field(
        description="The date and time of the event", default=None)
    event_end_date: Optional[datetime] = Field(
        description="The end date and time of the event, if applicable", default=None)
    event_location: Optional[str] = Field(
        description="The location of the event", default=None)
    event_description: str = Field(
        description="A description of the event")
    attendees: Optional[List[str]] = Field(description="List of attendees")
    is_recurring: bool = Field(description="Whether the event is recurring")
    recurrence_pattern: Optional[str] = Field(
        description="The pattern of recurrence if the event is recurring")
    tags: Optional[List[str]] = Field(
        description="Tags or categories for the event")
    rules: Optional[List[str]] = Field(
        description="Rules or guidelines for the event")
    countries: Optional[List[str]] = Field(
        description="Countries involved or relevant to the event")


class SendingEmail(BaseModel):
    """Send an email to a recipient."""
    recipient: str = Field(description="The recipient of the email")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body of the email")


class Summarize(BaseModel):
    """Summarize a given text into json format."""
    summary: dict = Field(description="The summary of the text in JSON format")


class Output(BaseModel):
    """Possible outputs from the conversation."""
    result: Union[Joke, ConversationalResponse,
                  AddEvent, SendingEmail, Summarize]


router = APIRouter()


# @router.post("/chatModelCallTool")
# async def structure_output(query: Question):
#     llm = ChatOpenAI()
#     query = query.question
#     llm_with_tools = llm.bind_tools(
#         [Joke, ConversationalResponse, AddEvent, SendingEmail, Summarize])
#     return llm_with_tools.invoke(query)


# other way using agent


def joke_tool(query: str) -> str:
    llm = ChatOpenAI()
    joke_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["query"],
        template="""Create a joke based on this query: {query}
        in the following format
        setup: The setup of the joke
        punchline: The punchline to the joke
        rating: How funny the joke is, from 1 to 10"""
    ))
    return joke_chain.run(query)


def add_event_tool(query: str) -> str:
    llm = ChatOpenAI()
    event_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["query"],
        template="""
        You are an AI assistant tasked with adding events to a calendar. Based on the user's request, create an AddEvent object with the information provided. If any required information is missing, ask the user for it.

        User request: {query}

        Required fields:
        - event_name: The name of the event (required)
        - event_date: The date and time of the event (required, ask if not provided)
        - event_location: The location of the event (ask if not provided)
        - event_description: A description of the event (can be brief if not provided)

        Optional fields (only fill if information is provided):
        - event_end_date: The end date and time of the event
        - attendees: List of attendees
        - is_recurring: Whether the event is recurring (default to False if not mentioned)
        - recurrence_pattern: The pattern of recurrence if the event is recurring
        - tags: Tags or categories for the event
        - rules: Rules or guidelines for the event
        - delivery_date: Delivery date for relevant events
        - countries: Countries involved or relevant to the event

        If any required information is missing, respond with a question to get that information. Otherwise, create the AddEvent object and return it as a string representation.

        Example response if information is missing:
        "To add this event to your calendar, I need some more information. What date and time is the Google event?"

        Example response if all required information is provided:
        AddEvent(
            event_name="Google Event",
            event_date="2023-07-15T14:00:00",
            event_location="Online",
            event_description="Join the Google event to learn about new features and updates from Google.",
            is_recurring=False
        )
        """
    ))
    return event_chain.run(query)


def send_email_tool(query: str) -> str:
    llm = ChatOpenAI()
    email_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["query"],
        template="""Compose an email based on this request: {query}"""
    ))
    return email_chain.run(query)


def summarize_tool(query: str) -> str:
    llm = ChatOpenAI()
    summary_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["query"],
        template="""Summarize the following text into the most important points: {query}

        Return the summary as a JSON object with numbered points, like this:
        {{
            "point 1": "First important point",
            "point 2": "Second important point",
            "point 3": "Third important point",
            ...
        }}

        Ensure that the output is a valid JSON object."""
    ))
    summary = summary_chain.run(query)
    try:
        json_summary = json.loads(summary)
        return json.dumps(json_summary, indent=2)
    except json.JSONDecodeError:
        lines = summary.split('\n')
        json_summary = {f"point {i+1}": line.strip()
                        for i, line in enumerate(lines) if line.strip()}
        return json.dumps(json_summary, indent=2)


tools = [
    Tool(
        name="Joke",
        func=joke_tool,
        description="Useful for when you need to tell a joke"
    ),
    Tool(
        name="AddEvent",
        func=add_event_tool,
        description="Use this to add an event to the calendar. Required information: event name, date, time, and location."
    ),
    Tool(
        name="SendEmail",
        func=send_email_tool,
        description="Useful for when you need to send an email"
    ),
    Tool(
        name="Summarize",
        func=summarize_tool,
        description="Useful for when you need to summarize a text into a JSON format with numbered points"
    ),
    Tool(
        name="AskMoreInfo",
        func=lambda query: "I need more information to proceed with this task.",
        description="Useful for prompting the user for more information"
    )
]

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

llm = ChatOpenAI()

agent = initialize_agent(
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True,
)


@router.post("/chatModelCallTool")
async def structure_output(query: Question):
    query_str = query.question
    response = agent(query_str)
    return response
