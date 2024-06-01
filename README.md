# PDF Assistant

## Overview

PDF Assistant is an interactive web application built with Streamlit that allows users to upload a PDF document, ask questions about its content, and receive answers using a conversational AI model. The tool leverages LangChain, OpenAI's GPT-3.5, and FAISS for text embedding and retrieval.

## Installation

1. **Clone the Repository:**

```bash
https://github.com/Unknown-Pplayer/LLM-Application.git
cd LLM-Application
```

2. **Create a Virtual Environment:**

```bash
python -m venv venv
source .venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables:**

You can find your OpenAI API Key https://platform.openai.com/api-keys

Create a .env File at the porject root directory and add the following

```bash
OPENAI_API_KEY=YOUR_KEY
```

5. **Run the Application**

```bash
streamlit run app.py
```
