
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI
import os
import json

pdf_path = Path(__file__).parent / "BTP_Report.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=""  # Replace with your actual Gemini API key
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="learning_langchain2",
    embedding=embedder
)

vector_store.add_documents(documents = split_docs)


retriver = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name="learning_langchain2",
    embedding=embedder
)
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

while True:
    user_query = input('> ')
    relevant_chunks = retriver.similarity_search(
    query = user_query
    )

    SYSTEM_PROMPT = f"""
You are an helpful AI Assistant who responds base of the avaiable context.

Context: {relevant_chunks}
 You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant context.
    Wait for the observation and based on the observation from the context resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next intput 
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "input": "The input query of the user",
    }}

    Example:
    User Query:  What is the content of this pdf?
    Output: {{ "step": "plan", "content": "The user is interested in the content of this pdf" }}
    Output: {{ "step": "plan", "content": "From the available context I should read Context: {relevant_chunks}" }}
    Output: {{ "step": "output", "content": "The pdf describes about Artificial Intelligence" }}

"""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_query},
    ]

    messages.append({ 'role': 'user', 'content': user_query })

    while True:
        response = client.chat.completions.create(
            model='gemini-2.0-flash',
            response_format={"type": "json_object"},
            messages=messages,
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ 'role': 'assistant', 'content': json.dumps(parsed_output) })

        if parsed_output['step'] == 'plan':
            print(f"🧠: {parsed_output.get('content')}")
            continue

        if parsed_output['step'] == 'output':
            print(f"🤖: {parsed_output.get('content')}")
            break