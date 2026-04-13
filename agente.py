import os
from dotenv import load_dotenv
from rich import print
from langchain_openai import ChatOpenAI

load_dotenv

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))