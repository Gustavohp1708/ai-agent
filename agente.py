import os
from dotenv import load_dotenv
from rich import print
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))



print(llm.invoke("Olá langsmith"))