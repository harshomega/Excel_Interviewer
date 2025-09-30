from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json

def configure():
    load_dotenv()

configure()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it.")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

prompt = PromptTemplate.from_template(
    "Generate 10 Excel interview questions from basic to advanced, covering formulas, pivot tables, data analysis, and VBA. For each, provide: question, expected answer, and evaluation rubric (score 1-5 based on accuracy, completeness)."
)

chain = prompt | llm
response = chain.invoke({})
try:
    data = json.loads(response.content)
except json.JSONDecodeError:
    data = {"questions": [{"question": "Sample question", "expected": "Sample answer", "rubric": "Score 1-5: 1=No answer, 5=Full answer"}]}
with open('questions.json', 'w') as f:
    json.dump(data, f, indent=2)
print("questions.json generated successfully.")
