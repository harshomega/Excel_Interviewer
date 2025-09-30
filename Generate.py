from langchain_openai import ChatOpenAI  # Or equivalent for Grok
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o", api_key="")
prompt = PromptTemplate.from_template(
    "Generate 10 Excel interview questions from basic to advanced, covering formulas, pivot tables, data analysis, and VBA. For each, provide: question, expected answer, and evaluation rubric (score 1-5 based on accuracy, completeness)."
)
chain = prompt | llm
response = chain.invoke({})
print(response.content)  # Save to a JSON file, e.g., questions.json