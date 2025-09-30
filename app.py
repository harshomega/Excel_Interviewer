import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import json

# Load bootstrapped questions
with open('question.json', 'r') as f:
    questions = json.load(f)


# Access the API key from secrets
llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"])
# Memory for state management
memory = ConversationBufferMemory()

# Prompts
intro_prompt = "Introduce yourself as an AI Excel Interviewer, explain the process (5-10 questions, evaluation at end)."
question_prompt = PromptTemplate.from_template("Based on previous conversation: {history}\nAsk the next question: {question}")
eval_prompt = PromptTemplate.from_template(
    "Evaluate this answer: {answer}\nAgainst expected: {expected}\nRubric: {rubric}\nScore 1-5 and give feedback."
)
summary_prompt = PromptTemplate.from_template(
    "Summarize performance: {history}\nInclude overall score, strengths, weaknesses."
)

# Chains
question_chain = question_prompt | llm
eval_chain = eval_prompt | llm
summary_chain = summary_prompt | llm

# Streamlit UI
st.title("AI Excel Mock Interviewer")

if 'stage' not in st.session_state:
    st.session_state.stage = 'intro'
    st.session_state.question_index = 0
    st.session_state.scores = []
    st.session_state.history = []

user_input = st.text_input("Your response:")

if st.button("Submit"):
    if st.session_state.stage == 'intro':
        intro = llm.invoke(intro_prompt).content
        st.write(intro)
        st.session_state.stage = 'question'
        memory.save_context({"input": intro}, {"output": ""})

    elif st.session_state.stage == 'question':
        if st.session_state.question_index < len(questions):
            q = questions[st.session_state.question_index]
            history = memory.load_memory_variables({})['history']
            next_q = question_chain.invoke({"history": history, "question": q['question']}).content
            st.write(next_q)
            memory.save_context({"input": next_q}, {"output": user_input})

            # Evaluate previous answer if not first
            if user_input:
                eval_result = eval_chain.invoke({"answer": user_input, "expected": q['expected'], "rubric": q['rubric']}).content
                score = int(eval_result.split("Score:")[1].strip()[0])  # Parse score
                st.session_state.scores.append(score)
                st.session_state.history.append({"q": q['question'], "a": user_input, "feedback": eval_result})
                st.write(f"Feedback: {eval_result}")

            st.session_state.question_index += 1
        else:
            st.session_state.stage = 'summary'

    if st.session_state.stage == 'summary':
        history = "\n".join([f"Q: {h['q']}\nA: {h['a']}\nFeedback: {h['feedback']}" for h in st.session_state.history])
        summary = summary_chain.invoke({"history": history}).content
        overall_score = sum(st.session_state.scores) / len(st.session_state.scores)
        st.write(f"Interview Complete!\nOverall Score: {overall_score}/5\n{summary}")
