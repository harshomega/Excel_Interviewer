import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import json

# Load and validate bootstrapped questions
questions = []
try:
    with open('question.json', 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            questions = data
        else:
            questions = data.get('questions', [])
        # Validate each question object
        for q in questions:
            if not all(key in q for key in ['question', 'expected', 'rubric']):
                st.error("Invalid question format in questions.json. Missing 'question', 'expected', or 'rubric'.")
                questions = []
                break
except FileNotFoundError:
    st.error("questions.json not found. Please generate it using Generate.py.")
except json.JSONDecodeError:
    st.error("questions.json contains invalid JSON. Please regenerate it.")

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
            if user_input and st.session_state.question_index > 0:
                prev_q = questions[st.session_state.question_index - 1]
                eval_result = eval_chain.invoke({"answer": user_input, "expected": prev_q['expected'], "rubric": prev_q['rubric']}).content
                try:
                    score_part = eval_result.split("Score:")[1].strip()
                    score = int(''.join(filter(str.isdigit, score_part.split()[0])))
                    st.session_state.scores.append(score)
                except (IndexError, ValueError) as e:
                    st.error(f"Error parsing score: {e}. Defaulting to 0.")
                    st.session_state.scores.append(0)
                st.session_state.history.append({"q": prev_q['question'], "a": user_input, "feedback": eval_result})
                st.write(f"Feedback: {eval_result}")

            st.session_state.question_index += 1
        else:
            st.session_state.stage = 'summary'

    if st.session_state.stage == 'summary':
        history = "\n".join([f"Q: {h['q']}\nA: {h['a']}\nFeedback: {h['feedback']}" for h in st.session_state.history])
        summary = summary_chain.invoke({"history": history}).content
        overall_score = sum(st.session_state.scores) / len(st.session_state.scores) if st.session_state.scores else 0
        st.write(f"Interview Complete!\nOverall Score: {overall_score}/5\n{summary}")
