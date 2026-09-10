from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver


# ---------------- DATABASE ----------------

db = SQLDatabase.from_uri("sqlite:///mytask.db")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK(
            status IN ('pending', 'in progress', 'completed')
        ) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

print("DB table executed")


# ---------------- LLM ----------------

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500
)


# ---------------- TOOLS ----------------

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)

tools = [
    tool
    for tool in toolkit.get_tools()
    if tool.name == "sql_db_query"
]


# ---------------- MEMORY ----------------

memory = InMemorySaver()


# ---------------- SYSTEM PROMPT ----------------

system = """
You are a task management assistant that interacts with a SQLite
database containing a table called `tasks`.

Table schema:

tasks(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,
    created_at TIMESTAMP
)

Allowed status values:
- pending
- in progress
- completed

Rules:

1. SELECT queries must return a maximum of 10 results.
2. For task lists, use ORDER BY created_at DESC.
3. After INSERT, UPDATE, or DELETE, verify the operation with a SELECT query.
4. Only work with the tasks table.
5. Never delete or update a task unless the user explicitly asks.
6. Give concise responses.
"""


# ---------------- AGENT ----------------

@st.cache_resource
def get_agent():

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system
    )

    return agent


agent = get_agent()


# ---------------- STREAMLIT UI ----------------

st.subheader("TaskBot - Manage Your Tasks")


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    st.chat_message(
        message["role"]
    ).markdown(
        message["content"]
    )


# User input

prompt = st.chat_input("Ask me to manage your tasks")


if prompt:

    # Display user message
    st.chat_message(
        "user",
        avatar="👤"
    ).markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


    # Agent response

    with st.chat_message("assistant"):

        with st.spinner("Processing..."):

            res = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": "1"
                    }
                }
            )

        result = res["messages"][-1].content

        st.markdown(result)


    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })