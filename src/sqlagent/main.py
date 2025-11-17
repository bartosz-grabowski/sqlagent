"""Main module for the SQLAgent FastAPI application."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from sqlagent.model_utils import is_model_available


# MySQL connection details
DB_USER = "root"
DB_PASSWORD_FILE = os.environ["MYSQL_ROOT_PASSWORD_FILE"]
DB_HOST = "db"
DB_NAME = "sqlagent_db"
# OLLAMA MODEL
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")


app = FastAPI()


@app.get("/")
def main(q: str) -> dict:
    """Main endpoint to process user queries."""
    if not is_model_available(OLLAMA_MODEL):
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model '{OLLAMA_MODEL}' not ready or failed to load",
        )
    llm = init_chat_model(f"ollama:{OLLAMA_MODEL}", temperature=0)

    db_passwd = Path(DB_PASSWORD_FILE).read_text()
    db_uri = f"mysql+pymysql://{DB_USER}:{db_passwd}@{DB_HOST}/{DB_NAME}"
    try:
        db = SQLDatabase.from_uri(db_uri)
    except Exception:
        raise HTTPException(
            status_code=503, detail="Database not reachable or misconfigured"
        )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.
    The SQL schema contains only sample rows, not the full dataset. Treat it as structure only.
    Generate SQL queries to retrieve information instead of assuming data from the schema.
    
    Do not include any special characters in your final output, e.g. new line characters.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": q}]})
    return dict(response["messages"][-1])
