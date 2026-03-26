"""Main module for the SQLAgent FastAPI application."""

from fastapi import FastAPI, HTTPException
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama

from sqlagent.config import get_settings
from sqlagent.model_utils import is_model_available

app = FastAPI(title="SQLAgent")


def get_llm() -> ChatOllama:
    """Create a ChatOllama client for the configured Ollama endpoint."""
    settings = get_settings()
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def get_database() -> SQLDatabase:
    """Create a SQLDatabase instance from the configured connection details."""
    settings = get_settings()
    return SQLDatabase.from_uri(settings.database_uri)


@app.get("/health")
def health() -> dict:
    """Report whether the LLM and database are reachable."""
    settings = get_settings()
    model_ready = is_model_available(
        settings.ollama_model,
        host=settings.ollama_base_url,
    )
    if not model_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model '{settings.ollama_model}' is not ready",
        )

    try:
        get_database()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database not reachable or misconfigured",
        ) from exc

    return {
        "status": "ok",
        "ollama_model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
    }


@app.get("/")
def main(q: str) -> dict:
    """Main endpoint to process user queries."""
    settings = get_settings()

    if not is_model_available(settings.ollama_model, host=settings.ollama_base_url):
        raise HTTPException(
            status_code=503,
            detail=f"Ollama model '{settings.ollama_model}' not ready or failed to load",
        )
    llm = get_llm()

    try:
        db = get_database()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Database not reachable or misconfigured"
        ) from exc

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
        top_k=settings.top_k,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": q}]})
    return dict(response["messages"][-1])
