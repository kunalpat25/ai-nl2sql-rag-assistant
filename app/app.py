import os
import json
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.cohere import Cohere as CohereLLM
import cohere
import chromadb

# Load environment variables
load_dotenv()

# Constants
DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample_transactions.db")
DOC_PATH = str(Path(__file__).resolve().parent.parent / "data" / "product_docs")
PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "storage")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
cohere_llm_client = cohere.ClientV2(api_key=COHERE_API_KEY)


def query_database(nl_query):
    """
    Converts a natural language query into an SQL query and executes it on the database.
    """
    system_prompt = f"""
    Convert the following natural language query into an SQL statement compatible with SQLite:
    SQL table schema: 
    Table Name: transactions
    Columns:
    id (Integer), user_id (Integer), transaction_amount (Float), transaction_type (String),
    status (String), transaction_date (Date in YYYY-MM-DD)

    Query: "{nl_query}"
    """

    try:
        # Generate SQL query using Cohere
        response = cohere_llm_client.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": system_prompt}],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "required": ["sql_query", "description"],
                    "properties": {
                        "sql_query": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
        )
        response_data = json.loads(response.message.content[0].text)
        sql_query = response_data.get("sql_query", "").strip()

        # Execute SQL query on the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(sql_query)
        result = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()

        return sql_query, columns, result
    except Exception as e:
        return sql_query if 'sql_query' in locals() else "", [], [["Error executing SQL:", str(e)]]


def ask_docs(query):
    """
    Answers questions about product documents using a vector store and Cohere LLM.
    """
    cohere_embedding_client = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )
    cohere_llm = CohereLLM(api_key=COHERE_API_KEY, model="command-r-plus")

    try:
        # Check if storage directory exists; if not, create and persist index
        if not os.path.exists(PERSIST_DIR):
            documents = SimpleDirectoryReader(DOC_PATH).load_data()
            vector_store = ChromaVectorStore(
                chroma_collection=chromadb.Client().create_collection("docs")
            )
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=cohere_embedding_client,
                vector_store=vector_store,
            )
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # Load index from storage
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(
                storage_context,
                embed_model=cohere_embedding_client,
            )

        # Query the index
        query_engine = index.as_query_engine(llm=cohere_llm)
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error: {e}"


def main():
    """
    Streamlit app entry point.
    """
    st.title("🧠 AI Assistant: NL to SQL + Doc QA")

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["🗄️ Ask the DB", "📄 Ask the Docs"])

    # Tab 1: Query the database
    with tab1:
        nl_input = st.text_input("Enter your query to fetch data from the DB:")
        if st.button("Ask DB") and nl_input:
            sql, cols, rows = query_database(nl_input)
            st.markdown(f"**Generated SQL:** `{sql}`")
            if cols:
                st.dataframe([dict(zip(cols, row)) for row in rows])
            else:
                st.write(rows)

    # Tab 2: Query the documents
    with tab2:
        doc_input = st.text_input("Ask a question about the product documents:")
        if st.button("Ask Docs") and doc_input:
            answer = ask_docs(doc_input)
            st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
