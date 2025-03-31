# ai_nl2sql_rag_assistant/app/app.py

import json
import streamlit as st
import sqlite3
import cohere
from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load environment variables (for Cohere API key)
load_dotenv()
client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample_transactions.db")
DOC_PATH = str(Path(__file__).resolve().parent.parent / "data" / "product_docs")
PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "storage")


def query_database(nl_query):
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
        response = client.chat(
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
            }
        )
        response_data = json.loads(response.message.content[0].text)
        sql_query = response_data.get("sql_query", "").strip()

        conn = sqlite3.connect(DB_PATH)
        result = conn.execute(sql_query).fetchall()
        columns = [description[0] for description in conn.execute(sql_query).description]
        conn.close()
        return sql_query, columns, result
    except Exception as e:
        return sql_query if 'sql_query' in locals() else "", [], [["Error executing SQL:", str(e)]]


def ask_docs(query):
    try:
        if not os.path.exists(PERSIST_DIR):
            documents = SimpleDirectoryReader(DOC_PATH).load_data()
            vector_store = ChromaVectorStore(chroma_collection=chromadb.Client().create_collection("docs"))
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=CohereEmbedding(model_name="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY")),
                vector_store=vector_store,
            )
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error: {e}"


def main():
    st.set_page_config(page_title="AI Assistant", layout="wide")
    st.title("🧠 AI Assistant: NL to SQL + Doc QA")

    with st.expander("ℹ️ Reference Info: Click to view sample data and docs"):
        st.markdown("""
        ### 🗄️ Database Schema
        - **Table**: `transactions`
        - **Columns**:
          - `id` (Integer) – Primary key
          - `user_id` (Integer)
          - `transaction_amount` (Float)
          - `transaction_type` (String): `transfer`, `withdrawal`, `deposit`, `payment`
          - `status` (String): `success`, `pending`, `failed`
          - `transaction_date` (Date): `YYYY-MM-DD`

        ### 📄 Documents Info
        **MoneyTransferPro.txt**:
        - Daily limit: ₹1,00,000
        - Max 20 transactions/day
        - Delay possible between 11PM–4AM
        - 2FA & AI fraud detection

        **PaySecurePlus.txt**:
        - International payments up to ₹2,00,000/txn
        - Biometric + OTP login
        - Geo-fencing enabled
        - Chat support 9AM–9PM IST
        """)

    tab1, tab2 = st.tabs(["🗄️ Ask the DB", "📄 Ask the Docs"])

    with tab1:
        nl_input = st.text_input("Enter your query to fetch data from the DB:")
        if st.button("Ask DB") and nl_input:
            sql, cols, rows = query_database(nl_input)
            st.markdown(f"**Generated SQL:** `{sql}`")
            if cols:
                st.dataframe([dict(zip(cols, row)) for row in rows])
            else:
                st.write(rows)

    with tab2:
        doc_input = st.text_input("Ask a question about the product documents:")
        if st.button("Ask Docs") and doc_input:
            answer = ask_docs(doc_input)
            st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
