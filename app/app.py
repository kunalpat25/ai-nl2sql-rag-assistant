import json
import streamlit as st
import sqlite3
import cohere
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables (for Cohere API key)
load_dotenv()
client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample_transactions.db")

def query_database(nl_query):
    # Sample prompt - improve this later
    system_prompt = f"""
    Convert the following natural language query into an SQL statement compatible with SQLite:
    SQL table schema: 
Table Name: transactions

Columns:
id (Integer) – Primary key; unique identifier for each transaction.
user_id (Integer) – ID of the user performing the transaction.
transaction_amount (Float) – The amount involved in the transaction.
transaction_type (String) – Type of transaction: can be transfer, withdrawal, deposit, or payment.
status (String) – Status of the transaction: values include success, pending, or failed.
transaction_date (Date) – Date when the transaction took place, in YYYY-MM-DD format.

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
        print("Response:", response_data)
        sql_query = response_data.get("sql_query", "").strip()
        description = response_data.get("description", "").strip()
        print("Description:", description)

        conn = sqlite3.connect(DB_PATH)
        result = conn.execute(sql_query).fetchall()
        columns = [description[0] for description in conn.execute(sql_query).description]
        conn.close()
        return sql_query, columns, result
    except Exception as e:
        return sql_query if 'sql_query' in locals() else "", [], [["Error executing SQL:", str(e)]]


def main():
    st.title("🧠 AI Assistant: NL to SQL + Doc QA")

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
        st.write("📄 RAG-based Doc QA coming next...")


if __name__ == "__main__":
    main()