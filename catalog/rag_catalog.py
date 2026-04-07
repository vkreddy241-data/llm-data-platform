"""
RAG Data Catalog: Natural language interface to your data assets.

Users can ask questions like:
  - "Which table has transaction amounts?"
  - "What are all tables with anomaly data?"
  - "Show me tables updated in the last 24 hours"
  - "Which columns contain PII?"

Uses ChromaDB for semantic search + Claude for answer generation.
"""

import json
import logging
import os
from typing import Optional

import anthropic
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL  = "claude-3-5-sonnet-20241022"
CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_catalog")
COLLECTION_NAME  = "data_catalog"
EMBEDDING_MODEL  = "text-embedding-3-small"
TOP_K            = 5   # number of tables to retrieve per query


class RAGDataCatalog:

    def __init__(self):
        self.client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.chroma  = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=EMBEDDING_MODEL,
        )
        self.collection = self.chroma.get_or_create_collection(
            COLLECTION_NAME, embedding_function=self.ef
        )

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Semantic search over catalog embeddings."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )
        tables = []
        for i, doc in enumerate(results["documents"][0]):
            tables.append({
                "document":  doc,
                "metadata":  results["metadatas"][0][i],
                "distance":  results["distances"][0][i],
            })
        return tables

    def build_rag_prompt(self, query: str, context_tables: list[dict]) -> str:
        context = "\n\n".join([
            f"TABLE: {t['metadata']['table_name']}\n{t['document']}"
            for t in context_tables
        ])
        return f"""You are a helpful data catalog assistant for a data engineering team.

A user has asked the following question about the data catalog:
QUESTION: {query}

Here are the most relevant tables from the catalog:
{context}

Answer the user's question accurately and helpfully. If the answer references specific tables or columns,
be precise. If you're not sure, say so. Format your response clearly — use bullet points or a table if helpful.
Include the Delta Lake path when recommending a specific table."""

    def query(self, question: str) -> dict:
        """End-to-end RAG: retrieve relevant tables, generate LLM answer."""
        logger.info(f"RAG query: {question}")

        # Step 1: Retrieve relevant tables
        context_tables = self.retrieve(question)

        if not context_tables:
            return {
                "answer":  "No tables found in the catalog yet. Run the embedder first.",
                "sources": [],
            }

        # Step 2: Generate answer with LLM
        prompt = self.build_rag_prompt(question, context_tables)
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text.strip()

        return {
            "answer":  answer,
            "sources": [
                {
                    "table":    t["metadata"]["table_name"],
                    "path":     t["metadata"]["delta_path"],
                    "row_count": t["metadata"]["row_count"],
                    "relevance_score": round(1 - t["distance"], 3),
                }
                for t in context_tables
            ],
            "query":  question,
        }

    def list_tables(self) -> list[dict]:
        """List all tables in the catalog."""
        all_items = self.collection.get()
        return [
            {
                "table_name":   m["table_name"],
                "delta_path":   m["delta_path"],
                "row_count":    m["row_count"],
                "column_count": m["column_count"],
            }
            for m in all_items["metadatas"]
        ]

    def describe_table(self, table_name: str) -> Optional[dict]:
        """Get full description of a specific table."""
        results = self.collection.get(ids=[table_name])
        if not results["documents"]:
            return None
        meta = results["metadatas"][0]
        return {
            "table_name":  meta["table_name"],
            "delta_path":  meta["delta_path"],
            "row_count":   meta["row_count"],
            "columns":     json.loads(meta["columns_json"]),
            "description": results["documents"][0],
        }


if __name__ == "__main__":
    catalog = RAGDataCatalog()

    questions = [
        "Which table contains transaction amounts and fraud risk scores?",
        "What tables have anomaly data?",
        "Which columns might contain PII?",
        "What is the largest table by row count?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = catalog.query(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['table'] for s in result['sources']]}")
