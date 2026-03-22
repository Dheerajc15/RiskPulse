"""
RiskPulse RAG (Retrieval-Augmented Generation) Chain
=====================================================
Ingests documents, embeds with sentence-transformers, stores in ChromaDB,
and provides a retrieval chain for market intelligence queries.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# ---- Configuration ----
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
COLLECTION_NAME = "riskpulse_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def ingest_documents(documents_dir: str = DOCUMENTS_DIR) -> int:
    """
    Read all .txt files from documents_dir, chunk them,
    embed with sentence-transformers, and store in ChromaDB.
    
    Returns the number of chunks stored.
    """
    # Initialize ChromaDB with persistence
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Use sentence-transformers for embedding
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Get or create collection (delete old one to re-ingest)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    # Read and chunk all documents
    all_chunks = []
    all_metadatas = []
    all_ids = []

    txt_files = glob.glob(os.path.join(documents_dir, "*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {documents_dir}")
        return 0

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": filename, "chunk_index": i})
            all_ids.append(f"{filename}_{i}")

    # Add to ChromaDB
    collection.add(
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    logger.info(f"Ingested {len(all_chunks)} chunks from {len(txt_files)} documents")
    return len(all_chunks)


def retrieve_chunks(query: str, top_k: int = 3) -> List[dict]:
    """
    Query ChromaDB and return the top-k most relevant chunks.
    
    Returns list of dicts: [{text, source, score}, ...]
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": round(results["distances"][0][i], 4),
        })

    return chunks


def build_rag_prompt(question: str, chunks: List[dict]) -> str:
    """
    Build a prompt combining the user question with retrieved context chunks.
    """
    context = "\n\n---\n\n".join([c["text"] for c in chunks])

    prompt = f"""You are a quantitative market risk analyst. Answer the following question 
using ONLY the context provided below. If the context doesn't contain enough information, 
say so explicitly. Be precise and cite specific data points.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt


def get_rag_answer(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve chunks → build prompt → generate answer.
    
    Note: For the LLM generation step, this uses a simple template-based
    response from the retrieved chunks. To use OpenAI/local LLM, 
    uncomment the LLM section below and set your API key.
    """
    chunks = retrieve_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant documents found in the knowledge base.",
            "sources": [],
        }

    prompt = build_rag_prompt(question, chunks)

    # ---- Option A: Template-based answer (no LLM API needed) ----
    # This extracts and presents the most relevant context directly
    answer_parts = []
    for i, chunk in enumerate(chunks):
        answer_parts.append(f"[Source: {chunk['source']}] {chunk['text']}")

    answer = (
        f"Based on {len(chunks)} retrieved document chunks:\n\n"
        + "\n\n".join(answer_parts)
    )

    # ---- Option B: OpenAI LLM (uncomment to use) ----
    # import openai
    # client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.2,
    #     max_tokens=500,
    # )
    # answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": [{"source": c["source"], "relevance_distance": c["distance"]} for c in chunks],
        "prompt_used": prompt,
    }


# ---- CLI for testing ----
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Step 1: Ingesting documents...")
    print("=" * 60)
    n_chunks = ingest_documents()
    print(f"Stored {n_chunks} chunks in ChromaDB\n")

    print("=" * 60)
    print("Step 2: Testing retrieval...")
    print("=" * 60)
    test_queries = [
        "What was the market impact of Section 301 tariffs on EUR/USD?",
        "What is the current VIX level and what does GARCH analysis show?",
        "What is the Federal Funds Rate and when might the Fed cut rates?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 40)
        result = get_rag_answer(q, top_k=2)
        print(json.dumps({k: v for k, v in result.items() if k != "prompt_used"}, indent=2))

    print("\n✅ RAG pipeline test complete!")