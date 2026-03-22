# %% [markdown]
# # 03 — RAG Market Intelligence Pipeline
#
# Ingest FRED press releases and tariff documents → embed with sentence-transformers →
# store in ChromaDB → retrieve and answer market intelligence queries.

# %%
import sys
sys.path.insert(0, "..")

from core.rag_chain import ingest_documents, retrieve_chunks, get_rag_answer, build_rag_prompt
import json

# %% [markdown]
# ## Step 1: Ingest Documents into ChromaDB

# %%
n_chunks = ingest_documents()
print(f"Total chunks ingested: {n_chunks}")

# %% [markdown]
# ## Step 2: Test Retrieval (Semantic Search)

# %%
query = "What was the market impact of Section 301 tariffs on EUR/USD?"
chunks = retrieve_chunks(query, top_k=3)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} (distance: {chunk['distance']}) ---")
    print(f"Source: {chunk['source']}")
    print(chunk['text'][:200] + "...")

# %% [markdown]
# ## Step 3: Full RAG Pipeline

# %%
test_queries = [
    "What was the market impact of Section 301 tariffs on EUR/USD?",
    "What does the VIX indicate about current market conditions?",
    "When is the Fed expected to cut rates and how does it affect equities?",
    "What happened to GARCH volatility persistence after tariff announcements?",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print('='*60)
    result = get_rag_answer(q, top_k=2)
    print(f"\nAnswer:\n{result['answer'][:300]}...")
    print(f"\nSources: {[s['source'] for s in result['sources']]}")

# %% [markdown]
# ## Step 4: View the Prompt Sent to LLM

# %%
chunks = retrieve_chunks("Section 301 tariff impact", top_k=3)
prompt = build_rag_prompt("What was the market impact of Section 301 tariffs on EUR/USD?", chunks)
print(prompt)

# %% [markdown]
# ## Step 5: Test via FastAPI
# 
# With the server running (`uvicorn api.main:app --reload`), test:
# ```bash
# curl -X POST http://localhost:8000/rag/query \
#   -H "Content-Type: application/json" \
#   -d '{"question": "What was the market impact of Section 301 tariffs on EUR/USD?"}'
# ```