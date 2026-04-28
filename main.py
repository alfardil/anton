import logging
import json
import requests
from retriever import GDPRRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a GDPR compliance expert assistant. Answer questions strictly based on the provided context excerpts from the GDPR document. 
 
Rules:
- Only use information from the provided context
- If the context doesn't contain enough information, say so clearly
- Cite the page numbers from the context when possible
- Be precise and concise"""


def build_prompt(query: str, context: str) -> str:
    return f"""Context excerpts from the GDPR:
 
{context}
 
---
 
Question: {query}
 
Answer based only on the context above:"""


def query_ollama(prompt: str, stream: bool = True) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": stream,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, stream=stream)
    response.raise_for_status()

    if stream:
        full_response = ""
        print("\nAnswer:\n")
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                if chunk.get("done"):
                    print("\n")
                    break
        return full_response
    else:
        return response.json()["response"]


def rag_query(query: str, n_results: int = 5, stream: bool = True) -> str:
    """Full RAG pipeline: retrieve → format → generate."""

    retriever = GDPRRetriever(n_results=n_results)
    chunks = retriever.retrieve(query)

    print(f"\nRetrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"  - Page {c.page} (distance: {c.distance:.4f}): {c.text[:80]}...")

    context = retriever.format_context(chunks)
    prompt = build_prompt(query, context)

    logger.info(f"Prompt length: {len(prompt)} chars — sending to {MODEL}")

    return query_ollama(prompt, stream=stream)


def interactive_mode():
    """Simple REPL for chatting with your GDPR knowledge base."""
    print(f"GDPR RAG System | Model: {MODEL}")
    print("Type 'quit' to exit, 'chunks N' to change number of retrieved chunks\n")

    n_results = 5

    while True:
        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower().startswith("chunks "):
            try:
                n_results = int(query.split()[1])
                print(f"Now retrieving {n_results} chunks per query")
            except ValueError:
                print("Usage: chunks <number>")
            continue

        try:
            rag_query(query, n_results=n_results)
        except requests.ConnectionError:
            print("Cannot connect to Ollama. Is it running? Try: ollama serve")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    interactive_mode()
