import chromadb
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    distance: float


class GDPRRetriever:
    def __init__(
        self, collection_name: str = "gdpr_knowledge_base", n_results: int = 5
    ):
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_collection(name=collection_name)
        self.n_results = n_results
        logger.info(
            f"Retriever connected. Collection has {self.collection.count()} chunks."
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.n_results,
        )

        documents = results["documents"] or []
        metadatas = results["metadatas"] or []
        distances = results["distances"] or []

        chunks = []
        for text, metadata, distance in zip(
            documents[0] if documents else [],
            metadatas[0] if metadatas else [],
            distances[0] if distances else [],
        ):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    page=int(str((metadata.get("page", 0)))),
                    distance=distance,
                )
            )

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
        return chunks

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formats retrieved chunks into a single context string for the LLM."""
        parts = [
            f"[Source: {chunk.source}, Page {chunk.page}]\n{chunk.text}"
            for chunk in chunks
        ]
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    retriever = GDPRRetriever(n_results=3)
    query = "What are the rights of data subjects under GDPR?"
    chunks = retriever.retrieve(query)

    for chunk in chunks:
        print(f"\n📄 Page {chunk.page} | Distance: {chunk.distance:.4f}")
        print(chunk.text[:300])
        print("...")
