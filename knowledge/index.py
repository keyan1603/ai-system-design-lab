"""Access-controlled retrieval on Requisite's own filtered retrievers (0.41.0+).

Every chunk is stamped with the groups allowed to read it (`groups=[...]`);
every retrieval that serves a user passes `filter={"groups": {"$in": user.groups}}`.
Requisite applies that filter inside the store and, for `HybridRetriever`,
to the keyword side as well, before scoring and fusion, so an unauthorized
chunk is never a candidate and cannot influence the rank or score of allowed
chunks. A user with no groups gets `{"$in": []}`, which matches nothing
(default deny). Authorization is enforced here, in code, never by asking the
model to behave.
"""

from __future__ import annotations

from dataclasses import dataclass

from requisite.rag import HybridRetriever, Retriever
from requisite.rag.vectorstores import InMemoryVectorStore

from knowledge.corpus import Doc, User


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    score: float


class SecureIndex:
    def __init__(self, embedding_provider, hybrid: bool = True, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        cls = HybridRetriever if hybrid else Retriever
        self.retriever = cls(embedding_provider=embedding_provider, vector_store=InMemoryVectorStore())
        self.hybrid = hybrid
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def ingest(self, docs: list[Doc]) -> int:
        """Chunk, embed and store every document with its access groups. Returns the chunk count."""
        total = 0
        for doc in docs:
            meta = {"title": doc.title, "groups": list(doc.allowed_groups)}
            total += len(self.retriever.add_texts([doc.text], metadatas=[meta], doc_ids=[doc.id],
                                                  chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap))
        return total

    def search(self, user: User, query: str, top_k: int = 4, enforce_acl: bool = True) -> list[Hit]:
        # enforce_acl=False is the control condition for the evaluation only: a call without a filter sees everything.
        flt = {"groups": {"$in": list(user.groups)}} if enforce_acl else None
        scored = self.retriever.retrieve(query, top_k=top_k, filter=flt)
        return [Hit(sc.chunk.metadata["doc_id"], sc.chunk.metadata["title"], sc.chunk.text, sc.score) for sc in scored]
