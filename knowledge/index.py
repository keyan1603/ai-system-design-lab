"""Access-controlled retrieval on top of Requisite's RAG components.

Requisite supplies the embedding provider, the vector store and its metadata
filter. What it does not supply today is an access-control layer, for two
reasons found by reading the source:

* `Retriever.retrieve()` and `HybridRetriever.retrieve()` take no `filter`;
  only `BaseVectorStore.search()` does.
* that filter is exact equality per key, so "readable by any of this user's
  groups" cannot be written as one filter.

The lab's answer: every chunk is stamped with one boolean flag per allowed
group (`acl_eng=True`), and retrieval runs one *pre-filtered* store search per
group the user belongs to, then merges. Filtering happens inside the store,
before ranking, so an unauthorized chunk is never a candidate and cannot
influence rank or count. A user with no groups gets nothing (default deny).
Authorization is enforced here, in code, never by asking the model to behave.

`HybridRetriever` is not used: its BM25 side is a separate unfiltered index,
so keyword matches would bypass the store's filter entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from requisite.rag import Retriever
from requisite.rag.base import ScoredChunk
from requisite.rag.vectorstores import InMemoryVectorStore

from knowledge.corpus import Doc, User


def acl_flag(group: str) -> str:
    return f"acl_{group}"


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    score: float


class SecureIndex:
    def __init__(self, embedding_provider, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        self.retriever = Retriever(embedding_provider=embedding_provider, vector_store=InMemoryVectorStore())
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def ingest(self, docs: list[Doc]) -> int:
        """Chunk, embed and store every document with its access flags. Returns the chunk count."""
        total = 0
        for doc in docs:
            meta = {"doc_id": doc.id, "title": doc.title, **{acl_flag(g): True for g in doc.allowed_groups}}
            total += len(self.retriever.add_texts([doc.text], metadatas=[meta],
                                                  chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap))
        return total

    def search(self, user: User, query: str, top_k: int = 4, enforce_acl: bool = True) -> list[Hit]:
        embedding = self.retriever.embedding_provider.embed_one(query)
        store = self.retriever.vector_store
        if enforce_acl:
            best: dict[str, ScoredChunk] = {}
            for group in user.groups:                      # no groups -> no searches -> no results
                for sc in store.search(embedding, top_k=top_k, filter={acl_flag(group): True}):
                    if sc.chunk.id not in best or sc.score > best[sc.chunk.id].score:
                        best[sc.chunk.id] = sc
            ranked = sorted(best.values(), key=lambda s: s.score, reverse=True)[:top_k]
        else:
            # Control condition for the evaluation only: retrieval with no access control.
            ranked = store.search(embedding, top_k=top_k)
        return [Hit(sc.chunk.metadata["doc_id"], sc.chunk.metadata["title"], sc.chunk.text, sc.score) for sc in ranked]
