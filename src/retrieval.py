"""Vector search over the student's notes with ChromaDB.

How it works:
1. Each chunk is embedded with EmbeddingGemma and stored in ChromaDB
   together with its document id, document name and page number.
2. A question is embedded the same way and ChromaDB returns the closest chunks.
3. Grounding check (safeguard #1): if even the best chunk is less similar than
   MIN_RELEVANCE, the question is treated as NOT covered by the notes and
   Gemma is never asked to answer it.

Limitation (be honest in the demo): a similarity threshold catches clearly
off-topic questions ("What is the GDP of India?"), but an on-topic question
whose detail is missing ("Who discovered photosynthesis?") can still pass.
Phase 2 adds a second check inside the tutor for that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from src import config
from src.document_loader import LoadedDocument
from src.embeddings import Embedder
from src.text_chunker import Chunk, chunk_document


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float              # cosine similarity: 1.0 = identical meaning, ~0 = unrelated


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)   # top-k, best first
    min_score: float = 0.0

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    @property
    def supported(self) -> bool:
        """True when the notes look relevant enough to answer from."""
        return self.top_score >= self.min_score

    @property
    def evidence(self) -> list[RetrievedChunk]:
        """Only the chunks that passed the threshold; these go to Gemma."""
        return [c for c in self.chunks if c.score >= self.min_score]


class NoteIndex:
    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str | Path | None = None,
        collection_name: str = "notewise_notes",
    ) -> None:
        self.embedder = embedder
        path = Path(persist_dir or config.CHROMA_DIR)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # embedding_function=None: we always pass our own EmbeddingGemma vectors,
        # so Chroma never downloads its default model.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    # ---------- indexing ----------
    def has_document(self, doc_id: str) -> bool:
        found = self._collection.get(where={"doc_id": doc_id}, limit=1)
        return bool(found["ids"])

    def chunk_count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"doc_id": doc_id})["ids"])

    def add_document(self, doc: LoadedDocument) -> int:
        """Index a document once. Re-adding the same file is a no-op."""
        if self.has_document(doc.doc_id):
            return self.chunk_count(doc.doc_id)
        chunks = chunk_document(doc)
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def get_chunks(self, doc_ids: list[str]) -> list[Chunk]:
        """Every stored passage for these documents, in page order (used by the quiz)."""
        if not doc_ids:
            return []
        where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        found = self._collection.get(where=where, include=["documents", "metadatas"])
        chunks = [
            Chunk(
                chunk_id=cid,
                doc_id=str(meta["doc_id"]),
                doc_name=str(meta["doc_name"]),
                page_number=int(meta["page_number"]),
                chunk_index=int(meta["chunk_index"]),
                text=text,
            )
            for cid, text, meta in zip(found["ids"], found["documents"], found["metadatas"])
        ]
        return sorted(chunks, key=lambda c: (c.doc_name, c.page_number, c.chunk_index))

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    # ---------- search ----------
    def search(
        self,
        query: str,
        doc_ids: list[str],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Search ONLY the given documents. No documents -> nothing is supported,
        so we never silently answer from a different set of notes."""
        min_score = config.MIN_RELEVANCE if min_score is None else min_score
        result = RetrievalResult(query=query, min_score=min_score)
        query = query.strip()
        if not query or not doc_ids:
            return result

        where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        available = self._collection.get(where=where, limit=1)["ids"]
        if not available:
            return result

        response = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k or config.TOP_K,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        for chunk_id, text, meta, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=str(meta["doc_id"]),
                doc_name=str(meta["doc_name"]),
                page_number=int(meta["page_number"]),
                chunk_index=int(meta["chunk_index"]),
                text=text,
            )
            result.chunks.append(RetrievedChunk(chunk=chunk, score=round(1.0 - distance, 4)))
        result.chunks.sort(key=lambda c: c.score, reverse=True)
        return result
