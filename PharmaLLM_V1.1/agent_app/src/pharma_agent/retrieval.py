import hashlib
import math
import re

import httpx
from sqlalchemy import and_, or_, select, text

from pharma_agent.db import Document, DocumentChunk, new_id
from pharma_agent.errors import AppError
from pharma_agent.observability import digest, span
from pharma_agent.schemas import Citation, DocumentInput, ReferencesResult


class EmbeddingGateway:
    def __init__(self, settings, client):
        self.settings, self.client = settings, client
        self.version = (
            "chunk-v1-"
            + digest(
                {
                    "provider": settings.embedding_provider,
                    "name": settings.embedding_model_name,
                    "revision": settings.embedding_revision,
                    "dimensions": settings.embedding_dimensions,
                }
            )[7:31]
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        dimensions = self.settings.embedding_dimensions
        if self.settings.embedding_provider == "hash":
            results = []
            for content in texts:
                normalized = re.sub(r"\s+", "", content.lower())
                grams = [normalized[i : i + 2] for i in range(max(1, len(normalized) - 1))]
                vector = [0.0] * dimensions
                for gram in grams:
                    code = hashlib.sha256(gram.encode()).digest()
                    vector[int.from_bytes(code[:4], "big") % dimensions] += 1 if code[4] % 2 else -1
                norm = math.sqrt(sum(v * v for v in vector)) or 1
                results.append([v / norm for v in vector])
            return results
        headers = {}
        key = self.settings.embedding_api_key.get_secret_value()
        if key:
            headers["Authorization"] = "Bearer " + key
        try:
            response = await self.client.post(
                self.settings.embedding_base_url.rstrip("/") + "/embeddings",
                headers=headers,
                json={"model": self.settings.embedding_model_name, "input": texts},
                timeout=30,
            )
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
            if len(vectors) != len(texts) or any(len(v) != dimensions for v in vectors):
                raise ValueError("embedding dimensions mismatch")
            if any(not math.isfinite(float(x)) for v in vectors for x in v):
                raise ValueError("nonfinite embedding")
            if any(not any(float(x) != 0 for x in v) for v in vectors):
                raise ValueError("zero embedding")
            return vectors
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise AppError(
                "EMBEDDING_UNAVAILABLE", "Embedding 服务不可用或输出维度不匹配", 503, True
            ) from None


class RetrievalService:
    def __init__(self, sessions, embedding):
        self.sessions, self.embedding = sessions, embedding

    async def ingest(self, item: DocumentInput, tenant_id: str | None, public: bool = False):
        if public and tenant_id is not None:
            raise ValueError("public document cannot have tenant_id")
        chunks = [item.content[i : i + 600] for i in range(0, len(item.content), 500)]
        vectors = []
        for start in range(0, len(chunks), 16):
            vectors.extend(await self.embedding.embed(chunks[start : start + 16]))
        with span("rag.ingest", index_version=self.embedding.version, chunk_count=len(chunks)):
            async with self.sessions() as session, session.begin():
                lock_hash = hashlib.sha256(
                    f"{tenant_id}:{item.source}:{item.source_version}".encode()
                ).digest()
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": int.from_bytes(lock_hash[:8], "big", signed=True)},
                )
                document = (
                    await session.scalars(
                        select(Document)
                        .where(
                            Document.tenant_id == tenant_id,
                            Document.source == item.source,
                            Document.source_version == item.source_version,
                        )
                        .with_for_update()
                    )
                ).first()
                if document is None:
                    document = Document(
                        id=new_id("doc"),
                        tenant_id=tenant_id,
                        visibility="public" if public else "tenant",
                        title=item.title,
                        source=item.source,
                        source_version=item.source_version,
                        license=item.license,
                        published_at=item.published_at,
                        content_hash=digest(item.content),
                    )
                    session.add(document)
                    await session.flush()
                else:
                    if (
                        document.content_hash != digest(item.content)
                        or document.title != item.title
                        or document.license != item.license
                        or document.published_at != item.published_at
                    ):
                        raise AppError(
                            "DOCUMENT_VERSION_CONFLICT",
                            "同一来源版本的内容不可覆盖，请递增 source_version",
                            409,
                        )
                    indexed = (
                        await session.scalars(
                            select(DocumentChunk.id)
                            .where(
                                DocumentChunk.document_id == document.id,
                                DocumentChunk.index_version == self.embedding.version,
                            )
                            .limit(1)
                        )
                    ).first()
                    if indexed:
                        return {
                            "document_id": document.id,
                            "chunks": len(chunks),
                            "index_version": self.embedding.version,
                        }
                for index, (content, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    session.add(
                        DocumentChunk(
                            id=new_id("chunk"),
                            document_id=document.id,
                            chunk_index=index,
                            content=content,
                            embedding=vector,
                            index_version=self.embedding.version,
                        )
                    )
                return {
                    "document_id": document.id,
                    "chunks": len(chunks),
                    "index_version": self.embedding.version,
                }

    async def search(self, tenant_id: str, query: str, top_k: int) -> ReferencesResult:
        vector = (await self.embedding.embed([query]))[0]
        distance = DocumentChunk.embedding.cosine_distance(vector).label("distance")
        with span(
            "rag.search", tenant_id=tenant_id, index_version=self.embedding.version, query_hash=digest(query)
        ):
            async with self.sessions() as session:
                rows = (
                    await session.execute(
                        select(Document, DocumentChunk, distance)
                        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
                        .where(
                            or_(
                                and_(Document.visibility == "public", Document.tenant_id.is_(None)),
                                and_(Document.visibility == "tenant", Document.tenant_id == tenant_id),
                            ),
                            DocumentChunk.index_version == self.embedding.version,
                        )
                        .order_by(distance)
                        .limit(top_k)
                    )
                ).all()
            references = []
            for document, chunk, dist in rows:
                if dist is None or not math.isfinite(float(dist)) or dist >= 0.95:
                    continue
                references.append(
                    Citation(
                        document_id=document.id,
                        chunk_id=chunk.id,
                        title=document.title,
                        source=document.source,
                        source_version=document.source_version,
                        license=document.license,
                        published_at=document.published_at,
                        index_version=chunk.index_version,
                        excerpt=chunk.content,
                        score=round(1 - float(dist), 5),
                    )
                )
            return ReferencesResult(
                references=references, retrieval_mode=self.embedding.settings.embedding_provider
            )
