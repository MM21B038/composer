from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union, overload

from langchain_openai import OpenAIEmbeddings


class Vector:
    """OpenAI-compatible embedding client with optional dim truncation and L2 normalization."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: Optional[int] = None,
        normalized: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dim = dim
        self.normalized = normalized

        kwargs: dict = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }
        if dim is not None:
            kwargs["dimensions"] = dim
        self._client = OpenAIEmbeddings(**kwargs)

    def _postprocess(self, vec: List[float]) -> List[float]:
        out = vec[: self.dim] if self.dim is not None else vec
        if not self.normalized:
            return out
        norm = math.sqrt(sum(x * x for x in out))
        if norm == 0:
            return out
        return [x / norm for x in out]

    @staticmethod
    def _validate_docs(docs: Sequence[str]) -> None:
        for i, doc in enumerate(docs):
            if not isinstance(doc, str):
                raise TypeError(
                    f"Expected str documents, got {type(doc).__name__} at index {i}"
                )

    @overload
    def vector(self, docs: str) -> List[float]: ...

    @overload
    def vector(self, docs: Sequence[str]) -> List[List[float]]: ...

    def vector(
        self, docs: Union[str, Sequence[str]]
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(docs, str):
            return self._postprocess(self._client.embed_query(docs))
        if not isinstance(docs, (list, tuple)):
            raise TypeError(
                f"Expected str or sequence of str, got {type(docs).__name__}"
            )
        if len(docs) == 0:
            return []
        self._validate_docs(docs)
        return [self._postprocess(v) for v in self._client.embed_documents(list(docs))]

    @overload
    async def avector(self, docs: str) -> List[float]: ...

    @overload
    async def avector(self, docs: Sequence[str]) -> List[List[float]]: ...

    async def avector(
        self, docs: Union[str, Sequence[str]]
    ) -> Union[List[float], List[List[float]]]:
        if isinstance(docs, str):
            return self._postprocess(await self._client.aembed_query(docs))
        if not isinstance(docs, (list, tuple)):
            raise TypeError(
                f"Expected str or sequence of str, got {type(docs).__name__}"
            )
        if len(docs) == 0:
            return []
        self._validate_docs(docs)
        raw = await self._client.aembed_documents(list(docs))
        return [self._postprocess(v) for v in raw]

    def __repr__(self) -> str:
        url = "set" if self.base_url else "default"
        return (
            f"Vector(model={self.model!r}, dim={self.dim}, "
            f"normalized={self.normalized}, base_url={url})"
        )
