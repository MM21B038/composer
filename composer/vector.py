from __future__ import annotations

from typing import Optional, Sequence, Union, overload

import numpy as np
from langchain_openai import OpenAIEmbeddings
from numpy.typing import NDArray


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

        client_kwargs: dict = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            # OpenAI SDK defaults to base64; many OpenAI-compatible providers
            # (e.g. OpenRouter) only accept float embeddings.
            "model_kwargs": {"encoding_format": "float"},
        }
        if dim is not None:
            client_kwargs["dimensions"] = dim
        if base_url is not None:
            # Send raw text instead of tiktoken chunks for third-party endpoints.
            client_kwargs["check_embedding_ctx_length"] = False
        self._client = OpenAIEmbeddings(**client_kwargs)

    def _postprocess(self, vec: list[float]) -> NDArray[np.float64]:
        out = np.asarray(vec, dtype=np.float64)
        if self.dim is not None:
            out = out[: self.dim]
        if not self.normalized:
            return out
        norm = np.linalg.norm(out)
        if norm == 0:
            return out
        return out / norm

    @staticmethod
    def _validate_docs(docs: Sequence[str]) -> None:
        for i, doc in enumerate(docs):
            if not isinstance(doc, str):
                raise TypeError(
                    f"Expected str documents, got {type(doc).__name__} at index {i}"
                )

    @overload
    def vector(self, docs: str) -> NDArray[np.float64]: ...

    @overload
    def vector(self, docs: Sequence[str]) -> NDArray[np.float64]: ...

    def vector(
        self, docs: Union[str, Sequence[str]]
    ) -> NDArray[np.float64]:
        if isinstance(docs, str):
            return self._postprocess(self._client.embed_query(docs))
        if not isinstance(docs, (list, tuple)):
            raise TypeError(
                f"Expected str or sequence of str, got {type(docs).__name__}"
            )
        if len(docs) == 0:
            return np.empty((0, 0), dtype=np.float64)
        self._validate_docs(docs)
        return np.stack(
            [self._postprocess(v) for v in self._client.embed_documents(list(docs))]
        )

    @overload
    async def avector(self, docs: str) -> NDArray[np.float64]: ...

    @overload
    async def avector(self, docs: Sequence[str]) -> NDArray[np.float64]: ...

    async def avector(
        self, docs: Union[str, Sequence[str]]
    ) -> NDArray[np.float64]:
        if isinstance(docs, str):
            return self._postprocess(await self._client.aembed_query(docs))
        if not isinstance(docs, (list, tuple)):
            raise TypeError(
                f"Expected str or sequence of str, got {type(docs).__name__}"
            )
        if len(docs) == 0:
            return np.empty((0, 0), dtype=np.float64)
        self._validate_docs(docs)
        raw = await self._client.aembed_documents(list(docs))
        return np.stack([self._postprocess(v) for v in raw])

    def __repr__(self) -> str:
        url = "set" if self.base_url else "default"
        return (
            f"Vector(model={self.model!r}, dim={self.dim}, "
            f"normalized={self.normalized}, base_url={url})"
        )
