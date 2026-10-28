# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-call request coalescing and bounded vector caching for embeddings.

Concurrent ``embed`` calls that ask for the same text share one underlying
request, and successful vectors are cached for later calls.  Cancelling one
waiter never cancels work another waiter still depends on.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

_GroupResult = tuple[tuple[tuple[float, ...], ...], InferenceUsage]


class _Waiter:
    """One caller waiting for texts a shared dispatch is about to produce."""

    __slots__ = ("texts", "future", "batches")

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.future: asyncio.Future[tuple[tuple[float, ...], ...]] = asyncio.get_running_loop().create_future()
        self.batches: list[_Batch] = []


class _Batch:
    """One underlying provider request shared by every waiter that needs it."""

    __slots__ = ("texts", "usage")

    def __init__(self, texts: tuple[str, ...], usage: InferenceUsage) -> None:
        self.texts = texts
        self.usage = usage


class CoalescingEmbeddingModel:
    """Wrap an embedding model with request coalescing and an LRU vector cache."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self.max_batch_size = max_batch_size
        self.coalesce_ms = coalesce_ms / 1000.0
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        # Texts with an in-flight provider request, mapped to the future every
        # waiter for that text observes.  The request itself is owned by the
        # dispatcher task, so cancelling a waiter cannot cancel it.
        self._pending: dict[str, asyncio.Future[_GroupResult]] = {}
        # Callers waiting inside the current coalescing window.  The first one
        # owns the dispatch; the rest are resolved from the shared results.
        self._window: list[_Waiter] = []
        self._window_task: asyncio.Task | None = None
        self._window_lock = asyncio.Lock()

    # -- public API ---------------------------------------------------------

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors for ``texts`` in the caller's original order."""

        if isinstance(texts, str) or not isinstance(texts, tuple):
            raise ValueError("texts must be a tuple of strings")  # noqa: TRY003
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        # Fast path: everything is cached (or already in flight elsewhere).
        missing = [text for text in texts if text not in self._cache]
        if not missing:
            self._touch(texts)
            return EmbeddingResult(
                vectors=tuple(self._cache[text] for text in texts),
                usage=InferenceUsage(requests=0),
            )

        waiter = _Waiter(texts)
        async with self._window_lock:
            self._window.append(waiter)
            owner = self._window_task is None
            if owner:
                self._window_task = asyncio.ensure_future(self._dispatch_window())

        try:
            vectors = await waiter.future
        except BaseException:
            if not waiter.future.done():
                waiter.future.cancel()
            raise

        return EmbeddingResult(vectors=vectors, usage=self._usage_for(waiter))

    # -- internals ----------------------------------------------------------

    def _touch(self, texts: tuple[str, ...]) -> None:
        for text in texts:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)

    def _usage_for(self, waiter: _Waiter) -> InferenceUsage:
        """Sum the usage of every distinct batch this waiter's texts landed in.

        A batch shared with other callers is counted once per waiter, and a pure
        cache hit contributes nothing.
        """

        requests = 0
        input_tokens = 0
        output_tokens = 0
        counted_tokens = False
        seen: set[int] = set()
        for batch in waiter.batches:
            if id(batch) in seen:
                continue
            seen.add(id(batch))
            requests += batch.usage.requests
            if batch.usage.input_tokens is not None:
                input_tokens += batch.usage.input_tokens
                counted_tokens = True
            if batch.usage.output_tokens is not None:
                output_tokens += batch.usage.output_tokens
                counted_tokens = True
        return InferenceUsage(
            requests=requests,
            input_tokens=input_tokens if counted_tokens else None,
            output_tokens=output_tokens if counted_tokens else None,
        )

    async def _dispatch_window(self) -> None:
        """Own one coalescing window: wait, then dispatch every waiting text."""

        try:
            if self.coalesce_ms > 0:
                await asyncio.sleep(self.coalesce_ms)
            async with self._window_lock:
                waiters = tuple(self._window)
                self._window.clear()
                self._window_task = None

            outstanding: list[str] = []
            for waiter in waiters:
                for text in waiter.texts:
                    if text in self._cache or text in self._pending:
                        continue
                    if text not in outstanding:
                        outstanding.append(text)

            groups = self._split(outstanding)
            results = await asyncio.gather(*(self._embed_group(group) for group in groups))

            # Map each text to the batch that produced it, so every waiter can
            # account for exactly the batches it waited for.
            by_text: dict[str, _Batch] = {}
            for group, (_vectors, usage) in zip(groups, results, strict=True):
                batch = _Batch(tuple(group), usage)
                for text in group:
                    by_text[text] = batch

            for waiter in waiters:
                waiter.batches = []
                for text in dict.fromkeys(waiter.texts):
                    if text in by_text:
                        waiter.batches.append(by_text[text])
                if not waiter.future.done():
                    waiter.future.set_result(tuple(self._cache.get(text) or () for text in waiter.texts))
        except BaseException as error:  # noqa: BLE001 - fail every waiter in the window
            for waiter in waiters:
                if not waiter.future.done():
                    waiter.future.set_exception(error)
            raise

    def _split(self, texts: list[str]) -> list[list[str]]:
        size = self.max_batch_size
        return [texts[index : index + size] for index in range(0, len(texts), size)]

    async def _embed_group(self, group: list[str]) -> _GroupResult:
        """Embed one batch, joining an in-flight request when it already exists."""

        mine: list[str] = []
        joined: list[asyncio.Future[_GroupResult]] = []
        for text in group:
            future = self._pending.get(text)
            if future is None:
                mine.append(text)
            else:
                joined.append(future)

        vectors: list[tuple[float, ...]] = []
        requests = 0
        input_tokens = 0
        output_tokens = 0
        counted_tokens = False

        if joined:
            for other_vectors, other_usage in await asyncio.gather(*joined):
                vectors.extend(other_vectors)
                requests += other_usage.requests
                if other_usage.input_tokens is not None:
                    input_tokens += other_usage.input_tokens
                    counted_tokens = True
                if other_usage.output_tokens is not None:
                    output_tokens += other_usage.output_tokens
                    counted_tokens = True

        if mine:
            batch = tuple(mine)
            loop = asyncio.get_running_loop()
            for text in mine:
                self._pending[text] = loop.create_future()
            try:
                result = await self._model.embed(batch)
            except BaseException as error:  # noqa: BLE001 - forwarded to every waiter
                for text in mine:
                    self._pending.pop(text, None).set_exception(error)
                raise
            if len(result.vectors) != len(batch):
                error = ValueError("embedding vector count does not match the batch size")  # noqa: TRY003
                for text in mine:
                    self._pending.pop(text, None).set_exception(error)
                raise error
            for text in mine:
                self._pending.pop(text, None).set_result((result.vectors, result.usage))
            vectors.extend(result.vectors)
            requests += result.usage.requests
            if result.usage.input_tokens is not None:
                input_tokens += result.usage.input_tokens
                counted_tokens = True
            if result.usage.output_tokens is not None:
                output_tokens += result.usage.output_tokens
                counted_tokens = True
            for text, vector in zip(mine, result.vectors, strict=True):
                self._store(text, vector)

        return tuple(vectors), InferenceUsage(
            requests=requests,
            input_tokens=input_tokens if counted_tokens else None,
            output_tokens=output_tokens if counted_tokens else None,
        )

    def _store(self, text: str, vector: tuple[float, ...]) -> None:
        if not self.cache_size:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


__all__ = ["CoalescingEmbeddingModel"]
