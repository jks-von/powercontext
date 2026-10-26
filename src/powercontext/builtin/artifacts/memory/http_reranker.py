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

"""HTTP scoring-service adapter for Memory reranking.

The adapter sends the coarse candidate pool to an HTTP cross-encoder style scoring
service in bounded batches, validates every response strictly, and then performs a
single global, stable selection over the whole pool: score descending, ties broken by
the original one-based global rank ascending. Candidate identity is never rewritten:
the decision only carries original global ranks, which the Memory service maps back to
the exact ``MemoryHit`` (entry ID and entry version ID) it supplied.

Wire protocol (one POST per batch)::

    request:  {"query": str, "candidates": [{"rank": int, "text": str}, ...]}
    response: {"scores": [{"rank": int, "score": finite number}, ...]}

``rank`` is always the candidate's one-based position in the *full* input, never a
batch-local position.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

if TYPE_CHECKING:
    import httpx

HTTP_MEMORY_RERANK_POLICY_ID = "http-cross-encoder-v1"

_RESPONSE_KEYS = frozenset({"scores"})
_SCORE_KEYS = frozenset({"rank", "score"})


class HTTPMemoryReranker:
    """Select Memory hits with scores returned by an HTTP scoring service.

    The caller owns ``client`` (an ``httpx.AsyncClient``); this adapter never opens,
    closes, or reconfigures it. Non-2xx responses and transport failures propagate as
    the original ``httpx`` exceptions; malformed JSON or schema violations raise
    ``ValueError``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = HTTP_MEMORY_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("HTTP reranker endpoint must be a non-empty string")  # noqa: TRY003
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("HTTP reranker batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("HTTP reranker policy_id must be a non-empty string")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    @property
    def endpoint(self) -> str:
        """The scoring-service URL receiving each batch."""

        return self._endpoint

    @property
    def batch_size(self) -> int:
        """The maximum number of candidates sent in one HTTP request."""

        return self._batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate in bounded batches and select the global top ``limit``."""

        candidate_tuple = tuple(candidates)
        _validate_bounds(len(candidate_tuple), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidate_tuple), self._batch_size):
            batch_ranks = tuple(range(start + 1, min(start + self._batch_size, len(candidate_tuple)) + 1))
            payload = {
                "query": query,
                "candidates": [{"rank": rank, "text": candidate_tuple[rank - 1].text} for rank in batch_ranks],
            }
            requests += 1
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            batch_scores = _parse_batch_scores(response.content, batch_ranks)
            for rank, score in batch_scores.items():
                if rank in scores:  # Defensive: ranks are partitioned across batches.
                    raise ValueError(f"rerank score for rank {rank} was returned twice")  # noqa: TRY003
                scores[rank] = score
        if len(scores) != len(candidate_tuple):
            raise ValueError("rerank scores do not cover every candidate exactly once")  # noqa: TRY003
        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_batch_scores(content: bytes, batch_ranks: Sequence[int]) -> dict[int, float]:
    try:
        body = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("rerank response is not valid JSON") from error  # noqa: TRY003
    if not isinstance(body, Mapping) or frozenset(body) != _RESPONSE_KEYS:
        raise ValueError("rerank response must be an object with exactly one 'scores' field")  # noqa: TRY003
    items = body["scores"]
    if not isinstance(items, list):
        raise ValueError("rerank response 'scores' must be a list")  # noqa: TRY003, TRY004
    expected = frozenset(batch_ranks)
    parsed: dict[int, float] = {}
    for item in items:
        if not isinstance(item, Mapping) or frozenset(item) != _SCORE_KEYS:
            raise ValueError("each rerank score must be an object with exactly 'rank' and 'score'")  # noqa: TRY003
        rank = _parse_rank(item["rank"])
        if rank not in expected:
            raise ValueError(f"rerank response returned rank {rank} outside this batch")  # noqa: TRY003
        if rank in parsed:
            raise ValueError(f"rerank response returned rank {rank} more than once")  # noqa: TRY003
        parsed[rank] = _parse_score(item["score"])
    if len(parsed) != len(expected):
        raise ValueError("rerank response does not cover every rank in the batch")  # noqa: TRY003
    return parsed


def _parse_rank(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("rerank score 'rank' must be an integer")  # noqa: TRY003, TRY004
    return value


def _parse_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("rerank score 'score' must be a number")  # noqa: TRY003, TRY004
    try:
        score = float(value)
    except OverflowError as error:
        raise ValueError("rerank score 'score' must be finite") from error  # noqa: TRY003
    if not math.isfinite(score):
        raise ValueError("rerank score 'score' must be finite")  # noqa: TRY003
    return score


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["HTTP_MEMORY_RERANK_POLICY_ID", "HTTPMemoryReranker"]
