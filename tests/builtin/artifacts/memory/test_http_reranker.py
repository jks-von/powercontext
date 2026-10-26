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

# ruff: noqa: S311 - deterministic test fixtures, not cryptography.

from __future__ import annotations

import asyncio
import json
import math
import random
from typing import Any

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryEntryInput, MemoryReranker
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts

ENDPOINT = "https://reranker.invalid/score"


def _hit(rank: int, text: str | None = None) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"e{rank}",
        entry_version_id=f"v{rank}",
        text=text if text is not None else f"candidate {rank}",
        score=1.0 / rank,
        matched_by=("fts",),
    )


def _run(coro):
    return asyncio.run(coro)


def _scoring_handler(scores: dict[int, float], seen: list[dict[str, Any]], *, shuffle: random.Random | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert set(body) == {"query", "candidates"}
        for item in body["candidates"]:
            assert set(item) == {"rank", "text"}
        seen.append(body)
        items = [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]
        if shuffle is not None:
            shuffle.shuffle(items)
        return httpx.Response(200, json={"scores": items})

    return handler


def test_satisfies_reranker_protocol_and_defaults() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client)
            assert reranker.policy_id == "http-cross-encoder-v1"
            assert reranker.batch_size == 32
            protocol_view: MemoryReranker = reranker
            assert protocol_view is reranker
            custom = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=3, policy_id="custom")
            assert custom.policy_id == "custom"

    _run(scenario())


def test_synthetic_example_from_spec_selects_rank_three_then_two() -> None:
    texts = ["数据库备份计划", "客户端连接超时参数说明", "连接池耗尽排查记录", "网络连通性检查方法"]
    scores = {1: 0.10, 2: 0.90, 3: 0.95, 4: 0.90}
    seen: list[dict[str, Any]] = []

    async def scenario() -> None:
        hits = tuple(_hit(rank, text) for rank, text in enumerate(texts, start=1))
        async with httpx.AsyncClient(transport=httpx.MockTransport(_scoring_handler(scores, seen))) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2).rerank(
                "数据库连接超时怎么排查", hits, 2
            )
        assert decision.selected_ranks == (3, 2)
        assert decision.usage.requests == 2
        assert [[c["rank"] for c in body["candidates"]] for body in seen] == [[1, 2], [3, 4]]
        for body in seen:
            assert body["query"] == "数据库连接超时怎么排查"
            for item in body["candidates"]:
                assert item["text"] == texts[item["rank"] - 1]

    _run(scenario())


@pytest.mark.parametrize(("count", "batch_size"), [(1, 1), (5, 5), (6, 5), (173, 7), (10, 32), (64, 32)])
def test_request_count_and_exact_coverage(count: int, batch_size: int) -> None:
    rng = random.Random(count * 31 + batch_size)
    scores = {rank: rng.random() for rank in range(1, count + 1)}
    seen: list[dict[str, Any]] = []

    async def scenario() -> None:
        hits = tuple(_hit(rank) for rank in range(1, count + 1))
        handler = _scoring_handler(scores, seen, shuffle=rng)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=batch_size).rerank(
                "q", hits, min(10, count)
            )
        assert decision.usage.requests == math.ceil(count / batch_size) == len(seen)
        assert all(1 <= len(body["candidates"]) <= batch_size for body in seen)
        sent = sorted(item["rank"] for body in seen for item in body["candidates"])
        assert sent == list(range(1, count + 1))
        expected = tuple(sorted(scores, key=lambda r: (-scores[r], r))[: min(10, count)])
        assert decision.selected_ranks == expected
        assert not decision.used_fallback

    _run(scenario())


def test_cross_batch_ties_use_original_global_rank() -> None:
    scores = {1: 0.5, 2: 0.7, 3: 0.7, 4: 0.7, 5: 0.9, 6: 0.7, 7: 0.1}
    seen: list[dict[str, Any]] = []

    async def scenario() -> None:
        hits = tuple(_hit(rank) for rank in range(1, 8))
        handler = _scoring_handler(scores, seen, shuffle=random.Random(7))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2).rerank("q", hits, 7)
        assert decision.selected_ranks == (5, 2, 3, 4, 6, 1, 7)
        assert decision.usage.requests == 4

    _run(scenario())


def test_integer_scores_and_negative_scores_are_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"scores": [{"rank": item["rank"], "score": -item["rank"]} for item in body["candidates"]]}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2).rerank(
                "q", tuple(_hit(r) for r in range(1, 4)), 3
            )
        assert decision.selected_ranks == (1, 2, 3)

    _run(scenario())


def _single_response(payload: object) -> httpx.MockTransport:
    content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return httpx.MockTransport(lambda _request: httpx.Response(200, content=content))


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"\xff\xfe",
        [],
        {},
        {"scores": None},
        {"scores": {"rank": 1, "score": 0.5}},
        {"scores": [], "extra": True},
        {"scores": []},
        {"scores": [{"rank": 2, "score": 0.5}]},
        {"scores": [{"rank": 0, "score": 0.5}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.4}]},
        {"scores": [{"rank": 1, "score": float("nan")}]},
        {"scores": [{"rank": 1, "score": float("inf")}]},
        {"scores": [{"rank": 1, "score": float("-inf")}]},
        b'{"scores":[{"rank":1,"score":1e999}]}',
        {"scores": [{"rank": 1, "score": 0.5, "extra": True}]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"score": 0.5}]},
        {"scores": [{"rank": "1", "score": 0.5}]},
        {"scores": [{"rank": 1.0, "score": 0.5}]},
        {"scores": [{"rank": True, "score": 0.5}]},
        {"scores": [{"rank": 1, "score": "0.5"}]},
        {"scores": [{"rank": 1, "score": True}]},
        {"scores": [{"rank": 1, "score": None}]},
        {"scores": [1]},
    ],
)
def test_rejects_invalid_json_or_schema(payload: object) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(transport=_single_response(payload)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("q", (_hit(1),), 1)

    _run(scenario())


def test_rejects_cross_batch_rank_in_second_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        ranks = [item["rank"] for item in body["candidates"]]
        if ranks == [3]:
            return httpx.Response(200, json={"scores": [{"rank": 1, "score": 0.9}]})
        return httpx.Response(200, json={"scores": [{"rank": r, "score": 0.1} for r in ranks]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2).rerank(
                    "q", tuple(_hit(r) for r in range(1, 4)), 1
                )

    _run(scenario())


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_non_2xx_raises_httpx_status_error(status: int) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(status, json={"scores": []}))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("q", (_hit(1),), 1)
            assert caught.value.response.status_code == status

    _run(scenario())


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_transport_errors_propagate_unchanged(error_type: type[httpx.TransportError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("offline", request=request)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(error_type):
                await HTTPMemoryReranker(ENDPOINT, client=client).rerank("q", (_hit(1),), 1)

    _run(scenario())


def test_parameter_bounds() -> None:
    async def scenario() -> None:
        transport = _single_response({"scores": [{"rank": 1, "score": 0.5}]})
        async with httpx.AsyncClient(transport=transport) as client:
            bad_sizes: tuple[Any, ...] = (0, -1, True, 1.5)
            for bad in bad_sizes:
                with pytest.raises(ValueError):
                    HTTPMemoryReranker(ENDPOINT, client=client, batch_size=bad)
            reranker = HTTPMemoryReranker(ENDPOINT, client=client)
            with pytest.raises(ValueError):
                await reranker.rerank("q", (), 1)
            for limit in (0, -1, 2):
                with pytest.raises(ValueError):
                    await reranker.rerank("q", (_hit(1),), limit)

    _run(scenario())


def test_simulated_scifact_quality_shape_matches_global_cosine_order() -> None:
    """Mirror the formal gate: 173 docs, 20 queries, 384-d vectors, batch_size=7, top 10."""

    rng = random.Random(20260)
    dim, doc_count = 384, 173

    def unit(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    docs = [unit([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(doc_count)]
    texts = [f"abstract {i}" for i in range(doc_count)]
    text_to_doc = {text: i for i, text in enumerate(texts)}

    async def scenario() -> None:
        recall_total = 0.0
        mrr_total = 0.0
        for q in range(20):
            relevant = rng.sample(range(doc_count), k=rng.choice((1, 2, 3)))
            base = [sum(docs[r][d] for r in relevant) for d in range(dim)]
            query_vec = unit([x + 0.3 * rng.gauss(0, 1) / math.sqrt(dim) for x in base])
            order = list(range(doc_count))
            rng.shuffle(order)  # coarse order differs from service order
            hits = tuple(_hit(rank, texts[order[rank - 1]]) for rank in range(1, doc_count + 1))
            seen: list[dict[str, Any]] = []

            def handler(request: httpx.Request, *, _qv=query_vec, _seen=seen) -> httpx.Response:
                body = json.loads(request.content)
                _seen.append(body)
                items = [
                    {
                        "rank": item["rank"],
                        "score": sum(a * b for a, b in zip(_qv, docs[text_to_doc[item["text"]]], strict=True)),
                    }
                    for item in body["candidates"]
                ]
                rng.shuffle(items)
                return httpx.Response(200, json={"scores": items})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                decision = await HTTPMemoryReranker(ENDPOINT, client=client, batch_size=7).rerank(
                    f"claim {q}", hits, 10
                )
            assert decision.usage.requests == math.ceil(doc_count / 7) == len(seen)
            cos = {
                rank: sum(a * b for a, b in zip(query_vec, docs[order[rank - 1]], strict=True))
                for rank in range(1, doc_count + 1)
            }
            expected = tuple(sorted(cos, key=lambda r: (-cos[r], r))[:10])
            assert decision.selected_ranks == expected
            selected_docs = [order[rank - 1] for rank in decision.selected_ranks]
            recall_total += len(set(selected_docs) & set(relevant)) / len(relevant)
            first = next((i for i, d in enumerate(selected_docs, start=1) if d in relevant), None)
            mrr_total += 0.0 if first is None else 1.0 / first
        assert recall_total / 20 >= 0.95
        assert mrr_total / 20 >= 0.80

    _run(scenario())


def test_native_memory_search_uses_service_scores_and_keeps_identity() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        scores = [
            {"rank": item["rank"], "score": 1.0 if "gamma" in item["text"] else 0.0} for item in body["candidates"]
        ]
        return httpx.Response(200, json={"scores": scores})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(ENDPOINT, client=client, batch_size=2)
            async with open_builtin_contexts(
                BuiltinConfig(database=SQLiteConfig()), memory_reranker=reranker
            ) as contexts:
                service = (await contexts.get("http-reranker-native")).artifacts.memory
                memory = await service.remember(
                    memory=None,
                    entries=(
                        MemoryEntryInput(kind="fact", text="Project alpha fact."),
                        MemoryEntryInput(kind="fact", text="Project beta fact."),
                        MemoryEntryInput(kind="fact", text="Project gamma fact."),
                    ),
                    mode="append",
                )
                assert memory is not None
                result = await service.search("project", memories=(memory,), limit=1, mode="fts")
        assert result.rerank is not None
        assert result.rerank.policy_id == "http-cross-encoder-v1"
        candidates = result.rerank.candidate_hits
        assert len(candidates) == 3
        assert result.rerank.usage.requests == math.ceil(len(candidates) / 2) == len(seen)
        (rank,) = result.rerank.selected_ranks
        chosen = result.hits[0]
        assert chosen == candidates[rank - 1]
        assert "gamma" in chosen.text
        assert chosen.entry_id == candidates[rank - 1].entry_id
        assert chosen.entry_version_id == candidates[rank - 1].entry_version_id

    _run(scenario())
