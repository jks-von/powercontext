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

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import random
from typing import Any, cast

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, PrepareContextRequest, open_builtin_runtime
from powercontext.builtin.runtime.context_pack import (
    CONTEXT_PACK_DELTA_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _oracle_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": [base64.b64encode(item).decode() for item in items]}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


# --- wire format -----------------------------------------------------------------


@pytest.mark.parametrize(
    "items", [(), (b"",), (b"alpha\n", "\u6d77\u6d0b".encode()), tuple(b"x" * n for n in range(40))]
)
def test_pack_wire_matches_independent_oracle(items: tuple[bytes, ...]) -> None:
    pack = ContextPack.build(items)
    wire = pack.to_bytes()
    payload = json.loads(wire)
    assert set(payload) == {"schema", "items", "digest", "byte_count"}
    assert payload["schema"] == CONTEXT_PACK_SCHEMA
    assert payload["items"] == [base64.b64encode(item).decode() for item in items]
    assert payload["digest"] == pack.digest == _oracle_digest(items)
    assert payload["byte_count"] == pack.byte_count == len(wire)
    assert wire == _canonical(payload)
    assert ContextPack.from_bytes(wire) == pack


def test_byte_count_fixed_point_across_digit_boundaries() -> None:
    for size in range(0, 400):
        pack = ContextPack.build((b"a" * size,))
        assert json.loads(pack.to_bytes())["byte_count"] == len(pack.to_bytes()) == pack.byte_count


def test_delta_wire_matches_contract() -> None:
    base = ContextPack.build((b"one", b"two"))
    target = ContextPack.build((b"one", b"three", b"four"))
    delta = ContextPackDelta.between(base, target)
    payload = json.loads(delta.to_bytes())
    assert payload == {
        "schema": CONTEXT_PACK_DELTA_SCHEMA,
        "base_digest": base.digest,
        "target_digest": target.digest,
        "prefix_items": 1,
        "appended": [base64.b64encode(b"three").decode(), base64.b64encode(b"four").decode()],
    }
    assert delta.to_bytes() == _canonical(payload)
    assert delta.byte_count == len(delta.to_bytes())
    assert "byte_count" not in payload
    assert ContextPackDelta.from_bytes(delta.to_bytes()) == delta
    assert isinstance(delta.appended, tuple)


def test_non_canonical_but_valid_json_is_accepted() -> None:
    pack = ContextPack.build((b"alpha", b"beta"))
    assert ContextPack.from_bytes(json.dumps(json.loads(pack.to_bytes()), indent=2).encode()) == pack
    delta = ContextPackDelta.between(pack, ContextPack.build((b"alpha",)))
    assert ContextPackDelta.from_bytes(json.dumps(json.loads(delta.to_bytes())).encode()) == delta


# --- chunking ----------------------------------------------------------------------


def test_from_content_is_bounded_valid_and_lossless() -> None:
    rng = random.Random(7)  # noqa: S311
    alphabet = ["a", " ", "\n", "\u6d77", "\u00e9", "\U0001f600", '{"citation":', '"revision":', '"entry_id":']
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300)))
        limit = rng.randint(4, 64)
        pack = ContextPack.from_content(text, chunk_bytes=limit)
        assert b"".join(pack.items) == text.encode()
        for chunk in pack.items:
            assert 0 < len(chunk) <= limit
            chunk.decode("utf-8")
        assert ContextPack.from_content(text, chunk_bytes=limit) == pack


def test_from_content_default_limit_is_512() -> None:
    pack = ContextPack.from_content("x" * 2000)
    assert max(len(item) for item in pack.items) == 512
    assert b"".join(pack.items) == b"x" * 2000


def test_from_content_empty_and_none_are_the_empty_pack() -> None:
    assert ContextPack.from_content("") == ContextPack.build(())
    assert ContextPack.from_content(None) == ContextPack.build(())


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_from_content_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("abc", chunk_bytes=cast(Any, limit))


def test_from_content_rejects_limit_below_character_width() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("\u6d77", chunk_bytes=2)
    assert ContextPack.from_content("\u6d77", chunk_bytes=3).items == ("\u6d77".encode(),)


def test_equal_regions_produce_equal_leading_chunks() -> None:
    header = "Header line\nSecond line\n"
    first = ContextPack.from_content(header + '{"items":[{"citation":{"revision":1},"content":"a"}]}')
    second = ContextPack.from_content(header + '{"items":[{"citation":{"revision":2},"content":"a"}]}')
    delta = ContextPackDelta.between(first, second)
    kept = b"".join(first.items[: delta.prefix_items])
    assert kept.startswith(header.encode())
    assert apply_context_pack_delta(first, delta) == second


# --- reconstruction -----------------------------------------------------------------


@pytest.mark.parametrize(
    "target_items",
    [
        (b"a", b"b", b"old"),
        (b"a", b"b", b"old", b"tail"),
        (b"new", b"b", b"old"),
        (b"a", b"new", b"old"),
        (b"a", b"b", b"new"),
        (b"a",),
        (),
    ],
)
def test_every_valid_prefix_reconstructs_target(target_items: tuple[bytes, ...]) -> None:
    base = ContextPack.build((b"a", b"b", b"old"))
    target = ContextPack.build(target_items)
    common = 0
    while common < min(len(base.items), len(target.items)) and base.items[common] == target.items[common]:
        common += 1
    for prefix in range(common + 1):
        delta = ContextPackDelta(base.digest, target.digest, prefix, target.items[prefix:])
        restored = apply_context_pack_delta(
            ContextPack.from_bytes(base.to_bytes()), ContextPackDelta.from_bytes(delta.to_bytes())
        )
        assert restored.to_bytes() == target.to_bytes()
    best = ContextPackDelta.between(base, target)
    assert best.prefix_items == common


def test_delta_from_empty_base_and_to_empty_target() -> None:
    empty = ContextPack.build(())
    full = ContextPack.from_content("hello\nworld")
    assert apply_context_pack_delta(empty, ContextPackDelta.between(empty, full)) == full
    assert apply_context_pack_delta(full, ContextPackDelta.between(full, empty)).items == ()


# --- rejection ----------------------------------------------------------------------


def _pack_payload() -> dict[str, object]:
    return json.loads(ContextPack.build((b"alpha", b"beta")).to_bytes())


def _delta_payload() -> dict[str, object]:
    base = ContextPack.build((b"alpha",))
    return json.loads(ContextPackDelta.between(base, ContextPack.build((b"alpha", b"beta"))).to_bytes())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("schema"),
        lambda p: p.pop("items"),
        lambda p: p.pop("digest"),
        lambda p: p.pop("byte_count"),
        lambda p: p.update(extra=1),
        lambda p: p.update(schema="powercontext.context-pack.v2"),
        lambda p: p.update(items="YWxwaGE="),
        lambda p: p.update(items=["not base64!"]),
        lambda p: p.update(items=["YWxwaGE"]),
        lambda p: p.update(items=[1]),
        lambda p: p.update(digest="sha256:" + "0" * 64),
        lambda p: p.update(digest="md5:abc"),
        lambda p: p.update(byte_count=p["byte_count"] + 1),  # type: ignore[operator]
        lambda p: p.update(byte_count=True),
        lambda p: p.update(byte_count=float(p["byte_count"])),  # type: ignore[arg-type]
    ],
)
def test_pack_decoder_rejects_invalid_payloads(mutate) -> None:
    payload = _pack_payload()
    mutate(payload)
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(_canonical(payload))


@pytest.mark.parametrize("data", [b"", b"[]", b"\xff\xfe", b"{", b'{"schema":1,"schema":2}', "x"])
def test_decoders_reject_malformed_bytes(data: object) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(cast(Any, data))
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(cast(Any, data))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("schema"),
        lambda p: p.pop("base_digest"),
        lambda p: p.pop("target_digest"),
        lambda p: p.pop("prefix_items"),
        lambda p: p.pop("appended"),
        lambda p: p.update(byte_count=10),
        lambda p: p.update(schema=CONTEXT_PACK_SCHEMA),
        lambda p: p.update(prefix_items=True),
        lambda p: p.update(prefix_items=False),
        lambda p: p.update(prefix_items=1.0),
        lambda p: p.update(prefix_items=-1),
        lambda p: p.update(prefix_items="1"),
        lambda p: p.update(appended=["%%%"]),
        lambda p: p.update(appended="YmV0YQ=="),
        lambda p: p.update(base_digest="sha256:XYZ"),
    ],
)
def test_delta_decoder_rejects_invalid_payloads(mutate) -> None:
    payload = _delta_payload()
    mutate(payload)
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(_canonical(payload))


def test_apply_rejects_wrong_base_prefix_out_of_range_and_wrong_target() -> None:
    base = ContextPack.build((b"one", b"two"))
    target = ContextPack.build((b"one", b"three"))
    delta = ContextPackDelta.between(base, target)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(ContextPack.build((b"other",)), delta)
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, target.digest, 3, ()))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, target.digest, 2, (b"three",)))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(base, ContextPackDelta(base.digest, "sha256:" + "0" * 64, 1, (b"three",)))


def test_constructor_rejects_invalid_prefix_types() -> None:
    digest = ContextPack.build(()).digest
    for prefix in (True, 1.5, -1, "0"):
        with pytest.raises(ContextPackError):
            ContextPackDelta(digest, digest, cast(Any, prefix), ())


# --- Runtime integration --------------------------------------------------------


class _CandidatePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="conversation-turn", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


_TURNS = (
    "Project Manager: Okay . What are your experiences with remote controls ? We had three videos and a TV set up.",
    "Project Manager: but that was quite good , the fact that you did not have six remote controls in front of you .",
    "Industrial Designer: the remote should have fewer buttons , big buttons for the channel and the volume .",
    "Marketing: the price of the remote matters , we want to sell it for twenty five Euro and keep the design simple .",
    "User Interface: the user wants a remote that is easy to find , maybe it beeps when you whistle for it .",
    "Industrial Designer: the case of the remote could be rubber so it survives being dropped on the floor .",
    "Marketing: the trend is a fancy look and feel , the remote should look sleek and modern for the user .",
    "Project Manager: so the remote needs a channel button , a volume button and a power button at minimum .",
    "User Interface: the remote could have a small screen showing the channel and the time for the user .",
    "Industrial Designer: a battery of the remote should last a long time and be easy to replace .",
)


@pytest.mark.asyncio
async def test_prepare_incremental_empty_scope_is_empty_pack() -> None:
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        assert runtime.scopes is not None
        scope = await runtime.scopes.create(ScopeDraft(title="Empty", summary="Empty", idempotency_key="pack-empty"))
        context = runtime.context.for_scope(scope.scope_id)
        request = PrepareContextRequest(query="remote", max_bytes=4096)
        prepared = await context.prepare(request)
        assert prepared.content is None
        first = await context.prepare_incremental(request)
        assert isinstance(first, ContextPack)
        assert first.items == ()
        again = await context.prepare_incremental(request, base_pack=ContextPack.from_bytes(first.to_bytes()))
        assert isinstance(again, ContextPackDelta)
        assert b"".join(apply_context_pack_delta(first, ContextPackDelta.from_bytes(again.to_bytes())).items) == b""


@pytest.mark.asyncio
async def test_prepare_incremental_sequence_rebuilds_prepare_and_saves_bytes() -> None:
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=_CandidatePipeline()) as runtime:
        assert runtime.scopes is not None
        scope = await runtime.scopes.create(
            ScopeDraft(title="Incremental", summary="Sequence", idempotency_key="pack-sequence")
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        context = runtime.context.for_scope(scope.scope_id)
        appended = 0

        async def append() -> None:
            nonlocal appended
            await sources.capture(CaptureSource(source_id=f"turn-{appended}", content=_TURNS[appended], metadata={}))
            appended += 1
            while (await memory.flush()).processed:
                pass

        for _ in range(8):
            await append()

        queries = ("the remote", "remote", "the user", "the remote")
        receiver: ContextPack | None = None
        raw_bytes = incremental_bytes = full_bytes = gzip_bytes = 0
        distinct: set[str] = set()
        for turn in range(40):
            if turn in (13, 27):
                await append()
            request = PrepareContextRequest(query=queries[(turn // 5) % len(queries)], max_bytes=4096)
            sent = await context.prepare_incremental(request, base_pack=receiver)
            expected = await context.prepare(request)
            assert expected.content
            wire = sent.to_bytes()
            incremental_bytes += len(wire)
            if receiver is None:
                assert isinstance(sent, ContextPack)
                receiver = ContextPack.from_bytes(wire)
            else:
                assert isinstance(sent, ContextPackDelta)
                receiver = apply_context_pack_delta(
                    ContextPack.from_bytes(receiver.to_bytes()), ContextPackDelta.from_bytes(wire)
                )
            assert b"".join(receiver.items).decode("utf-8") == expected.content
            content = expected.content.encode("utf-8")
            raw_bytes += len(content)
            full_bytes += ContextPack.from_content(expected.content).byte_count
            gzip_bytes += len(gzip.compress(content, compresslevel=9, mtime=0))
            distinct.add(expected.content)

    assert len(distinct) >= 4
    assert 1 - incremental_bytes / raw_bytes >= 0.45
    assert incremental_bytes < full_bytes
    assert gzip_bytes > 0
