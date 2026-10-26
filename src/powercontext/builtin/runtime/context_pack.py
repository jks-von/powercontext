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

"""Chunked Context Packs and prefix-reuse deltas for incremental context transfer.

A :class:`ContextPack` is the complete prepared-context body split into bounded,
valid UTF-8 chunks and sealed by a SHA-256 digest. A :class:`ContextPackDelta`
tells a receiver that already holds a *base* pack how many leading chunks to keep
and which chunks to append so that it reconstructs the *target* pack exactly.

The wire format is the interoperability contract: canonical JSON (UTF-8, sorted
keys, ``,``/``:`` separators, no extra whitespace) with chunks encoded as standard
Base64 strings. Decoding relies only on the serialized bytes, never on sender state.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

# Content-defined chunk boundaries. Cutting after every newline, before every
# prepared-context item, and before the citation fields that change between Memory
# revisions or entries keeps unchanged leading regions (instructions, envelope,
# leading items) byte-identical across turns. Boundaries depend only on local bytes,
# so equal content yields equal chunks and a shared prefix survives at chunk
# granularity, which is what a prefix-reuse delta can exploit.
_BOUNDARY_PATTERN = re.compile(rb'\n|(?=\{"citation":)|(?="revision":)|(?="entry_id":)')


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta violates the protocol or chunk contract."""


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )


def _encode_item(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_item(value: object, *, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError(f"{field_name} entries must be Base64 strings")  # noqa: TRY003
    try:
        raw = value.encode("ascii")
        decoded = base64.b64decode(raw, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ContextPackError(f"{field_name} contains invalid Base64") from error  # noqa: TRY003
    if base64.b64encode(decoded) != raw:
        raise ContextPackError(f"{field_name} contains non-canonical Base64")  # noqa: TRY003
    return decoded


def _coerce_items(items: Iterable[object], *, field_name: str) -> tuple[bytes, ...]:
    if isinstance(items, (str, bytes, bytearray, memoryview)):
        raise ContextPackError(f"{field_name} must be a sequence of bytes chunks")  # noqa: TRY003
    try:
        values = tuple(items)
    except TypeError as error:
        raise ContextPackError(f"{field_name} must be a sequence of bytes chunks") from error  # noqa: TRY003
    coerced: list[bytes] = []
    for item in values:
        if isinstance(item, bytes):
            coerced.append(item)
        elif isinstance(item, (bytearray, memoryview)):
            coerced.append(bytes(item))
        else:
            raise ContextPackError(f"{field_name} entries must be bytes")  # noqa: TRY003
    return tuple(coerced)


def _validate_digest(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContextPackError(f"{field_name} must be sha256:<64 lowercase hex>")  # noqa: TRY003
    return value


def _validate_prefix(value: object) -> int:
    if type(value) is not int:
        raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
    if value < 0:
        raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContextPackError(f"duplicate field {key!r}")  # noqa: TRY003
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ContextPackError(f"invalid JSON constant {value}")  # noqa: TRY003


def _load_object(data: object, *, kind: str, fields: frozenset[str]) -> dict[str, Any]:
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise ContextPackError(f"{kind} wire data must be bytes")  # noqa: TRY003
    try:
        text = data.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except ContextPackError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ContextPackError(f"{kind} wire data is not valid UTF-8 JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError(f"{kind} wire data must be a JSON object")  # noqa: TRY003
    keys = frozenset(payload)
    if keys != fields:
        missing = sorted(fields - keys)
        extra = sorted(keys - fields)
        raise ContextPackError(f"{kind} fields mismatch (missing={missing}, extra={extra})")  # noqa: TRY003
    return payload


def _digest_items(items: Sequence[bytes]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": [_encode_item(item) for item in items]}
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _utf8_char_start(data: bytes, index: int) -> bool:
    return index >= len(data) or (data[index] & 0xC0) != 0x80


def _split_bounded(segment: bytes, limit: int) -> list[bytes]:
    """Split one boundary segment into chunks of at most ``limit`` bytes on UTF-8 starts."""
    chunks: list[bytes] = []
    start = 0
    length = len(segment)
    while start < length:
        end = min(start + limit, length)
        while end > start and not _utf8_char_start(segment, end):
            end -= 1
        if end == start:
            raise ContextPackError(f"chunk limit {limit} cannot hold a complete UTF-8 character")  # noqa: TRY003
        chunks.append(segment[start:end])
        start = end
    return chunks


def _content_bytes(content: object) -> bytes:
    if content is None:
        return b""
    if isinstance(content, str):
        try:
            return content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContextPackError("content is not encodable as UTF-8") from error  # noqa: TRY003
    if isinstance(content, (bytes, bytearray, memoryview)):
        data = bytes(content)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextPackError("content bytes are not valid UTF-8") from error  # noqa: TRY003
        return data
    raise ContextPackError("content must be text")  # noqa: TRY003


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete, chunked prepared-context body with its digest and wire size."""

    items: tuple[bytes, ...]
    _digest: str = field(init=False, repr=False, compare=False)
    _wire: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        items = _coerce_items(self.items, field_name="items")
        object.__setattr__(self, "items", items)
        digest = _digest_items(items)
        object.__setattr__(self, "_digest", digest)
        object.__setattr__(self, "_wire", self._serialize(items, digest))

    @staticmethod
    def _serialize(items: tuple[bytes, ...], digest: str) -> bytes:
        encoded = [_encode_item(item) for item in items]
        count = 0
        # byte_count includes its own decimal digits; iterate to the fixed point.
        for _ in range(8):
            wire = _canonical_json({
                "schema": CONTEXT_PACK_SCHEMA,
                "items": encoded,
                "digest": digest,
                "byte_count": count,
            })
            if len(wire) == count:
                return wire
            count = len(wire)
        raise ContextPackError("byte_count did not converge")  # noqa: TRY003  # pragma: no cover

    @classmethod
    def build(cls, items: Iterable[bytes]) -> ContextPack:
        """Build a complete pack from an ordered tuple of byte chunks."""
        return cls(_coerce_items(items, field_name="items"))

    @classmethod
    def empty(cls) -> ContextPack:
        return cls(())

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Deterministically chunk ``content`` into bounded valid UTF-8 chunks.

        Boundaries fall after newlines and before prepared-context items, and any
        segment longer than ``chunk_bytes`` is cut at the last UTF-8 character start
        that fits. Joining the chunks reproduces the original bytes exactly.
        """
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        data = _content_bytes(content)
        if not data:
            return cls(())
        chunks: list[bytes] = []
        start = 0
        for match in _BOUNDARY_PATTERN.finditer(data):
            end = match.end()
            if end > start:
                chunks.extend(_split_bounded(data[start:end], chunk_bytes))
                start = end
        if start < len(data):
            chunks.extend(_split_bounded(data[start:], chunk_bytes))
        for chunk in chunks:
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError as error:  # pragma: no cover - guarded by boundary selection
                raise ContextPackError("chunk is not valid UTF-8") from error  # noqa: TRY003
        return cls(tuple(chunks))

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def byte_count(self) -> int:
        return len(self._wire)

    @property
    def content(self) -> bytes:
        """The reconstructed body bytes."""
        return b"".join(self.items)

    def text(self) -> str:
        """The reconstructed body as text."""
        try:
            return self.content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextPackError("pack content is not valid UTF-8") from error  # noqa: TRY003

    def to_bytes(self) -> bytes:
        return self._wire

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and verify a complete pack: fields, schema, Base64, digest and byte count."""
        payload = _load_object(data, kind="context pack", fields=_PACK_FIELDS)
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")  # noqa: TRY003
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise ContextPackError("items must be an array")  # noqa: TRY003
        digest = _validate_digest(payload["digest"], field_name="digest")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ContextPackError("byte_count must be a non-negative integer")  # noqa: TRY003
        pack = cls(tuple(_decode_item(value, field_name="items") for value in raw_items))
        if pack.digest != digest:
            raise ContextPackError("context pack digest mismatch")  # noqa: TRY003
        if pack.byte_count != byte_count:
            raise ContextPackError("context pack byte_count mismatch")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Keep ``prefix_items`` chunks of the base pack, then append ``appended``."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]
    _wire: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_digest(self.base_digest, field_name="base_digest")
        _validate_digest(self.target_digest, field_name="target_digest")
        _validate_prefix(self.prefix_items)
        appended = _coerce_items(self.appended, field_name="appended")
        object.__setattr__(self, "appended", appended)
        wire = _canonical_json({
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode_item(item) for item in appended],
        })
        object.__setattr__(self, "_wire", wire)

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Describe ``target`` relative to ``base`` using their longest common chunk prefix."""
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be ContextPack instances")  # noqa: TRY003
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self._wire)

    def to_bytes(self) -> bytes:
        return self._wire

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and structurally verify a delta; base and target checks happen on apply."""
        payload = _load_object(data, kind="context pack delta", fields=_DELTA_FIELDS)
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")  # noqa: TRY003
        raw_appended = payload["appended"]
        if not isinstance(raw_appended, list):
            raise ContextPackError("appended must be an array")  # noqa: TRY003
        return cls(
            _validate_digest(payload["base_digest"], field_name="base_digest"),
            _validate_digest(payload["target_digest"], field_name="target_digest"),
            _validate_prefix(payload["prefix_items"]),
            tuple(_decode_item(value, field_name="appended") for value in raw_appended),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Verify ``delta`` against ``base`` and return the reconstructed target pack."""
    if not isinstance(base, ContextPack):
        raise ContextPackError("base must be a ContextPack")  # noqa: TRY003
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta must be a ContextPackDelta")  # noqa: TRY003
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base_digest does not match the base pack")  # noqa: TRY003
    prefix = _validate_prefix(delta.prefix_items)
    if prefix > len(base.items):
        raise ContextPackError("delta prefix_items exceeds the base pack")  # noqa: TRY003
    target = ContextPack((*base.items[:prefix], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("reconstructed pack does not match target_digest")  # noqa: TRY003
    return target
