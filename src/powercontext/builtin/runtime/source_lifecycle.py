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

"""Source-driven availability propagation over a Scope's evidence DAG.

A Scope's evidence forms a directed acyclic graph: Sources support Memory
entries, Memory entries support Topic Memory.  Withdrawing a Source must take
the content it uniquely supports out of search and context, while independent
facts keep working, and restoring it must bring them back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from powercontext.artifacts import Artifact, ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.family_management import _RepositoryFamilyWriter
from powercontext.sources.models import SourceRef

Policy = Literal["all", "any"]


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration or event violates the contract."""


class SourceLifecycleEvent(BaseModel):
    """One availability change for one registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int
    sequence: int
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("node_id and reason must be non-empty strings")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("revision and sequence must be positive integers")  # noqa: TRY003
        return value


@dataclass(frozen=True)
class _Binding:
    """One registered node and how its availability is derived."""

    node_id: str
    kind: Literal["source", "memory", "topic"]
    parents: tuple[str, ...]
    policy: Policy
    reference: Any


class SourceLifecycleContent(BaseModel):
    """The persisted evidence graph of one Scope."""

    model_config = ConfigDict(extra="forbid")

    generation: int
    bindings: list[dict[str, Any]]
    sources: dict[str, dict[str, Any]]


class SourceLifecycle(Artifact[SourceLifecycleContent]):
    """An immutable snapshot of one Scope's lifecycle graph."""

    family: ClassVar[str] = "source-lifecycle"


@dataclass
class _SourceState:
    revision: int
    available: bool
    last_sequence: int = 0


class _Graph:
    """The registered evidence graph and its computed availability."""

    def __init__(self) -> None:
        self.bindings: dict[str, _Binding] = {}
        self.order: list[str] = []
        self.sources: dict[str, _SourceState] = {}
        self.active: set[str] = set()
        self._generation = 0

    # -- registration -------------------------------------------------------

    def register(
        self,
        node_id: str,
        kind: Literal["source", "memory", "topic"],
        parents: tuple[str, ...],
        policy: Policy,
        reference: Any,
    ) -> None:
        if node_id in self.bindings:
            raise SourceLifecycleError(f"node {node_id!r} is already registered")
        if kind != "source":
            if not parents:
                raise SourceLifecycleError(f"node {node_id!r} must have at least one parent")
            if node_id in parents:
                raise SourceLifecycleError(f"node {node_id!r} cannot depend on itself")
            for parent in parents:
                if parent not in self.bindings:
                    raise SourceLifecycleError(f"parent {parent!r} of {node_id!r} is not registered")
            if len(set(parents)) != len(parents):
                raise SourceLifecycleError(f"node {node_id!r} repeats a parent")
        if kind == "source":
            self.sources[node_id] = _SourceState(revision=1, available=True)
        self.bindings[node_id] = _Binding(node_id, kind, tuple(parents), policy, reference)
        self.order.append(node_id)
        self._propagate()

    # -- events -------------------------------------------------------------

    def apply(self, event: SourceLifecycleEvent) -> list[str]:
        binding = self.bindings.get(event.node_id)
        if binding is None or binding.kind != "source":
            raise SourceLifecycleError(f"unknown Source {event.node_id!r}")
        state = self.sources[event.node_id]
        if state.revision != event.revision:
            raise SourceLifecycleError(
                f"event revision {event.revision} does not match {event.node_id!r} revision {state.revision}"
            )
        if event.sequence <= state.last_sequence:
            raise SourceLifecycleError(
                f"event sequence {event.sequence} does not advance {event.node_id!r} sequence {state.last_sequence}"
            )
        if state.last_sequence == event.sequence and state.available != event.available:
            raise SourceLifecycleError(f"event sequence {event.sequence} for {event.node_id!r} was already applied")
        state.last_sequence = event.sequence
        state.available = event.available
        self._propagate()
        return sorted(self.active)

    # -- availability -------------------------------------------------------

    def _propagate(self) -> None:
        active: set[str] = {node for node, state in self.sources.items() if state.available}
        changed = True
        # A node becomes available when its policy over its parents' current
        # availability holds; iterate until the fixed point so transitive
        # withdrawal and restoration both settle.
        while changed:
            changed = False
            for node_id in self.order:
                if node_id in active:
                    continue
                binding = self.bindings[node_id]
                if binding.kind == "source":
                    continue
                support = [parent in active for parent in binding.parents]
                satisfied = all(support) if binding.policy == "all" else any(support)
                if satisfied:
                    active.add(node_id)
                    changed = True
        self.active = active

    def is_active(self, node_id: str) -> bool:
        return node_id in self.active

    # -- persistence --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "generation": self._generation,
            "bindings": [
                {
                    "node_id": binding.node_id,
                    "kind": binding.kind,
                    "parents": list(binding.parents),
                    "policy": binding.policy,
                    "reference": _encode_reference(binding.reference),
                }
                for binding in (self.bindings[node] for node in self.order)
            ],
            "sources": {
                node: {
                    "revision": state.revision,
                    "available": state.available,
                    "last_sequence": state.last_sequence,
                }
                for node, state in self.sources.items()
            },
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        for record in snapshot.get("bindings", ()):
            self.register(
                record["node_id"],
                record["kind"],
                tuple(record["parents"]),
                record["policy"],
                _decode_reference(record.get("reference")),
            )
        for node, state in snapshot.get("sources", {}).items():
            if node in self.sources:
                self.sources[node] = _SourceState(
                    revision=state["revision"],
                    available=state["available"],
                    last_sequence=state.get("last_sequence", 0),
                )
        self._generation = int(snapshot.get("generation", 0))
        self._propagate()


def _encode_reference(reference: Any) -> dict[str, Any] | None:
    """Encode one binding reference so a restart can resolve it again."""

    if reference is None:
        return None
    if isinstance(reference, MemoryCitation):
        return {
            "kind": "memory",
            "memory_ref": reference.memory_ref.model_dump(mode="json"),
            "entry_id": reference.entry_id,
            "entry_version_id": reference.entry_version_id,
        }
    if isinstance(reference, ArtifactRef):
        return {"kind": "artifact", "artifact_ref": reference.model_dump(mode="json")}
    if isinstance(reference, SourceRef):
        return {"kind": "source", "source_ref": reference.model_dump(mode="json")}
    return None


def _decode_reference(payload: Any) -> Any:
    """Rebuild one binding reference from its persisted form."""

    if not isinstance(payload, dict):
        return None
    if payload.get("kind") == "memory":
        return MemoryCitation(
            memory_ref=ArtifactRef.model_validate(payload["memory_ref"]),
            entry_id=payload["entry_id"],
            entry_version_id=payload["entry_version_id"],
        )
    if payload.get("kind") == "artifact":
        return ArtifactRef.model_validate(payload["artifact_ref"])
    if payload.get("kind") == "source":
        return SourceRef.model_validate(payload["source_ref"])
    return None


class SourceLifecycleService:
    """One Scope's lifecycle: register evidence, apply events, resolve uses."""

    def __init__(self, runtime: Any, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._graph = _Graph()
        self._loaded = False

    # -- loading ------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        snapshot = await self._read_snapshot()
        if snapshot:
            self._graph.restore(snapshot)

    async def _read_snapshot(self) -> dict[str, Any] | None:
        records = self._runtime.records.for_scope(self.scope_id)
        page = await records.query_artifacts("source-lifecycle", limit=50, cursor=None)
        newest: dict[str, Any] | None = None
        for item in page.items:
            record = await records.get_artifact_revision(
                "source-lifecycle", item.artifact_id, item.revision
            )
            content = record.content
            if not isinstance(content, dict):
                continue
            if newest is None or int(content.get("generation", 0)) > int(newest.get("generation", 0)):
                newest = content
        return newest

    async def _write_snapshot(self) -> None:
        from powercontext.builtin.records import ArtifactWrite

        records = self._runtime.records.for_scope(self.scope_id)
        self._graph._generation += 1
        await records.create_artifact(
            "source-lifecycle",
            ArtifactWrite(content=self._graph.snapshot()),
        )

    # -- registration -------------------------------------------------------

    async def register_source(self, node_id: str, source_ref: Any, *, revision: int) -> None:
        await self._ensure_loaded()
        self._graph.register(node_id, "source", (), "all", source_ref)
        self._graph.sources[node_id] = _SourceState(revision=revision, available=True)
        self._graph._propagate()
        await self._write_snapshot()

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Policy,
    ) -> None:
        await self._ensure_loaded()
        self._graph.register(node_id, "memory", tuple(parents), policy, citation)
        await self._write_snapshot()
        # A node registered under withdrawn parents is unavailable from the
        # moment it is registered, not after the next event.
        await self._propagate_to_runtime()

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Policy,
    ) -> None:
        await self._ensure_loaded()
        self._graph.register(node_id, "topic", tuple(parents), policy, artifact_ref)
        await self._write_snapshot()

    # -- events -------------------------------------------------------------

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        await self._ensure_loaded()
        # Replay of an already applied event is idempotent.
        binding = self._graph.bindings.get(event.node_id)
        if binding is not None and binding.kind == "source":
            state = self._graph.sources[event.node_id]
            if state.last_sequence == event.sequence and state.available == event.available:
                return sorted(self._graph.active)
        active = self._graph.apply(event)
        await self._write_snapshot()
        await self._propagate_to_runtime()
        await self._propagate_topics()
        return active

    async def _propagate_to_runtime(self) -> None:
        """Reflect the computed availability on the Runtime's Memory entries.

        A stored citation names the revision it was created against, and both
        ``retire`` and ``reactivate`` commit new revisions, so the citation is
        re-anchored on the current head before every state change.
        """

        from powercontext.artifacts import MemoryCitation
        from powercontext.builtin.runtime.models import (
            ReactivateMemoryEntryRequest,
            RetireMemoryEntryRequest,
        )

        memory = self._runtime.memory.for_scope(self.scope_id)
        page = await memory.list(include_inactive=True)
        if page.memory_ref is None:
            return
        by_version = {
            (entry.entry.entry_id, entry.entry.entry_version_id): entry for entry in page.entries
        }
        for node_id in self._graph.order:
            binding = self._graph.bindings[node_id]
            if binding.kind != "memory":
                continue
            citation = binding.reference
            if citation is None:
                continue
            record = by_version.get((citation.entry_id, citation.entry_version_id))
            if record is None:
                continue
            current = MemoryCitation(
                memory_ref=page.memory_ref,
                entry_id=citation.entry_id,
                entry_version_id=citation.entry_version_id,
            )
            should_be_active = self._graph.is_active(node_id)
            if should_be_active and record.state != "active":
                await memory.reactivate(
                    ReactivateMemoryEntryRequest(citation=current, reason="source restored")
                )
            elif not should_be_active and record.state == "active":
                await memory.retire(RetireMemoryEntryRequest(citation=current, reason="source withdrawn"))
            page = await memory.list(include_inactive=True)
            if page.memory_ref is None:
                return
            by_version = {
                (entry.entry.entry_id, entry.entry.entry_version_id): entry for entry in page.entries
            }

    async def _propagate_topics(self) -> None:
        """Topic availability is enforced by search, not by deleting published data.

        A published Topic must keep its active projection, so a withdrawn Topic
        is filtered out of recall instead of being removed from storage.
        """

        return None

    async def is_topic_available(self, reference: ArtifactRef) -> bool:
        """Report whether one Topic is still supported by its registered parents."""

        await self._ensure_loaded()
        for node_id, binding in self._graph.bindings.items():
            if binding.kind != "topic":
                continue
            if binding.reference is None:
                continue
            if binding.reference.artifact_id != reference.artifact_id:
                continue
            return self._graph.is_active(node_id)
        return True

    # -- resolution ---------------------------------------------------------

    async def resolve(self, node_id: str, *, purpose: Literal["generation", "audit"]) -> Any:
        await self._ensure_loaded()
        binding = self._graph.bindings.get(node_id)
        if binding is None:
            raise SourceLifecycleError(f"unknown node {node_id!r}")
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError(f"unknown purpose {purpose!r}")
        if purpose == "generation" and not self._graph.is_active(node_id):
            raise SourceLifecycleError(f"node {node_id!r} is not available for generation")
        return binding.reference


class SourceLifecycleManagementWriter(_RepositoryFamilyWriter):
    """Family writer that persists one Scope's lifecycle graph snapshot."""

    family = "source-lifecycle"
    content_type = SourceLifecycleContent

    def artifact_id_for_create(self, generated: str, /) -> str:
        return generated

    async def create(
        self,
        connection: Any,
        scope_id: str,
        artifact_id: str,
        content: BaseModel,
        direct_source: Any,
        /,
    ) -> Artifact[Any]:
        validated = cast(SourceLifecycleContent, content)
        return await self._create_artifact(connection, scope_id, artifact_id, validated, direct_source)

    async def replace(
        self,
        connection: Any,
        scope_id: str,
        current: Artifact[Any],
        content: BaseModel,
        direct_source: Any,
        /,
    ) -> Artifact[Any]:
        validated = cast(SourceLifecycleContent, content)
        return await self._revise_artifact(connection, scope_id, current, validated, direct_source)


class SourceLifecycleApplication:
    """Expose one Scope's lifecycle service through ``runtime.lifecycle``."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._services: dict[str, SourceLifecycleService] = {}

    def for_scope(self, scope_id: str, /) -> SourceLifecycleService:
        service = self._services.get(scope_id)
        if service is None:
            service = SourceLifecycleService(self._runtime, scope_id)
            self._services[scope_id] = service
        return service


__all__ = [
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleService",
    "SourceLifecycleApplication",
]
