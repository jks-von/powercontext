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


"""Count-based admission for Source-window processing (Memory / Topic Memory)."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest
from pydantic import ValidationError
from sqlalchemy import insert, update

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.source_counts import source_pending_counts
from powercontext.builtin.persistence.sqlite import SQLiteConfig, SQLiteProfile
from powercontext.builtin.persistence.tables import SHARED_TABLES, SOURCE_JOURNAL_HEADS_TABLE
from powercontext.builtin.runtime.artifact_processing import (
    ArtifactProcessingBinding,
    ArtifactProcessingSupervisor,
    ArtifactProcessingWorkAssignment,
    ArtifactProcessingWorkerCompletion,
)
from powercontext.builtin.runtime.config import RuntimeConfig
from powercontext.builtin.sources.journal import SourceCursor

BINDING = "memory-source-window-test"
QUIET = 0.8


@pytest.mark.parametrize("field", ["memory_source_count_threshold", "topic_memory_source_count_threshold"])
def test_runtime_config_threshold_is_strict_optional_positive_int(field: str) -> None:
    assert getattr(RuntimeConfig(), field) is None
    for value in (None, 1, 3, 100):
        assert getattr(RuntimeConfig(**{field: value}), field) == value
    for value in (True, False, 0, -1, 1.0, 2.5, "3", [3]):
        with pytest.raises(ValidationError):
            RuntimeConfig(**{field: value})


def test_binding_rejects_invalid_threshold() -> None:
    for value in (0, -2, True):
        with pytest.raises(ValueError):
            ArtifactProcessingBinding(BINDING, "memory", _Launcher(None), source_count_threshold=value)


async def _set_head(database: AsyncDatabase, scope: str, position: int) -> None:
    async with database.transaction() as connection:
        result = await connection.execute(
            update(SOURCE_JOURNAL_HEADS_TABLE)
            .where(SOURCE_JOURNAL_HEADS_TABLE.c.scope_id == scope)
            .values(position=position)
        )
        if result.rowcount == 0:
            await connection.execute(insert(SOURCE_JOURNAL_HEADS_TABLE).values(scope_id=scope, position=position))
        await ArtifactProcessingIntentRepository().mark_dirty(connection, scope, BINDING)


async def _cursor(database: AsyncDatabase, scope: str) -> int:
    async with database.transaction() as connection:
        stored = await SourceCursorRepository().load(connection, scope, BINDING)
    return 0 if stored is None else stored.cursor.sequence


class _Handle:
    def __init__(self, coroutine) -> None:
        self._task = asyncio.ensure_future(coroutine)

    async def wait(self) -> ArtifactProcessingWorkerCompletion:
        return await self._task

    async def terminate(self) -> None:
        self._task.cancel()
        with suppress(BaseException):
            await self._task


class _Launcher:
    """In-process stand-in for the Memory worker: one bounded window per invocation."""

    def __init__(self, database: AsyncDatabase | None, *, window: int = 1, fail_first: int = 0, block=None) -> None:
        self.database = database
        self.window = window
        self.fail_first = fail_first
        self.block = block
        self.started: list[ArtifactProcessingWorkAssignment] = []

    async def start(self, assignment: ArtifactProcessingWorkAssignment) -> _Handle:
        self.started.append(assignment)
        return _Handle(self._run(assignment))

    async def _run(self, assignment: ArtifactProcessingWorkAssignment) -> ArtifactProcessingWorkerCompletion:
        if self.block is not None:
            await self.block.wait()
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("controlled failure")  # noqa: TRY003
        assert self.database is not None
        intents = ArtifactProcessingIntentRepository()
        cursors = SourceCursorRepository()
        async with self.database.transaction() as connection:
            intent = await intents.load(connection, assignment.scope_id, BINDING, for_update=True)
            stored = await cursors.load(connection, assignment.scope_id, BINDING)
            after = 0 if stored is None else stored.cursor.sequence
            head = (await source_pending_counts(connection, BINDING, (assignment.scope_id,)))[assignment.scope_id]
            through = after + min(head, self.window)
            if through > after:
                await cursors.save(
                    connection,
                    assignment.scope_id,
                    BINDING,
                    SourceCursor(sequence=through),
                    expected_generation=None if stored is None else stored.generation,
                )
            drained = head <= self.window
            await intents.acknowledge(
                connection,
                assignment.scope_id,
                BINDING,
                assignment.claimed_request_generation,
                clean_generation=intent.dirty_generation if drained else None,
            )
        return ArtifactProcessingWorkerCompletion()


def _supervisor(
    database: AsyncDatabase, launcher: _Launcher, threshold: int | None, **kwargs
) -> ArtifactProcessingSupervisor:
    return ArtifactProcessingSupervisor(
        database=database,
        bindings=(
            ArtifactProcessingBinding(
                BINDING, "memory", launcher, max_workers=1, worker_timeout_seconds=30, source_count_threshold=threshold
            ),
        ),
        lease_mode="single-process",
        **kwargs,
    )


async def _until(predicate, limit: float = 5.0) -> None:
    async with asyncio.timeout(limit):
        while not await predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.02)


def test_pending_counts_use_head_minus_cursor(tmp_path) -> None:
    async def scenario() -> None:
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'counts.db'}")
        async with SQLiteProfile.open(config, tables=SHARED_TABLES) as profile:
            db = profile.database
            await _set_head(db, "a", 5)
            await _set_head(db, "b", 2)
            async with db.transaction() as connection:
                await SourceCursorRepository().save(
                    connection, "a", BINDING, SourceCursor(sequence=3), expected_generation=None
                )
                counts = await source_pending_counts(connection, BINDING, ("a", "b", "missing", "a"))
                other = await source_pending_counts(connection, "other-binding", ("a",))
            assert counts == {"a": 2, "b": 2, "missing": 0}
            assert other == {"a": 5}

    asyncio.run(scenario())


def test_threshold_admits_one_window_and_rechecks_remaining(tmp_path) -> None:
    async def scenario() -> None:
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'window.db'}")
        async with SQLiteProfile.open(config, tables=SHARED_TABLES) as profile:
            db = profile.database
            launcher = _Launcher(db, window=1)
            async with _supervisor(db, launcher, 3):
                await _set_head(db, "s", 2)
                await asyncio.sleep(QUIET)
                assert launcher.started == []
                await _set_head(db, "s", 3)

                async def first() -> bool:
                    return await _cursor(db, "s") == 1

                await _until(first)
                await asyncio.sleep(QUIET)
                # Remaining 2 < 3: no second count admission, no drain.
                assert len(launcher.started) == 1 and await _cursor(db, "s") == 1
                await _set_head(db, "s", 4)

                async def second() -> bool:
                    return await _cursor(db, "s") == 2

                await _until(second)
                await asyncio.sleep(QUIET)
                assert len(launcher.started) == 2 and await _cursor(db, "s") == 2

    asyncio.run(scenario())


def test_threshold_none_never_admits_and_scopes_are_independent(tmp_path) -> None:
    async def scenario() -> None:
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'scopes.db'}")
        async with SQLiteProfile.open(config, tables=SHARED_TABLES) as profile:
            db = profile.database
            launcher = _Launcher(db, window=100)
            async with _supervisor(db, launcher, None):
                await _set_head(db, "s", 50)
                await asyncio.sleep(QUIET)
                assert launcher.started == []
            launcher = _Launcher(db, window=100)
            async with _supervisor(db, launcher, 2):
                await _set_head(db, "t", 1)

                async def drained() -> bool:
                    return await _cursor(db, "s") == 50

                await _until(drained)
                await asyncio.sleep(QUIET)
                assert [a.scope_id for a in launcher.started] == ["s"]
                assert await _cursor(db, "t") == 0

    asyncio.run(scenario())


def test_failure_retries_without_consuming_cursor(tmp_path) -> None:
    async def scenario() -> None:
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'retry.db'}")
        async with SQLiteProfile.open(config, tables=SHARED_TABLES) as profile:
            db = profile.database
            launcher = _Launcher(db, window=10, fail_first=1)
            async with _supervisor(db, launcher, 2, retry_base_seconds=0.1, retry_jitter=lambda: 1.0):
                await _set_head(db, "s", 2)

                async def done() -> bool:
                    return await _cursor(db, "s") == 2

                await _until(done)
                assert len(launcher.started) == 2
                assert launcher.started[0].claimed_request_generation == launcher.started[1].claimed_request_generation

    asyncio.run(scenario())


def test_accepted_work_and_below_threshold_input_survive_restart(tmp_path) -> None:
    async def scenario() -> None:
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}")
        async with SQLiteProfile.open(config, tables=SHARED_TABLES) as profile:
            db = profile.database
            block = asyncio.Event()
            blocked = _Launcher(db, window=1, block=block)
            async with _supervisor(db, blocked, 2):
                await _set_head(db, "s", 2)
                await _set_head(db, "below", 1)

                async def started() -> bool:
                    return bool(blocked.started)

                await _until(started)
            assert await _cursor(db, "s") == 0
            # Restart: accepted request is rediscovered; threshold is not reapplied to it.
            launcher = _Launcher(db, window=1)
            async with _supervisor(db, launcher, 5):

                async def resumed() -> bool:
                    return await _cursor(db, "s") == 1

                await _until(resumed)
                await asyncio.sleep(QUIET)
                assert [a.scope_id for a in launcher.started] == ["s"]
                assert await _cursor(db, "below") == 0
            launcher = _Launcher(db, window=10)
            async with _supervisor(db, launcher, 1):

                async def both() -> bool:
                    return await _cursor(db, "s") == 2 and await _cursor(db, "below") == 1

                await _until(both)

    asyncio.run(scenario())
