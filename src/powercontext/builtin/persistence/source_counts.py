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


"""Count pending Source journal identities after a binding's committed cursor."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.tables import SOURCE_JOURNAL_HEADS_TABLE

_BATCH = 100


async def source_pending_counts(
    connection: AsyncConnection,
    binding_name: str,
    scope_ids: Sequence[str],
    /,
) -> dict[str, int]:
    """Return journal head minus the last successful cursor for each Scope.

    Journal positions are allocated contiguously per Scope, one per new Source
    identity (including lineage-only server Sources). Replays and receipt trust
    upgrades reuse the stored position, so they never add to the count. Only
    metadata is read: no Source bodies, token counts, or generated output.
    """

    ordered = tuple(dict.fromkeys(scope_ids))
    if not ordered:
        return {}
    heads: dict[str, int] = {}
    table = SOURCE_JOURNAL_HEADS_TABLE
    for start in range(0, len(ordered), _BATCH):
        batch = ordered[start : start + _BATCH]
        rows = await connection.execute(select(table.c.scope_id, table.c.position).where(table.c.scope_id.in_(batch)))
        for scope_id, position in rows:
            heads[str(scope_id)] = int(position)
    cursors = await SourceCursorRepository().load_many(connection, ordered, binding_name)
    counts: dict[str, int] = {}
    for scope_id in ordered:
        cursor = cursors.get(scope_id)
        consumed = 0 if cursor is None else cursor.cursor.sequence
        counts[scope_id] = max(0, heads.get(scope_id, 0) - consumed)
    return counts


__all__ = ["source_pending_counts"]
