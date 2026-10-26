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

"""Row-level capture of offline CSV documents as primary-keyed Sources.

Every data record of a CSV document becomes one ``CAPTURED`` Source whose
identity is derived from the document and the record's primary key value, so
the identity survives reordering of the file and edits to non-key fields.
Header names, column order and field values are preserved exactly as strings.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
_SOURCE_ID_PREFIX = "csv"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    return document_id


def _validate_primary_key(primary_key: object) -> str:
    if not isinstance(primary_key, str) or not primary_key:
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003
    return primary_key


def _validate_primary_key_value(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("primary key value must not be empty or whitespace-only")  # noqa: TRY003
    return value


def _validate_header(columns: tuple[str, ...]) -> None:
    if not columns:
        raise ValueError("CSV header must not be empty")  # noqa: TRY003
    seen: set[str] = set()
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("CSV header fields must be non-empty and not whitespace-only")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"CSV header field {column!r} is duplicated")  # noqa: TRY003
        seen.add(column)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable, delimiter-safe identity of one CSV row.

    Format: ``csv:{quote(document_id, safe="")}:{quote(primary_key_value, safe="")}``.
    """

    _validate_document_id(document_id)
    _validate_primary_key_value(primary_key_value)
    identity = f"{_SOURCE_ID_PREFIX}:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def _validate_row_fields(
    document_id: str,
    primary_key: str,
    primary_key_value: str,
    columns: tuple[str, ...],
    values: tuple[str, ...],
) -> str:
    _validate_document_id(document_id)
    _validate_primary_key(primary_key)
    _validate_header(columns)
    if len(values) != len(columns):
        raise ValueError("CSV row values must match the header width")  # noqa: TRY003
    if primary_key not in columns:
        raise ValueError(f"primary_key {primary_key!r} is not a header field")  # noqa: TRY003
    if values[columns.index(primary_key)] != primary_key_value:
        raise ValueError("primary_key_value must equal the row's primary key field")  # noqa: TRY003
    return csv_row_source_id(document_id, primary_key_value)


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document and the column used as its primary key."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated CSV data record in original header order."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row(self) -> CSVRowInput:
        _validate_row_fields(
            self.document_id,
            self.primary_key,
            self.primary_key_value,
            self.columns,
            self.values,
        )
        return self


class CSVRowSource(Source):
    """A captured CSV row identified by document and primary key value."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> CSVRowSource:
        expected = _validate_row_fields(
            self.document_id,
            self.primary_key,
            self.primary_key_value,
            self.columns,
            self.values,
        )
        if self.source_id != expected:
            raise ValueError("source_id does not match the CSV row identity")  # noqa: TRY003
        if self.name != self.source_id:
            raise ValueError("name must equal the CSV row source_id")  # noqa: TRY003
        if self.materialization is not SourceMaterialization.CAPTURED:
            raise ValueError("CSV row Sources must be captured")  # noqa: TRY003
        return self

    def row(self) -> dict[str, str]:
        """Return the full record as an ordered column-to-value mapping."""

        return dict(zip(self.columns, self.values, strict=True))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _decode_records(content: bytes) -> list[list[str]]:
    try:
        text = bytes(content).decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content is not valid UTF-8") from error  # noqa: TRY003
    if not text:
        raise ValueError("CSV document is empty")  # noqa: TRY003
    try:
        return list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"malformed CSV: {error}") from error  # noqa: TRY003


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a whole CSV document into primary-keyed rows.

    The document is decoded as strict UTF-8 (a leading BOM is accepted) and read
    with the standard ``csv.reader`` dialect with ``strict=True``.  Any encoding,
    CSV syntax or data validation failure raises ``ValueError``; no partial result
    is ever returned.
    """

    document_id = _validate_document_id(value.document_id)
    primary_key = _validate_primary_key(value.primary_key)
    records = _decode_records(value.content)

    if not records:
        raise ValueError("CSV document is empty")  # noqa: TRY003
    columns = tuple(records[0])
    _validate_header(columns)
    if primary_key not in columns:
        raise ValueError(f"primary_key {primary_key!r} is not a header field")  # noqa: TRY003
    key_index = columns.index(primary_key)
    width = len(columns)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for line_number, record in enumerate(records[1:], start=2):
        if not record:
            continue  # completely empty physical record
        if len(record) != width:
            raise ValueError(  # noqa: TRY003
                f"CSV record {line_number} has {len(record)} fields, expected {width}"
            )
        key_value = record[key_index]
        if not key_value.strip():
            raise ValueError(f"CSV record {line_number} has an empty primary key value")  # noqa: TRY003
        if key_value in seen_keys:
            raise ValueError(f"CSV record {line_number} duplicates primary key value {key_value!r}")  # noqa: TRY003
        seen_keys.add(key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Source Definition
# ---------------------------------------------------------------------------


class CSVRowSourceAdapter:
    """Resolve CSV row inputs into captured Sources and read them back."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version=CSV_ROW_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=tuple(source.columns),
            values=tuple(source.values),
        )


def csv_row_text_evidence(source: CSVRowSource) -> TextEvidence:
    """Build the canonical text evidence for one CSV row Source."""

    row = source.row()
    return TextEvidence(
        source_type=CSV_ROW_SOURCE_NAME,
        source_id=source.source_id,
        content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
        metadata={
            "document_id": source.document_id,
            "primary_key": source.primary_key,
            "primary_key_value": source.primary_key_value,
            "columns": list(source.columns),
            "row": dict(row),
        },
    )


class CSVRowTextEvidenceProjection:
    """Expose a CSV row as an ordered compact JSON object for Memory retrieval."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        return csv_row_text_evidence(source)


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
