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

import asyncio
import json
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceMaterialization,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _parse(content: bytes, *, document_id: str = "doc", primary_key: str = "id") -> tuple[CSVRowInput, ...]:
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


def _resolve(row: CSVRowInput) -> CSVRowSource:
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(row))
    assert isinstance(source, CSVRowSource)
    return source


def test_definition_metadata() -> None:
    assert CSV_ROW_SOURCE_DEFINITION.name == "csv-row"
    assert CSV_ROW_SOURCE_DEFINITION.version == "1"
    assert CSV_ROW_SOURCE_DEFINITION.input_class is CSVRowInput
    assert CSV_ROW_SOURCE_DEFINITION.source_class is CSVRowSource
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    row = _parse(b"id,name\n1,one\n")[0]
    source = _resolve(row)
    assert registry.projection_keys(source) == (TEXT_EVIDENCE_PROJECTION_KEY,)
    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
    assert manifest.name == "csv-row"
    assert manifest.version == "1"


def test_values_are_preserved_without_trim_or_type_conversion() -> None:
    rows = _parse(b"id,amount,flag,date,blank, padded \n007, 1.50 ,TRUE,2026-01-01,, x \n")
    assert rows[0].columns == ("id", "amount", "flag", "date", "blank", " padded ")
    assert rows[0].values == ("007", " 1.50 ", "TRUE", "2026-01-01", "", " x ")
    source = _resolve(rows[0])
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id == "csv:doc:007"
    evidence = TextEvidence.model_validate(
        SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,)).project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    )
    assert (
        evidence.content
        == '{"id":"007","amount":" 1.50 ","flag":"TRUE","date":"2026-01-01","blank":""," padded ":" x "}'
    )
    assert list(evidence.metadata) == ["document_id", "primary_key", "primary_key_value", "columns", "row"]
    assert list(evidence.metadata["row"]) == list(rows[0].columns)  # type: ignore[arg-type]


def test_non_ascii_content_is_not_escaped_and_identity_is_percent_encoded() -> None:
    rows = _parse("编号,名称\n北京,首都国际机场\n".encode(), document_id="机场", primary_key="编号")
    source = _resolve(rows[0])
    assert source.source_id == f"csv:{quote('机场', safe='')}:{quote('北京', safe='')}"
    evidence = TextEvidence.model_validate(
        SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,)).project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    )
    assert evidence.content == json.dumps(
        {"编号": "北京", "名称": "首都国际机场"}, ensure_ascii=False, separators=(",", ":")
    )


def test_header_only_and_blank_records() -> None:
    assert _parse(b"id,name") == ()
    assert _parse(b"id,name\r\n") == ()
    assert _parse(b"\xef\xbb\xbfid,name\n\n\n") == ()
    rows = _parse(b"id,name\n\n1,one\n\r\n2,two\n\n")
    assert [row.primary_key_value for row in rows] == ["1", "2"]


@pytest.mark.parametrize(
    ("document_id", "primary_key", "content"),
    [
        ("doc", "id", b""),
        ("doc", "id", b"\xef\xbb\xbf"),
        ("doc", "id", b"\n"),
        ("doc", "id", b"\nid,name\n1,one\n"),
        ("doc", "", b"id,name\n1,one\n"),
        ("doc", " id", b"id,name\n1,one\n"),
        ("doc", "id ", b" id ,name\n1,one\n"),
        ("", "id", b"id,name\n1,one\n"),
        ("doc\n", "id", b"id,name\n1,one\n"),
        ("doc", "id", b"id,name\n 	,blank-key\n"),
        ("doc", "id", b"id,name\n1,one,extra\n"),
        ("doc", "id", b'id,name\n1,"a"b\n'),
        ("doc", "id", b"id,name\n1,one\n2,\xc3\x28\n"),
        ("doc", "id", b"\xef\xbb\xbf\xef\xbb\xbfid,name\n1,one\n"),
        ("doc", "id", b"id,name\n" + b"x" * 300 + b",long-key\n"),
    ],
)
def test_invalid_inputs_raise_value_error(document_id: str, primary_key: str, content: bytes) -> None:
    with pytest.raises(ValueError):
        _parse(content, document_id=document_id, primary_key=primary_key)


def test_errors_are_atomic_even_after_valid_rows() -> None:
    content = b"id,name\n" + b"".join(f"{index},row{index}\n".encode() for index in range(100)) + b"1,duplicate\n"
    with pytest.raises(ValueError):
        _parse(content)


def test_identity_is_bounded_and_collision_free() -> None:
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")
    assert csv_row_source_id("a%3Ab", "c") != csv_row_source_id("a:b", "c")
    long_value = "k" * (256 - len("csv:doc:"))
    assert len(csv_row_source_id("doc", long_value)) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("doc", long_value + "k")
    with pytest.raises(ValueError):
        csv_row_source_id("doc", "é" * 60)  # 6 encoded chars each -> exceeds 256
    with pytest.raises(ValueError):
        csv_row_source_id("doc", "  ")
    with pytest.raises(ValueError):
        csv_row_source_id("", "1")


def test_row_and_source_models_reject_inconsistent_values() -> None:
    with pytest.raises(ValidationError):
        CSVRowInput(document_id="doc", primary_key="id", primary_key_value="2", columns=("id",), values=("1",))
    with pytest.raises(ValidationError):
        CSVRowSource(
            name="csv:doc:2",
            materialization=SourceMaterialization.CAPTURED,
            source_id="csv:doc:2",
            document_id="doc",
            primary_key="id",
            primary_key_value="1",
            columns=("id",),
            values=("1",),
        )


def test_transport_observation_round_trip_preserves_projection() -> None:
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    rows = _parse(b'id,note\r\n1,"multi\r\nline ""quoted"""\r\n')
    source = _resolve(rows[0])
    observation = project_source_for_transport(registry, source)
    restored = SourceObservation.model_validate_json(observation.model_dump_json())
    assert restored.source_type == "csv-row"
    assert restored.name == source.source_id
    assert restored.projection(TEXT_EVIDENCE_PROJECTION_KEY) == registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    assert CSVRowSource.model_validate(restored.payload) == source
    assert asyncio.run(registry.read(CSVRowSource.model_validate(restored.payload))) == rows[0]
