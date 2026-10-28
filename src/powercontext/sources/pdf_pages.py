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

"""Full-page capture of PDF documents as physical-page Sources.

Every physical page of a PDF becomes one ``CAPTURED`` Source whose identity is
derived from the document and the page number, so the identity survives
reordering and edits of other pages.  The whole input bytes are hashed, so a
revised document produces a new identity while old references keep pointing at
the revision they were created from.
"""

from __future__ import annotations

import hashlib
import io
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
_SOURCE_ID_PREFIX = "pdf"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    return document_id


def _validate_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("document_sha256 must be a 64 character hex string")  # noqa: TRY003
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("document_sha256 must be hexadecimal") from error  # noqa: TRY003
    if value != value.lower():
        raise ValueError("document_sha256 must be lowercase")  # noqa: TRY003
    return value


def _validate_page_number(page_number: object) -> int:
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise ValueError("page_number must be an integer")  # noqa: TRY003
    if page_number < 1:
        raise ValueError("page_number must be at least 1")  # noqa: TRY003
    return page_number


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable, delimiter-safe identity of one physical page.

    Format: ``pdf:{quote(document_id, safe="")}:{document_sha256}:{page_number}``.
    """

    _validate_document_id(document_id)
    _validate_sha256(document_sha256)
    _validate_page_number(page_number)
    identity = f"{_SOURCE_ID_PREFIX}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"PDF page source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class PDFDocumentInput(BaseModel):
    """One complete offline PDF document identified by its caller."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One validated physical page of a PDF document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @model_validator(mode="after")
    def validate_page(self) -> PDFPageInput:
        if pdf_page_source_id(self.document_id, self.document_sha256, self.page_number) is None:
            raise ValueError("invalid PDF page identity")  # noqa: TRY003
        return self


class PDFPageSource(Source):
    """A captured PDF physical page identified by document and page number."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @model_validator(mode="after")
    def validate_identity(self) -> PDFPageSource:
        expected = pdf_page_source_id(self.document_id, self.document_sha256, self.page_number)
        if self.source_id != expected:
            raise ValueError("source_id does not match the PDF page identity")  # noqa: TRY003
        if self.name != self.source_id:
            raise ValueError("name must equal the PDF page source_id")  # noqa: TRY003
        if self.materialization is not SourceMaterialization.CAPTURED:
            raise ValueError("PDF page Sources must be captured")  # noqa: TRY003
        return self


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _page_text(page: object) -> str:
    """Extract the full extractable text of one physical page."""

    try:
        return page.extract_text() or ""
    except Exception:  # noqa: BLE001 - any extraction failure fails the whole document
        return ""


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse and validate a whole PDF document into physical-page inputs.

    The document bytes are hashed in full and every physical page is extracted in
    order, including pages without any extractable text.  Any decoding, structure
    or extraction failure raises ``ValueError``; no partial result is ever
    returned.
    """

    document_id = _validate_document_id(value.document_id)
    content = value.content
    if not isinstance(content, bytes):  # pragma: no cover - pydantic enforces bytes
        raise ValueError("PDF content must be bytes")  # noqa: TRY003
    if not content:
        raise ValueError("PDF document is empty")  # noqa: TRY003

    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content))
    except DependencyError as error:  # pragma: no cover - encryption dependency missing
        raise ValueError(f"PDF document cannot be read: {error}") from error  # noqa: TRY003
    except PdfReadError as error:
        raise ValueError(f"PDF document is corrupt: {error}") from error  # noqa: TRY003
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"PDF document is corrupt: {error}") from error  # noqa: TRY003

    try:
        page_count = len(reader.pages)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"PDF document cannot be read: {error}") from error  # noqa: TRY003
    if page_count < 1:
        raise ValueError("PDF document has no pages")  # noqa: TRY003

    document_sha256 = hashlib.sha256(content).hexdigest()

    pages: list[PDFPageInput] = []
    for index in range(page_count):
        try:
            page = reader.pages[index]
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"PDF page {index + 1} cannot be read: {error}") from error  # noqa: TRY003
        pages.append(
            PDFPageInput(
                document_id=document_id,
                document_sha256=document_sha256,
                page_number=index + 1,
                text=_page_text(page),
            )
        )
    return tuple(pages)


# ---------------------------------------------------------------------------
# Adapter and definition
# ---------------------------------------------------------------------------


class PDFPageSourceAdapter:
    """Resolve PDF page inputs into captured Sources and read them back."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version=PDF_PAGE_SOURCE_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


def pdf_page_text_evidence(source: PDFPageSource) -> TextEvidence:
    """Build the canonical text evidence for one PDF page Source."""

    return TextEvidence(
        source_type=PDF_PAGE_SOURCE_NAME,
        source_id=source.source_id,
        content=f"{source.document_id}\n{source.text}",
        metadata={
            "document_id": source.document_id,
            "document_sha256": source.document_sha256,
            "page_number": source.page_number,
        },
    )


class PDFPageTextEvidenceProjection:
    """Expose one PDF physical page as text evidence for Memory retrieval."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return pdf_page_text_evidence(source)


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def search_pdf_pages(
    memory,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search one scoped Memory for the evidence pages of ``question``.

    The search goes through the public Memory API; every returned citation is an
    exact Memory entry version, so the caller can read the page text and its
    document identity back through ``memory.get``.
    """

    from powercontext.builtin.runtime.models import SearchMemoryRequest

    if limit < 1:
        raise ValueError("limit must be at least 1")  # noqa: TRY003
    if not question or not question.strip():
        raise ValueError("question must contain non-whitespace content")  # noqa: TRY003

    # Over-fetch so that duplicates of the same page can be dropped while still
    # returning up to ``limit`` distinct pages.
    page = await memory.search(SearchMemoryRequest(query=question, limit=max(limit * 4, 40), mode="auto"))
    citations: list = []
    seen: set[tuple[str, str]] = set()
    for hit in page.hits:
        from powercontext.artifacts import MemoryCitation

        key = (hit.entry_id, hit.entry_version_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
