"""Formato en disco de un índice comprimido, y el lector de acceso aleatorio
que lo consulta sin descomprimirlo entero. Un índice comprimido es un
directorio con cinco archivos:

    manifest.json          versión de formato + nombres de archivo + parámetros
                            de codificación usados al construirlo (`block_size`,
                            `bitpack_threshold`), para que decodificar no dependa
                            de que el lector adivine con qué parámetros se
                            construyó
    documents.jsonl         copia sin modificar del `documents.jsonl` de origen
    stats.json               copia sin modificar del `stats.json` de origen
                            (estadísticas del corpus, no de la compresión)
    terms.jsonl               directorio de términos: una línea por término,
                            ordenado alfabéticamente, con su `(offset, length)`
                            dentro de `postings.bin`
    postings.bin                blob binario: concatenación de los blobs de
                            `postings_codec.encode_postings_list`, uno por
                            término, en el mismo orden que `terms.jsonl`
    compression_stats.json       métricas de la compresión (ver `models.CompressionStats`)

Se documenta explícitamente aquí y en el README para que `bm25-ranking-engine`,
`pagerank-link-analysis` y `distributed-index-sharding` puedan leerlo sin
ambigüedad ni depender del código Python de este paquete.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from index_compression_codec.block_max import (
    BlockDirectory,
    BlockMaxParams,
    decode_block_directory,
    decode_block_postings_list,
)
from index_compression_codec.models import CompressionStats, PostingsList, TermEntry
from index_compression_codec.postings_codec import decode_postings_list

FORMAT_VERSION: Final[int] = 1
"""Formato clásico: un blob por término (`postings_codec.py`)."""
BLOCK_MAX_FORMAT_VERSION: Final[int] = 2
"""Formato por bloques con cotas por bloque (`block_max.py`)."""
SUPPORTED_FORMAT_VERSIONS: Final[frozenset[int]] = frozenset(
    {FORMAT_VERSION, BLOCK_MAX_FORMAT_VERSION}
)

MANIFEST_FILENAME: Final[str] = "manifest.json"
DOCUMENTS_FILENAME: Final[str] = "documents.jsonl"
STATS_FILENAME: Final[str] = "stats.json"
TERMS_FILENAME: Final[str] = "terms.jsonl"
POSTINGS_FILENAME: Final[str] = "postings.bin"
COMPRESSION_STATS_FILENAME: Final[str] = "compression_stats.json"


def write_manifest(
    output_dir: Path,
    *,
    block_size: int,
    bitpack_threshold: int,
    block_max: BlockMaxParams | None = None,
    average_document_length: float | None = None,
) -> None:
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION if block_max is None else BLOCK_MAX_FORMAT_VERSION,
        "documents_file": DOCUMENTS_FILENAME,
        "stats_file": STATS_FILENAME,
        "terms_file": TERMS_FILENAME,
        "postings_file": POSTINGS_FILENAME,
        "compression_stats_file": COMPRESSION_STATS_FILENAME,
        "block_size": block_size,
        "bitpack_threshold": bitpack_threshold,
    }
    if block_max is not None:
        manifest["block_max"] = {
            "scoring": "bm25",
            "k1": block_max.k1,
            "b": block_max.b,
            "average_document_length": average_document_length,
        }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_manifest(input_dir: Path) -> dict[str, Any]:
    """Lee y valida el manifiesto de `input_dir`.

    Falla explícitamente si declara un `format_version` que esta versión de la
    librería no sabe leer, en vez de intentar decodificar un formato binario
    incompatible en silencio (mismo contrato que `inverted-index-builder`).
    """
    manifest_path = input_dir / MANIFEST_FILENAME
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    format_version = int(manifest["format_version"])
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"formato de índice comprimido no soportado: version {format_version}, "
            f"soportados {sorted(SUPPORTED_FORMAT_VERSIONS)}"
        )
    return manifest


def write_terms_directory(output_dir: Path, term_entries: list[TermEntry]) -> None:
    """Escribe el directorio de términos ordenado alfabéticamente.

    Se ordena aquí, no se asume que `term_entries` ya venga ordenado: es la
    misma garantía de determinismo por la que `inverted-index-builder` ordena
    sus propios ficheros de salida antes de escribirlos (ver `../CLAUDE.md`,
    sección 2.B), no un detalle que un llamador futuro deba recordar mantener.
    """
    terms_path = output_dir / TERMS_FILENAME
    with terms_path.open("w", encoding="utf-8") as terms_file:
        for entry in sorted(term_entries, key=lambda term_entry: term_entry.term):
            terms_file.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")


def _read_terms_directory(input_dir: Path, terms_filename: str) -> dict[str, TermEntry]:
    terms_path = input_dir / terms_filename
    entries: dict[str, TermEntry] = {}
    for line in terms_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = TermEntry.from_json_dict(json.loads(line))
        entries[entry.term] = entry
    return entries


def write_compression_stats(output_dir: Path, stats: CompressionStats) -> None:
    (output_dir / COMPRESSION_STATS_FILENAME).write_text(
        json.dumps(stats.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_compression_stats(input_dir: Path) -> CompressionStats:
    manifest = read_manifest(input_dir)
    stats_path = input_dir / str(manifest["compression_stats_file"])
    return CompressionStats.from_json_dict(json.loads(stats_path.read_text(encoding="utf-8")))


class CompressedIndexReader:
    """Lector de acceso aleatorio de un índice comprimido.

    Abre `postings.bin` una única vez y mantiene en memoria solo el directorio
    de términos (proporcional al tamaño del vocabulario, no al de los
    postings): `lookup` busca el `TermEntry` del término, hace `seek` al
    offset exacto y decodifica únicamente esos bytes — nunca lee ni descomprime
    el resto del índice. Es el requisito central de este repo (ver
    `../AGENTS.md`, integración).

    Se usa como gestor de contexto para garantizar el cierre explícito del
    descriptor de archivo (`../CLAUDE.md`, sección 2.E):

        with CompressedIndexReader(index_dir) as reader:
            postings = reader.lookup("search")
    """

    def __init__(self, index_dir: Path) -> None:
        manifest = read_manifest(index_dir)
        self._format_version = int(manifest["format_version"])
        self._block_size = int(manifest["block_size"])
        block_max = manifest.get("block_max")
        self._block_max_params = (
            BlockMaxParams(k1=float(block_max["k1"]), b=float(block_max["b"]))
            if block_max is not None
            else None
        )
        self._terms = _read_terms_directory(index_dir, str(manifest["terms_file"]))
        postings_path = index_dir / str(manifest["postings_file"])
        self._file = postings_path.open("rb")

    def __enter__(self) -> CompressedIndexReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def terms(self) -> list[str]:
        """Términos del índice, ordenados alfabéticamente."""
        return sorted(self._terms)

    def document_frequency(self, term: str) -> int | None:
        entry = self._terms.get(term)
        return entry.document_frequency if entry is not None else None

    @property
    def format_version(self) -> int:
        return self._format_version

    @property
    def block_max_params(self) -> BlockMaxParams | None:
        """Parámetros con los que se precalcularon las cotas (solo v2)."""
        return self._block_max_params

    def term_max_score(self, term: str) -> float | None:
        entry = self._terms.get(term)
        return entry.max_score if entry is not None else None

    def _read_blob(self, term: str) -> tuple[TermEntry, bytes] | None:
        entry = self._terms.get(term)
        if entry is None:
            return None
        self._file.seek(entry.offset)
        data = self._file.read(entry.length)
        if len(data) != entry.length:
            raise ValueError(
                f"lectura truncada para '{term}': esperados {entry.length} bytes, "
                f"leídos {len(data)}"
            )
        return entry, data

    def lookup(self, term: str) -> PostingsList | None:
        """Decodifica los postings de `term`, o `None` si no está en el índice."""
        blob = self._read_blob(term)
        if blob is None:
            return None
        entry, data = blob
        if self._format_version == BLOCK_MAX_FORMAT_VERSION:
            return decode_block_postings_list(
                term, entry.document_frequency, data, block_size=self._block_size
            )
        return decode_postings_list(
            term, entry.document_frequency, data, block_size=self._block_size
        )

    def block_directory(self, term: str) -> BlockDirectory | None:
        """Directorio de bloques de `term` (solo v2), sin decodificar postings."""
        if self._format_version != BLOCK_MAX_FORMAT_VERSION:
            raise ValueError("block_directory solo existe en el formato v2 (block-max)")
        blob = self._read_blob(term)
        return decode_block_directory(blob[1]) if blob is not None else None
