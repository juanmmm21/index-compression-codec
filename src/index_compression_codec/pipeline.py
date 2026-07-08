"""Orquestador de compresión: lee el índice sin comprimir producido por
`inverted-index-builder` (`documents.jsonl`, `postings.jsonl`, `stats.json`) y
escribe un índice equivalente con las listas de postings comprimidas.

A diferencia de `inverted-index-builder.pipeline.IndexBuilder` — que construye
un `InvertedIndex` completo en memoria antes de serializarlo — aquí se
comprime en streaming, línea a línea de `postings.jsonl` directamente a
`postings.bin`: no hay ninguna estructura intermedia que crezca con el tamaño
del índice. Es una desviación deliberada del patrón de `pipeline.py` +
`serialization.py` separados de ese repo, justificada por el propio objetivo
de este uno: escalar a millones de documentos sin desbordar memoria (ver
`../CLAUDE.md`, sección 2.E) sería imposible si hubiera que materializar el
índice entero, sin comprimir, antes de comprimirlo.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Final

from index_compression_codec.bitpacking import DEFAULT_BLOCK_SIZE
from index_compression_codec.models import CompressionStats, Posting, PostingsList, TermEntry
from index_compression_codec.postings_codec import DEFAULT_BITPACK_THRESHOLD, encode_postings_list
from index_compression_codec.serialization import (
    DOCUMENTS_FILENAME,
    POSTINGS_FILENAME,
    STATS_FILENAME,
    write_compression_stats,
    write_manifest,
    write_terms_directory,
)

SOURCE_DOCUMENTS_FILENAME: Final[str] = "documents.jsonl"
SOURCE_POSTINGS_FILENAME: Final[str] = "postings.jsonl"
SOURCE_STATS_FILENAME: Final[str] = "stats.json"


def _parse_postings_line(line: str) -> PostingsList:
    data: dict[str, Any] = json.loads(line)
    postings = tuple(
        Posting(
            doc_id=int(entry["doc_id"]),
            term_frequency=int(entry["term_frequency"]),
            positions=tuple(int(position) for position in entry["positions"]),
        )
        for entry in data["postings"]
    )
    return PostingsList(
        term=str(data["term"]),
        document_frequency=int(data["document_frequency"]),
        postings=postings,
    )


class CompressionPipeline:
    """Comprime un índice de `inverted-index-builder` de forma streaming.

    El tokenizador de `inverted-index-builder` es inyectable para testear la
    agregación de postings con dobles deterministas; aquí no hay nada
    equivalente que inyectar, porque el propio esquema de codificación
    (varint vs. bitpacking) se decide por lista según su longitud, no de forma
    global — el parámetro configurable es el umbral, no la implementación.
    """

    def __init__(
        self,
        *,
        bitpack_threshold: int = DEFAULT_BITPACK_THRESHOLD,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        if bitpack_threshold <= 0:
            raise ValueError(f"bitpack_threshold debe ser positivo, recibido {bitpack_threshold}")
        if block_size <= 0:
            raise ValueError(f"block_size debe ser positivo, recibido {block_size}")
        self._bitpack_threshold = bitpack_threshold
        self._block_size = block_size

    def compress(self, source_index_dir: Path, output_dir: Path) -> CompressionStats:
        postings_path = source_index_dir / SOURCE_POSTINGS_FILENAME
        documents_path = source_index_dir / SOURCE_DOCUMENTS_FILENAME
        stats_path = source_index_dir / SOURCE_STATS_FILENAME
        for path in (postings_path, documents_path, stats_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"no existe {path}: ¿es un índice válido de inverted-index-builder?"
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        source_stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))

        term_entries: list[TermEntry] = []
        uncompressed_bytes = 0
        compressed_bytes = 0
        total_postings = 0

        postings_bin_path = output_dir / POSTINGS_FILENAME
        with (
            postings_path.open("r", encoding="utf-8") as source_file,
            postings_bin_path.open("wb") as binary_file,
        ):
            offset = 0
            for raw_line in source_file:
                line = raw_line.strip()
                if not line:
                    continue
                uncompressed_bytes += len(raw_line.encode("utf-8"))

                postings_list = _parse_postings_line(line)
                encoded = encode_postings_list(
                    postings_list,
                    bitpack_threshold=self._bitpack_threshold,
                    block_size=self._block_size,
                )
                binary_file.write(encoded)

                term_entries.append(
                    TermEntry(
                        term=postings_list.term,
                        offset=offset,
                        length=len(encoded),
                        document_frequency=postings_list.document_frequency,
                    )
                )
                offset += len(encoded)
                compressed_bytes += len(encoded)
                total_postings += len(postings_list.postings)

        shutil.copyfile(documents_path, output_dir / DOCUMENTS_FILENAME)
        shutil.copyfile(stats_path, output_dir / STATS_FILENAME)

        stats = CompressionStats(
            total_documents=int(source_stats["total_documents"]),
            vocabulary_size=len(term_entries),
            total_postings=total_postings,
            uncompressed_bytes=uncompressed_bytes,
            compressed_bytes=compressed_bytes,
        )

        write_terms_directory(output_dir, term_entries)
        write_compression_stats(output_dir, stats)
        write_manifest(
            output_dir, block_size=self._block_size, bitpack_threshold=self._bitpack_threshold
        )

        return stats
