"""Tipos de datos del dominio para index-compression-codec.

`Posting` y `PostingsList` son copias propias del mismo concepto expuesto por
`inverted-index-builder`: no se importa su paquete Python (la integración entre
subproyectos de `beacon-search-engine` es siempre vía formatos serializados en
disco, nunca imports cruzados — ver `../AGENTS.md`), así que este repo define su
propia forma mínima de esos datos a partir del `postings.jsonl` que consume.

`TermEntry` es la entrada del directorio de términos que hace posible la
decodificación aleatoria: dado un término, permite ubicar sus bytes comprimidos
en `postings.bin` sin tener que descomprimir el índice entero (requisito central
de este repo, ver `../AGENTS.md`, sección de integración).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class EncodingMode(IntEnum):
    """Esquema de codificación aplicado a los deltas de `doc_id` de una posting list.

    Se decide por lista según su longitud (ver `postings_codec.py`): `VARINT` para
    listas cortas, donde el overhead de bitpacking por bloque no compensa; `PACKED`
    para listas largas (términos muy frecuentes), donde empaquetar cada bloque con
    el ancho de bits mínimo necesario reduce sustancialmente el tamaño frente a
    varint. El modo se guarda explícitamente en cada blob para que la decodificación
    sea autodescriptiva y no dependa de que el lector conozca el umbral usado al
    construir el índice.
    """

    VARINT = 0
    PACKED = 1


@dataclass(frozen=True, slots=True)
class Posting:
    """Aparición de un término en un documento concreto: frecuencia y posiciones."""

    doc_id: int
    term_frequency: int
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PostingsList:
    """Lista de postings de un término, ordenada por `doc_id` ascendente.

    El orden ascendente es un contrato heredado de `inverted-index-builder`, no un
    detalle de implementación: es el que hace posible el delta encoding sobre los
    IDs de documento (ver `../CLAUDE.md`, sección 2.E).
    """

    term: str
    document_frequency: int
    postings: tuple[Posting, ...]


@dataclass(frozen=True, slots=True)
class TermEntry:
    """Entrada del directorio `terms.jsonl`: ubica los bytes de un término en
    `postings.bin` sin necesidad de leer el resto del archivo.
    """

    term: str
    offset: int
    length: int
    document_frequency: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "offset": self.offset,
            "length": self.length,
            "document_frequency": self.document_frequency,
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> TermEntry:
        return TermEntry(
            term=str(data["term"]),
            offset=int(data["offset"]),
            length=int(data["length"]),
            document_frequency=int(data["document_frequency"]),
        )


@dataclass(frozen=True, slots=True)
class CompressionStats:
    """Métricas de compresión de una ejecución de `CompressionPipeline.compress`.

    `uncompressed_bytes` mide el tamaño en disco del `postings.jsonl` de origen, no
    una estimación en memoria: es la comparación justa frente a `compressed_bytes`
    (el tamaño real de `postings.bin`) para reportar el ratio de compresión.
    """

    total_documents: int
    vocabulary_size: int
    total_postings: int
    uncompressed_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 0.0
        return self.uncompressed_bytes / self.compressed_bytes

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "vocabulary_size": self.vocabulary_size,
            "total_postings": self.total_postings,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> CompressionStats:
        return CompressionStats(
            total_documents=int(data["total_documents"]),
            vocabulary_size=int(data["vocabulary_size"]),
            total_postings=int(data["total_postings"]),
            uncompressed_bytes=int(data["uncompressed_bytes"]),
            compressed_bytes=int(data["compressed_bytes"]),
        )
