"""Formato por bloques con puntuación máxima por bloque (formato de índice v2).

Una posting list se parte en bloques de `block_size` postings consecutivos y
delante de los datos se escribe un directorio de bloques con, para cada uno,
su último `doc_id`, su tamaño en bytes y cotas superiores de lo que puede
puntuar cualquier documento del bloque. Es lo que necesita Block-Max WAND
(`bm25-ranking-engine`): saltar un bloque entero sin decodificarlo cuando su
cota no alcanza el umbral del top-k, y saber dónde empieza el siguiente sin
leer el actual.

## Qué cota se guarda

BM25 de un término en un documento es `idf(t) * tf_part(tf, len)`, con
`tf_part = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl))`. El `idf`
depende del corpus completo (en un índice particionado, del corpus global, no
del shard), así que no se hornea: cada bloque guarda `max_score = max tf_part`
calculado sobre sus postings reales, y el lector multiplica por su propio
`idf`. Para un lector con otros `k1`/`b` la cota precalculada no vale, pero
sí `max_term_frequency` y `min_document_length`: `tf_part` crece con `tf` y
decrece con `len`, así que `tf_part(max_tf, min_len)` es una cota válida para
cualquier parámetro — más holgada, pero correcta.

`max_score` se guarda en float32 redondeado **hacia arriba**: una cota que
subestima por redondeo haría que BMW descartara un documento que sí entraba
en el top-k, y entonces dejaría de ser exacto.

## Layout del blob de un término

    varint  num_postings
    varint  num_blocks
    -- directorio, una entrada por bloque --
    varint  last_doc_id_delta    último doc_id del bloque menos el del bloque
                                 anterior (el primero, desde 0)
    varint  block_postings
    varint  block_byte_length
    varint  max_term_frequency
    varint  min_document_length
    4 bytes max_score            float32 little-endian, redondeado hacia arriba
    -- bloques, en el mismo orden --
    byte    encoding_mode        0 = VARINT, 1 = PACKED (solo los gaps de doc_id)
    varint  first_doc_id_delta   primer doc_id menos el último del bloque
                                 anterior (el primero, doc_id + 1)
    ...     doc_id_gaps          block_postings - 1 valores
    ...     term_frequencies     block_postings valores, varint
    -- por posting: posiciones exactamente como en el formato v1 --
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from index_compression_codec.bitpacking import PackedBlockCodec
from index_compression_codec.models import EncodingMode, Posting, PostingsList
from index_compression_codec.postings_codec import (
    _cumulative_sum,
    _decode_positions,
    _encode_positions,
    _gaps_of,
)
from index_compression_codec.protocols import IntegerBlockCodec
from index_compression_codec.varint import VarintBlockCodec, decode_varint, encode_varint

DEFAULT_BLOCK_MAX_SIZE: Final[int] = 64
"""Postings por bloque. Más pequeño que el bloque de bitpacking del formato v1
(128): un bloque menor da cotas más ajustadas — más bloques que BMW puede
saltar — a cambio de un directorio algo mayor."""

_FLOAT32: Final[struct.Struct] = struct.Struct("<f")
_UINT32: Final[struct.Struct] = struct.Struct("<I")
_VARINT_CODEC: Final[IntegerBlockCodec] = VarintBlockCodec()


@dataclass(frozen=True, slots=True)
class BlockMaxParams:
    """Parámetros BM25 con los que se precalcula `max_score`. Por defecto los
    mismos de `bm25-ranking-engine` (`k1=1.5`, `b=0.75`)."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError(f"k1 debe ser no negativo, recibido {self.k1}")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b debe estar en [0, 1], recibido {self.b}")


@dataclass(frozen=True, slots=True)
class BlockMeta:
    last_doc_id: int
    num_postings: int
    byte_length: int
    max_term_frequency: int
    min_document_length: int
    max_score: float
    """Cota superior de `tf_part` en el bloque (sin `idf`)."""


@dataclass(frozen=True, slots=True)
class BlockDirectory:
    num_postings: int
    blocks: tuple[BlockMeta, ...]
    data_offset: int
    """Offset dentro del blob donde empieza el primer bloque."""

    def block_offsets(self) -> list[int]:
        offsets: list[int] = []
        position = self.data_offset
        for block in self.blocks:
            offsets.append(position)
            position += block.byte_length
        return offsets


def bm25_tf_part(
    term_frequency: int,
    document_length: int,
    average_document_length: float,
    params: BlockMaxParams,
) -> float:
    if average_document_length <= 0:
        raise ValueError("average_document_length debe ser positiva")
    length_norm = 1.0 - params.b + params.b * (document_length / average_document_length)
    return term_frequency * (params.k1 + 1.0) / (term_frequency + params.k1 * length_norm)


def float32_ceil(value: float) -> float:
    """El menor float32 >= `value`. El siguiente float32 se obtiene
    incrementando su patrón de bits: `math.nextafter` trabaja en float64 y su
    paso es demasiado pequeño para sobrevivir a volver a empaquetar en 32 bits."""
    if math.isnan(value):
        raise ValueError("max_score no puede ser NaN")
    (rounded,) = _FLOAT32.unpack(_FLOAT32.pack(value))
    if rounded >= value:
        return float(rounded)
    (bits,) = _UINT32.unpack(_FLOAT32.pack(rounded))
    if rounded > 0.0:
        bits += 1
    elif rounded < 0.0:
        bits -= 1
    else:
        bits = 1  # el menor subnormal positivo
    (next_up,) = _FLOAT32.unpack(_UINT32.pack(bits))
    return float(next_up)


def encode_block_postings_list(
    postings_list: PostingsList,
    document_lengths: Mapping[int, int],
    average_document_length: float,
    *,
    params: BlockMaxParams,
    block_size: int = DEFAULT_BLOCK_MAX_SIZE,
    bitpack_threshold: int = 32,
) -> tuple[bytes, float]:
    """Codifica `postings_list` en el layout por bloques. Devuelve el blob y
    la cota del término completo (el máximo de las cotas de sus bloques), que
    va al directorio `terms.jsonl` para que WAND la use sin abrir el blob."""
    if block_size <= 0:
        raise ValueError(f"block_size debe ser positivo, recibido {block_size}")
    postings = postings_list.postings
    doc_ids = [posting.doc_id for posting in postings]
    _gaps_of(doc_ids, context=f"doc_id de '{postings_list.term}'")

    metas: list[BlockMeta] = []
    blocks: list[bytes] = []
    previous_last = -1
    for start in range(0, len(postings), block_size):
        chunk = postings[start : start + block_size]
        block = bytearray()
        mode = EncodingMode.PACKED if len(chunk) >= bitpack_threshold else EncodingMode.VARINT
        block.append(int(mode))
        block.extend(encode_varint(chunk[0].doc_id - previous_last))
        gaps = _gaps_of([posting.doc_id for posting in chunk], context=postings_list.term)
        codec: IntegerBlockCodec = (
            PackedBlockCodec(block_size=block_size)
            if mode is EncodingMode.PACKED
            else _VARINT_CODEC
        )
        codec.encode(gaps, block)
        _VARINT_CODEC.encode([posting.term_frequency for posting in chunk], block)
        for posting in chunk:
            _encode_positions(posting.positions, block, term=postings_list.term)

        lengths = [_length_of(document_lengths, posting.doc_id) for posting in chunk]
        max_score = max(
            bm25_tf_part(posting.term_frequency, length, average_document_length, params)
            for posting, length in zip(chunk, lengths, strict=True)
        )
        metas.append(
            BlockMeta(
                last_doc_id=chunk[-1].doc_id,
                num_postings=len(chunk),
                byte_length=len(block),
                max_term_frequency=max(posting.term_frequency for posting in chunk),
                min_document_length=min(lengths),
                max_score=float32_ceil(max_score),
            )
        )
        blocks.append(bytes(block))
        previous_last = chunk[-1].doc_id

    buffer = bytearray()
    buffer.extend(encode_varint(len(postings)))
    buffer.extend(encode_varint(len(metas)))
    previous_last_for_directory = 0
    for meta in metas:
        buffer.extend(encode_varint(meta.last_doc_id - previous_last_for_directory))
        buffer.extend(encode_varint(meta.num_postings))
        buffer.extend(encode_varint(meta.byte_length))
        buffer.extend(encode_varint(meta.max_term_frequency))
        buffer.extend(encode_varint(meta.min_document_length))
        buffer.extend(_FLOAT32.pack(meta.max_score))
        previous_last_for_directory = meta.last_doc_id
    for block_bytes in blocks:
        buffer.extend(block_bytes)

    term_max = max((meta.max_score for meta in metas), default=0.0)
    return bytes(buffer), term_max


def decode_block_directory(data: bytes) -> BlockDirectory:
    num_postings, pos = decode_varint(data, 0)
    num_blocks, pos = decode_varint(data, pos)
    blocks: list[BlockMeta] = []
    previous_last = 0
    for _ in range(num_blocks):
        delta, pos = decode_varint(data, pos)
        block_postings, pos = decode_varint(data, pos)
        byte_length, pos = decode_varint(data, pos)
        max_tf, pos = decode_varint(data, pos)
        min_length, pos = decode_varint(data, pos)
        if pos + _FLOAT32.size > len(data):
            raise ValueError("directorio de bloques truncado: falta max_score")
        (max_score,) = _FLOAT32.unpack_from(data, pos)
        pos += _FLOAT32.size
        previous_last += delta
        blocks.append(
            BlockMeta(
                last_doc_id=previous_last,
                num_postings=block_postings,
                byte_length=byte_length,
                max_term_frequency=max_tf,
                min_document_length=min_length,
                max_score=float(max_score),
            )
        )
    if sum(block.num_postings for block in blocks) != num_postings:
        raise ValueError("directorio de bloques inconsistente con num_postings")
    return BlockDirectory(num_postings=num_postings, blocks=tuple(blocks), data_offset=pos)


def decode_block(
    data: bytes, offset: int, meta: BlockMeta, previous_last_doc_id: int, *, block_size: int
) -> list[Posting]:
    """Decodifica un único bloque. `previous_last_doc_id` es el último
    `doc_id` del bloque anterior (`-1` para el primero)."""
    mode = EncodingMode(data[offset])
    first_delta, pos = decode_varint(data, offset + 1)
    codec: IntegerBlockCodec = (
        PackedBlockCodec(block_size=block_size) if mode is EncodingMode.PACKED else _VARINT_CODEC
    )
    gaps, pos = codec.decode(data, pos, meta.num_postings - 1)
    doc_ids = _cumulative_sum(previous_last_doc_id + first_delta, gaps)
    frequencies, pos = _VARINT_CODEC.decode(data, pos, meta.num_postings)
    postings: list[Posting] = []
    for doc_id, term_frequency in zip(doc_ids, frequencies, strict=True):
        positions, pos = _decode_positions(data, pos)
        postings.append(Posting(doc_id=doc_id, term_frequency=term_frequency, positions=positions))
    if pos - offset != meta.byte_length or doc_ids[-1] != meta.last_doc_id:
        raise ValueError("bloque corrupto: longitud o último doc_id no coinciden con el directorio")
    return postings


def decode_block_postings_list(
    term: str, document_frequency: int, data: bytes, *, block_size: int
) -> PostingsList:
    directory = decode_block_directory(data)
    if directory.num_postings != document_frequency:
        raise ValueError(
            f"inconsistencia de índice para '{term}': el directorio declara "
            f"document_frequency={document_frequency}, el blob codifica {directory.num_postings}"
        )
    postings: list[Posting] = []
    previous_last = -1
    for meta, offset in zip(directory.blocks, directory.block_offsets(), strict=True):
        postings.extend(decode_block(data, offset, meta, previous_last, block_size=block_size))
        previous_last = meta.last_doc_id
    return PostingsList(term=term, document_frequency=len(postings), postings=tuple(postings))


def _length_of(document_lengths: Mapping[int, int], doc_id: int) -> int:
    length = document_lengths.get(doc_id)
    if length is None:
        raise ValueError(f"doc_id {doc_id} aparece en postings pero no en documents.jsonl")
    return length
