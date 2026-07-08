"""Formato binario de una posting list individual: la unidad que se guarda en
`postings.bin` en el rango `[offset, offset + length)` de cada `TermEntry` (ver
`serialization.py`). Cada blob es autocontenido y autodescriptivo — se puede
decodificar en aislamiento, sin leer nada más del archivo, que es justo lo que
hace posible el acceso aleatorio por término exigido en `../AGENTS.md`.

Diseño del blob (todos los campos como se generan, en orden):

    byte    encoding_mode        0 = VARINT, 1 = PACKED (ver `models.EncodingMode`)
    varint  num_postings
    -- solo si num_postings > 0 --
    varint  first_doc_id
    ...     doc_id_gaps          num_postings - 1 valores, codificados con el
                                  `IntegerBlockCodec` indicado por encoding_mode
    ...     term_frequencies     num_postings valores, siempre varint (ver nota)
    -- por cada posting, en el mismo orden que doc_id_gaps --
        varint  num_positions
        -- solo si num_positions > 0 --
        varint  first_position
        ...     position_gaps    num_positions - 1 valores, siempre varint

`encoding_mode` solo afecta a los deltas de `doc_id`: es la sección que más
pesa en términos muy frecuentes y la única para la que el bitpacking por
bloques compensa su overhead de cabecera. Frecuencias y posiciones se quedan
en varint por diseño explícito: son secuencias cortas por posting (rara vez
compensan un bloque de bitpacking) y frecuencias en particular no son
monótonas, así que no hay delta que aplicarles. Mantenerlas en varint es una
decisión de alcance, no una limitación — ver el README para el detalle.

El primer `doc_id`/posición de cada sección se guarda aparte del bloque de
gaps, en vez de como "gap desde cero": mezclarlo dentro del primer bloque de
bitpacking obligaría a ese bloque a usar el ancho de bits del propio `doc_id`
absoluto (normalmente mucho mayor que los gaps reales entre postings
consecutivos), inflando el tamaño de todo el bloque.
"""

from __future__ import annotations

from typing import Final

from index_compression_codec.bitpacking import DEFAULT_BLOCK_SIZE, PackedBlockCodec
from index_compression_codec.models import EncodingMode, Posting, PostingsList
from index_compression_codec.protocols import IntegerBlockCodec
from index_compression_codec.varint import VarintBlockCodec, decode_varint, encode_varint

DEFAULT_BITPACK_THRESHOLD: Final[int] = 64

_VARINT_CODEC: Final[IntegerBlockCodec] = VarintBlockCodec()


def encode_postings_list(
    postings_list: PostingsList,
    *,
    bitpack_threshold: int = DEFAULT_BITPACK_THRESHOLD,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> bytes:
    """Codifica `postings_list` como un blob binario autocontenido.

    Exige que los postings vengan ordenados estrictamente ascendentes por
    `doc_id` (contrato heredado de `inverted-index-builder`) y que las
    posiciones dentro de cada posting vengan ordenadas estrictamente
    ascendentes: ambas cosas son necesarias para que el delta encoding
    produzca únicamente gaps positivos, y se validan explícitamente en vez de
    producir un blob corrupto en silencio.
    """
    postings = postings_list.postings
    num_postings = len(postings)
    mode = EncodingMode.PACKED if num_postings >= bitpack_threshold else EncodingMode.VARINT

    buffer = bytearray()
    buffer.append(int(mode))
    buffer.extend(encode_varint(num_postings))

    if num_postings == 0:
        return bytes(buffer)

    doc_id_codec: IntegerBlockCodec = (
        PackedBlockCodec(block_size=block_size) if mode is EncodingMode.PACKED else _VARINT_CODEC
    )

    doc_ids = [posting.doc_id for posting in postings]
    buffer.extend(encode_varint(doc_ids[0]))
    doc_id_gaps = _gaps_of(doc_ids, context=f"doc_id de '{postings_list.term}'")
    doc_id_codec.encode(doc_id_gaps, buffer)

    frequencies = [posting.term_frequency for posting in postings]
    _VARINT_CODEC.encode(frequencies, buffer)

    for posting in postings:
        _encode_positions(posting.positions, buffer, term=postings_list.term)

    return bytes(buffer)


def decode_postings_list(
    term: str,
    document_frequency: int,
    data: bytes,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> PostingsList:
    """Reconstruye la `PostingsList` de `term` a partir de su blob binario.

    `document_frequency` viene del directorio `terms.jsonl` (ver
    `serialization.py`) y se contrasta contra el `num_postings` codificado
    dentro del propio blob: una discrepancia indica un índice corrupto o un
    directorio desincronizado con `postings.bin`, y se rechaza explícitamente
    en vez de devolver un resultado parcialmente incorrecto.
    """
    if not data:
        raise ValueError(f"blob de postings vacío para '{term}': índice corrupto")

    mode = EncodingMode(data[0])
    num_postings, pos = decode_varint(data, 1)
    if num_postings != document_frequency:
        raise ValueError(
            f"inconsistencia de índice para '{term}': el directorio declara "
            f"document_frequency={document_frequency}, el blob codifica {num_postings}"
        )

    if num_postings == 0:
        return PostingsList(term=term, document_frequency=0, postings=())

    doc_id_codec: IntegerBlockCodec = (
        PackedBlockCodec(block_size=block_size) if mode is EncodingMode.PACKED else _VARINT_CODEC
    )

    first_doc_id, pos = decode_varint(data, pos)
    doc_id_gaps, pos = doc_id_codec.decode(data, pos, num_postings - 1)
    doc_ids = _cumulative_sum(first_doc_id, doc_id_gaps)

    frequencies, pos = _VARINT_CODEC.decode(data, pos, num_postings)

    postings = []
    for doc_id, term_frequency in zip(doc_ids, frequencies, strict=True):
        positions, pos = _decode_positions(data, pos)
        postings.append(Posting(doc_id=doc_id, term_frequency=term_frequency, positions=positions))

    return PostingsList(term=term, document_frequency=num_postings, postings=tuple(postings))


def _gaps_of(values: list[int], *, context: str) -> list[int]:
    gaps = []
    for previous, current in zip(values, values[1:], strict=False):
        gap = current - previous
        if gap <= 0:
            raise ValueError(
                f"{context} no están ordenados estrictamente ascendentes: "
                f"{previous} seguido de {current}"
            )
        gaps.append(gap)
    return gaps


def _cumulative_sum(first: int, gaps: list[int]) -> list[int]:
    values = [first]
    for gap in gaps:
        values.append(values[-1] + gap)
    return values


def _encode_positions(positions: tuple[int, ...], buffer: bytearray, *, term: str) -> None:
    buffer.extend(encode_varint(len(positions)))
    if not positions:
        return
    buffer.extend(encode_varint(positions[0]))
    gaps = _gaps_of(list(positions), context=f"posiciones de '{term}'")
    _VARINT_CODEC.encode(gaps, buffer)


def _decode_positions(data: bytes, offset: int) -> tuple[tuple[int, ...], int]:
    num_positions, pos = decode_varint(data, offset)
    if num_positions == 0:
        return (), pos
    first_position, pos = decode_varint(data, pos)
    gaps, pos = _VARINT_CODEC.decode(data, pos, num_positions - 1)
    return tuple(_cumulative_sum(first_position, gaps)), pos
