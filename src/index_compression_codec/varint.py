"""Codificación varint (LEB128 sin signo): la unidad base de todo el formato
binario de este repo. Cada entero no negativo se parte en grupos de 7 bits,
con el bit alto de cada byte indicando si sigue otro byte a continuación. Los
valores pequeños (deltas típicos entre `doc_id` consecutivos o posiciones
consecutivas) ocupan un único byte en vez de los 4-8 bytes fijos de un entero
sin comprimir — es la base mínima de compresión exigida en `../CLAUDE.md`
(sección 2.E), antes incluso de aplicar bitpacking.
"""

from __future__ import annotations

from collections.abc import Sequence

_CONTINUATION_BIT = 0x80
_PAYLOAD_MASK = 0x7F
_PAYLOAD_BITS = 7


def encode_varint(value: int) -> bytes:
    """Codifica un entero no negativo como varint LEB128."""
    if value < 0:
        raise ValueError(f"varint no soporta enteros negativos: {value}")

    result = bytearray()
    remaining = value
    while True:
        byte = remaining & _PAYLOAD_MASK
        remaining >>= _PAYLOAD_BITS
        if remaining:
            result.append(byte | _CONTINUATION_BIT)
        else:
            result.append(byte)
            break
    return bytes(result)


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decodifica un varint a partir de `offset`. Devuelve `(valor, nuevo_offset)`."""
    result = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(data):
            raise ValueError("varint truncado: fin del buffer antes del byte de terminación")
        byte = data[pos]
        pos += 1
        result |= (byte & _PAYLOAD_MASK) << shift
        if not (byte & _CONTINUATION_BIT):
            break
        shift += _PAYLOAD_BITS
    return result, pos


class VarintBlockCodec:
    """`IntegerBlockCodec` que codifica cada valor como un varint independiente.

    Es el esquema por defecto: sin overhead de cabecera por bloque, ideal para
    listas de postings cortas donde el bitpacking en bloques (`bitpacking.py`)
    no llega a amortizar su byte de ancho de bits por bloque.
    """

    def encode(self, values: Sequence[int], buffer: bytearray) -> None:
        for value in values:
            buffer.extend(encode_varint(value))

    def decode(self, data: bytes, offset: int, count: int) -> tuple[list[int], int]:
        values: list[int] = []
        pos = offset
        for _ in range(count):
            value, pos = decode_varint(data, pos)
            values.append(value)
        return values, pos
