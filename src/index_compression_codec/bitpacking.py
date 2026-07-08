"""Bitpacking por bloques de ancho fijo: el esquema que hace escalar el índice
a términos muy frecuentes (listas de postings largas, ver `../AGENTS.md`).

Cada lista se divide en bloques de `block_size` valores. Dentro de cada bloque
se calcula el ancho de bits mínimo necesario para representar su valor máximo
(frame-of-reference por bloque) y se empaquetan todos sus valores con ese ancho
exacto, sin redondear a 8/16/32 bits — un bloque de deltas pequeños ocupa solo
los bits que realmente necesita, en vez de gastar un byte entero por valor como
haría varint cuando los deltas superan 127. El coste es un único byte de
cabecera (el ancho de bits) por bloque, que se amortiza bien cuanto más larga
es la lista — de ahí que `postings_codec.py` solo use este esquema por encima
de un umbral de longitud.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

DEFAULT_BLOCK_SIZE: Final[int] = 128


def _bit_width_for(values: Sequence[int]) -> int:
    if not values:
        return 0
    maximum = max(values)
    if maximum < 0:
        raise ValueError(f"bitpacking no soporta enteros negativos: {maximum}")
    return maximum.bit_length()


def _pack_block(values: Sequence[int], bit_width: int) -> bytes:
    if bit_width == 0:
        return b""

    bit_buffer = 0
    bits_in_buffer = 0
    output = bytearray()
    for value in values:
        if value < 0 or value.bit_length() > bit_width:
            raise ValueError(f"valor {value} no cabe en el ancho de bits del bloque ({bit_width})")
        bit_buffer |= value << bits_in_buffer
        bits_in_buffer += bit_width
        while bits_in_buffer >= 8:
            output.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer > 0:
        output.append(bit_buffer & 0xFF)
    return bytes(output)


def _unpack_block(data: bytes, offset: int, count: int, bit_width: int) -> tuple[list[int], int]:
    if bit_width == 0:
        return [0] * count, offset

    total_bits = count * bit_width
    total_bytes = (total_bits + 7) // 8
    chunk = data[offset : offset + total_bytes]
    if len(chunk) < total_bytes:
        raise ValueError("bloque de bitpacking truncado: faltan bytes empaquetados")

    bit_buffer = int.from_bytes(chunk, byteorder="little")
    mask = (1 << bit_width) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bit_width
    return values, offset + total_bytes


class PackedBlockCodec:
    """`IntegerBlockCodec` que empaqueta valores en bloques de ancho de bits fijo.

    El tamaño de bloque se fija por instancia (no por bloque) porque debe
    coincidir exactamente entre codificación y decodificación: es el propio
    `postings_codec.py` quien pasa el `block_size` leído del manifiesto al
    decodificar, para que un cambio futuro de parámetro no rompa índices ya
    construidos con un valor distinto.
    """

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size debe ser positivo, recibido {block_size}")
        self._block_size = block_size

    def encode(self, values: Sequence[int], buffer: bytearray) -> None:
        for start in range(0, len(values), self._block_size):
            chunk = values[start : start + self._block_size]
            bit_width = _bit_width_for(chunk)
            buffer.append(bit_width)
            buffer.extend(_pack_block(chunk, bit_width))

    def decode(self, data: bytes, offset: int, count: int) -> tuple[list[int], int]:
        values: list[int] = []
        pos = offset
        remaining = count
        while remaining > 0:
            chunk_len = min(self._block_size, remaining)
            if pos >= len(data):
                raise ValueError("bloque de bitpacking truncado: falta el byte de ancho de bits")
            bit_width = data[pos]
            pos += 1
            chunk_values, pos = _unpack_block(data, pos, chunk_len, bit_width)
            values.extend(chunk_values)
            remaining -= chunk_len
        return values, pos
