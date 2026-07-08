"""Interfaz que desacopla `postings_codec.py` de la implementación concreta de
codificación de enteros, para poder testear cada codec (varint, bitpacking) de
forma aislada y elegir entre ellos por lista según su longitud sin que el
formato binario de una posting list conozca los detalles internos de ninguno.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class IntegerBlockCodec(Protocol):
    """Codifica/decodifica una secuencia de enteros no negativos como bytes.

    `encode` añade los bytes codificados al final de `buffer` (permite componer
    varias secciones consecutivas en un mismo blob sin copias intermedias).
    `decode` lee exactamente `count` valores a partir de `offset` en `data` y
    devuelve `(valores, nuevo_offset)`, para que el llamador pueda encadenar la
    lectura de la siguiente sección sin conocer cuántos bytes ocupó esta.
    """

    def encode(self, values: Sequence[int], buffer: bytearray) -> None: ...

    def decode(self, data: bytes, offset: int, count: int) -> tuple[list[int], int]: ...
