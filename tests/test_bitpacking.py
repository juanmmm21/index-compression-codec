import pytest

from index_compression_codec.bitpacking import (
    PackedBlockCodec,
    _bit_width_for,
    _pack_block,
    _unpack_block,
)


class TestBitWidthFor:
    def test_empty_sequence_needs_zero_bits(self) -> None:
        assert _bit_width_for([]) == 0

    def test_all_zero_values_need_zero_bits(self) -> None:
        assert _bit_width_for([0, 0, 0]) == 0

    def test_uses_the_bit_length_of_the_maximum_value(self) -> None:
        assert _bit_width_for([1, 2, 3]) == 2  # 3 == 0b11
        assert _bit_width_for([1, 8, 2]) == 4  # 8 == 0b1000

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError):
            _bit_width_for([-1])


class TestPackUnpackBlock:
    def test_round_trips_values_at_an_exact_bit_width(self) -> None:
        values = [0, 1, 2, 3, 3, 0, 2]
        bit_width = _bit_width_for(values)

        packed = _pack_block(values, bit_width)
        unpacked, consumed = _unpack_block(packed, 0, len(values), bit_width)

        assert unpacked == values
        assert consumed == len(packed)

    def test_zero_bit_width_round_trips_to_all_zeros(self) -> None:
        packed = _pack_block([0, 0, 0], 0)
        unpacked, consumed = _unpack_block(packed, 0, 3, 0)

        assert packed == b""
        assert unpacked == [0, 0, 0]
        assert consumed == 0

    def test_rejects_a_value_that_does_not_fit_the_bit_width(self) -> None:
        with pytest.raises(ValueError):
            _pack_block([4], bit_width=2)  # 4 necesita 3 bits, no 2

    def test_unpack_raises_on_truncated_data(self) -> None:
        packed = _pack_block([1, 2, 3], bit_width=2)
        with pytest.raises(ValueError):
            _unpack_block(packed[:-1] if len(packed) > 1 else b"", 0, 10, bit_width=2)


class TestPackedBlockCodec:
    def test_round_trips_a_single_block(self) -> None:
        codec = PackedBlockCodec(block_size=128)
        values = [1, 5, 3, 300, 7]
        buffer = bytearray()

        codec.encode(values, buffer)
        decoded, new_offset = codec.decode(bytes(buffer), 0, len(values))

        assert decoded == values
        assert new_offset == len(buffer)

    def test_round_trips_values_spanning_multiple_blocks(self) -> None:
        codec = PackedBlockCodec(block_size=4)
        values = list(range(1, 15))  # 14 valores, 4 bloques (4+4+4+2)
        buffer = bytearray()

        codec.encode(values, buffer)
        decoded, new_offset = codec.decode(bytes(buffer), 0, len(values))

        assert decoded == values
        assert new_offset == len(buffer)

    def test_each_block_uses_its_own_minimal_bit_width(self) -> None:
        # Primer bloque necesita pocos bits, segundo bloque un valor grande:
        # el tamaño no debe estar dominado por el máximo global.
        codec = PackedBlockCodec(block_size=2)
        small_block_values = [1, 1]
        large_block_values = [100_000, 1]
        buffer_small = bytearray()
        buffer_mixed = bytearray()

        codec.encode(small_block_values + [1, 1], buffer_small)
        codec.encode(small_block_values + large_block_values, buffer_mixed)

        assert len(buffer_mixed) > len(buffer_small)

    def test_round_trips_an_empty_sequence(self) -> None:
        codec = PackedBlockCodec(block_size=128)
        buffer = bytearray()

        codec.encode([], buffer)
        decoded, new_offset = codec.decode(bytes(buffer), 0, 0)

        assert decoded == []
        assert new_offset == 0
        assert buffer == bytearray()

    def test_rejects_a_non_positive_block_size(self) -> None:
        with pytest.raises(ValueError):
            PackedBlockCodec(block_size=0)

    def test_decode_raises_on_truncated_block_header(self) -> None:
        codec = PackedBlockCodec(block_size=128)
        with pytest.raises(ValueError):
            codec.decode(b"", 0, 5)
