import pytest

from index_compression_codec.varint import VarintBlockCodec, decode_varint, encode_varint


class TestEncodeDecodeVarint:
    @pytest.mark.parametrize(
        "value",
        [0, 1, 127, 128, 300, 16383, 16384, 2**20, 2**32, 2**63, 2**64],
    )
    def test_round_trips_boundary_and_large_values(self, value: int) -> None:
        encoded = encode_varint(value)
        decoded, new_offset = decode_varint(encoded, 0)

        assert decoded == value
        assert new_offset == len(encoded)

    def test_small_values_fit_in_a_single_byte(self) -> None:
        assert len(encode_varint(0)) == 1
        assert len(encode_varint(127)) == 1

    def test_values_above_127_need_more_than_one_byte(self) -> None:
        assert len(encode_varint(128)) == 2

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError):
            encode_varint(-1)

    def test_decode_raises_on_truncated_buffer(self) -> None:
        truncated = encode_varint(128)[:1]  # falta el byte final, sin bit de continuación
        with pytest.raises(ValueError):
            decode_varint(truncated, 0)

    def test_decode_starts_at_the_given_offset(self) -> None:
        buffer = encode_varint(5) + encode_varint(300)
        first, offset_after_first = decode_varint(buffer, 0)
        second, offset_after_second = decode_varint(buffer, offset_after_first)

        assert first == 5
        assert second == 300
        assert offset_after_second == len(buffer)


class TestVarintBlockCodec:
    def test_encode_decode_round_trips_a_sequence(self) -> None:
        codec = VarintBlockCodec()
        values = [0, 1, 127, 128, 999999]
        buffer = bytearray()

        codec.encode(values, buffer)
        decoded, new_offset = codec.decode(bytes(buffer), 0, len(values))

        assert decoded == values
        assert new_offset == len(buffer)

    def test_encode_appends_to_an_existing_buffer(self) -> None:
        codec = VarintBlockCodec()
        buffer = bytearray(b"\xff")

        codec.encode([1, 2], buffer)

        assert buffer[0] == 0xFF
        decoded, new_offset = codec.decode(bytes(buffer), 1, 2)
        assert decoded == [1, 2]
        assert new_offset == len(buffer)

    def test_decode_zero_count_consumes_nothing(self) -> None:
        codec = VarintBlockCodec()
        decoded, new_offset = codec.decode(b"\x05", 0, 0)

        assert decoded == []
        assert new_offset == 0
