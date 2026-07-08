from index_compression_codec.models import CompressionStats, TermEntry


class TestTermEntry:
    def test_json_round_trip(self) -> None:
        entry = TermEntry(term="search", offset=128, length=42, document_frequency=7)

        restored = TermEntry.from_json_dict(entry.to_json_dict())

        assert restored == entry


class TestCompressionStats:
    def test_json_round_trip(self) -> None:
        stats = CompressionStats(
            total_documents=100,
            vocabulary_size=500,
            total_postings=2000,
            uncompressed_bytes=100_000,
            compressed_bytes=25_000,
        )

        restored = CompressionStats.from_json_dict(stats.to_json_dict())

        assert restored == stats

    def test_compression_ratio_divides_uncompressed_by_compressed(self) -> None:
        stats = CompressionStats(
            total_documents=1,
            vocabulary_size=1,
            total_postings=1,
            uncompressed_bytes=100,
            compressed_bytes=25,
        )

        assert stats.compression_ratio == 4.0

    def test_compression_ratio_is_zero_when_compressed_bytes_is_zero(self) -> None:
        stats = CompressionStats(
            total_documents=0,
            vocabulary_size=0,
            total_postings=0,
            uncompressed_bytes=0,
            compressed_bytes=0,
        )

        assert stats.compression_ratio == 0.0

    def test_to_json_dict_includes_the_derived_ratio(self) -> None:
        stats = CompressionStats(
            total_documents=1,
            vocabulary_size=1,
            total_postings=1,
            uncompressed_bytes=100,
            compressed_bytes=50,
        )

        assert stats.to_json_dict()["compression_ratio"] == 2.0
