import json
from pathlib import Path

import pytest

from index_compression_codec.pipeline import CompressionPipeline
from index_compression_codec.serialization import (
    CompressedIndexReader,
    read_compression_stats,
    read_manifest,
)


def _write_source_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    documents = [
        {"doc_id": 0, "url": "http://a", "title": "A", "length": 4},
        {"doc_id": 1, "url": "http://b", "title": "B", "length": 4},
    ]
    postings = [
        {
            "term": "engines",
            "document_frequency": 2,
            "postings": [
                {"doc_id": 0, "term_frequency": 1, "positions": [1]},
                {"doc_id": 1, "term_frequency": 1, "positions": [1]},
            ],
        },
        {
            "term": "rank",
            "document_frequency": 1,
            "postings": [{"doc_id": 0, "term_frequency": 1, "positions": [2]}],
        },
        {
            "term": "search",
            "document_frequency": 2,
            "postings": [
                {"doc_id": 0, "term_frequency": 1, "positions": [0]},
                {"doc_id": 1, "term_frequency": 1, "positions": [0]},
            ],
        },
    ]
    stats = {
        "total_documents": 2,
        "vocabulary_size": 3,
        "total_postings": 5,
        "average_document_length": 4.0,
    }

    (index_dir / "documents.jsonl").write_text(
        "\n".join(json.dumps(doc) for doc in documents) + "\n", encoding="utf-8"
    )
    (index_dir / "postings.jsonl").write_text(
        "\n".join(json.dumps(posting) for posting in postings) + "\n", encoding="utf-8"
    )
    (index_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


class TestCompressionPipeline:
    def test_writes_all_expected_output_files(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)

        CompressionPipeline().compress(source_dir, output_dir)

        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "documents.jsonl").exists()
        assert (output_dir / "stats.json").exists()
        assert (output_dir / "terms.jsonl").exists()
        assert (output_dir / "postings.bin").exists()
        assert (output_dir / "compression_stats.json").exists()

    def test_copies_documents_and_stats_verbatim(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)

        CompressionPipeline().compress(source_dir, output_dir)

        assert (output_dir / "documents.jsonl").read_text(encoding="utf-8") == (
            source_dir / "documents.jsonl"
        ).read_text(encoding="utf-8")
        assert (output_dir / "stats.json").read_text(encoding="utf-8") == (
            source_dir / "stats.json"
        ).read_text(encoding="utf-8")

    def test_manifest_declares_format_version_and_parameters(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)

        CompressionPipeline(bitpack_threshold=32, block_size=64).compress(source_dir, output_dir)
        manifest = read_manifest(output_dir)

        assert manifest["format_version"] == 1
        assert manifest["block_size"] == 64
        assert manifest["bitpack_threshold"] == 32

    def test_compression_stats_report_plausible_counts(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)

        stats = CompressionPipeline().compress(source_dir, output_dir)

        assert stats.total_documents == 2
        assert stats.vocabulary_size == 3
        assert stats.total_postings == 5
        assert stats.uncompressed_bytes > 0
        assert stats.compressed_bytes > 0
        assert read_compression_stats(output_dir) == stats

    def test_compressing_twice_is_byte_for_byte_deterministic(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _write_source_index(source_dir)

        first_output = tmp_path / "first"
        second_output = tmp_path / "second"
        CompressionPipeline().compress(source_dir, first_output)
        CompressionPipeline().compress(source_dir, second_output)

        assert (first_output / "postings.bin").read_bytes() == (
            second_output / "postings.bin"
        ).read_bytes()
        assert (first_output / "terms.jsonl").read_text(encoding="utf-8") == (
            second_output / "terms.jsonl"
        ).read_text(encoding="utf-8")

    def test_rejects_a_source_dir_missing_postings_file(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "documents.jsonl").write_text("", encoding="utf-8")
        (source_dir / "stats.json").write_text("{}", encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            CompressionPipeline().compress(source_dir, tmp_path / "out")


class TestCompressedIndexReaderRoundTrip:
    def test_random_access_decode_matches_original_postings_exactly(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        CompressionPipeline().compress(source_dir, output_dir)

        original = {
            entry["term"]: entry
            for entry in (
                json.loads(line)
                for line in (source_dir / "postings.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }

        with CompressedIndexReader(output_dir) as reader:
            for term, expected in original.items():
                postings_list = reader.lookup(term)
                assert postings_list is not None
                assert postings_list.term == term
                assert postings_list.document_frequency == expected["document_frequency"]
                decoded_postings = [
                    (posting.doc_id, posting.term_frequency, list(posting.positions))
                    for posting in postings_list.postings
                ]
                expected_postings = [
                    (entry["doc_id"], entry["term_frequency"], entry["positions"])
                    for entry in expected["postings"]
                ]
                assert decoded_postings == expected_postings

    def test_lookup_of_unknown_term_returns_none(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        CompressionPipeline().compress(source_dir, output_dir)

        with CompressedIndexReader(output_dir) as reader:
            assert reader.lookup("nonexistent") is None

    def test_terms_are_listed_alphabetically(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        CompressionPipeline().compress(source_dir, output_dir)

        with CompressedIndexReader(output_dir) as reader:
            assert reader.terms() == ["engines", "rank", "search"]

    def test_context_manager_closes_the_underlying_file(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        CompressionPipeline().compress(source_dir, output_dir)

        with CompressedIndexReader(output_dir) as reader:
            pass

        assert reader._file.closed  # se prueba explícitamente el cierre de recursos

    def test_document_frequency_lookup_matches_the_postings_list(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        CompressionPipeline().compress(source_dir, output_dir)

        with CompressedIndexReader(output_dir) as reader:
            assert reader.document_frequency("search") == 2
            assert reader.document_frequency("nonexistent") is None
