import json
from pathlib import Path

import pytest

from index_compression_codec.__main__ import (
    _build_parser,
    _run_benchmark,
    _run_compress,
    _run_lookup,
    _run_stats,
)


def _write_source_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    documents = [{"doc_id": 0, "url": "http://a", "title": "A", "length": 4}]
    postings = [
        {
            "term": "search",
            "document_frequency": 1,
            "postings": [{"doc_id": 0, "term_frequency": 1, "positions": [0]}],
        },
        {
            "term": "engines",
            "document_frequency": 1,
            "postings": [{"doc_id": 0, "term_frequency": 1, "positions": [1]}],
        },
    ]
    stats = {
        "total_documents": 1,
        "vocabulary_size": 2,
        "total_postings": 2,
        "average_document_length": 4.0,
    }
    (index_dir / "documents.jsonl").write_text(
        "\n".join(json.dumps(doc) for doc in documents) + "\n", encoding="utf-8"
    )
    (index_dir / "postings.jsonl").write_text(
        "\n".join(json.dumps(posting) for posting in postings) + "\n", encoding="utf-8"
    )
    (index_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


class TestArgumentParsing:
    def test_compress_requires_output_dir(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compress", "source_dir"])

    def test_lookup_requires_a_term(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["lookup", "index_dir"])

    def test_unknown_command_exits(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["bogus"])


class TestRunCompress:
    def test_reports_missing_source_dir(self, tmp_path: Path) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["compress", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")]
        )

        assert _run_compress(args) == 2

    def test_compresses_and_writes_expected_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        parser = _build_parser()
        args = parser.parse_args(["compress", str(source_dir), "--output-dir", str(output_dir)])

        exit_code = _run_compress(args)

        assert exit_code == 0
        assert (output_dir / "postings.bin").exists()
        assert (output_dir / "terms.jsonl").exists()
        output = capsys.readouterr().out
        assert "2 términos" in output


class TestRunStats:
    def test_reports_missing_index_dir(self, tmp_path: Path) -> None:
        parser = _build_parser()
        args = parser.parse_args(["stats", str(tmp_path / "missing")])

        assert _run_stats(args) == 2

    def test_prints_compression_stats(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        compress_parser = _build_parser()
        _run_compress(
            compress_parser.parse_args(
                ["compress", str(source_dir), "--output-dir", str(output_dir)]
            )
        )

        stats_parser = _build_parser()
        exit_code = _run_stats(stats_parser.parse_args(["stats", str(output_dir)]))

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Documentos: 1" in output
        assert "Vocabulario: 2" in output


class TestRunLookup:
    def test_reports_missing_index_dir(self, tmp_path: Path) -> None:
        parser = _build_parser()
        args = parser.parse_args(["lookup", str(tmp_path / "missing"), "search"])

        assert _run_lookup(args) == 2

    def test_prints_postings_for_an_existing_term(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        compress_parser = _build_parser()
        _run_compress(
            compress_parser.parse_args(
                ["compress", str(source_dir), "--output-dir", str(output_dir)]
            )
        )

        lookup_parser = _build_parser()
        exit_code = _run_lookup(lookup_parser.parse_args(["lookup", str(output_dir), "search"]))

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "doc_id=0" in output

    def test_reports_absent_term_without_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        compress_parser = _build_parser()
        _run_compress(
            compress_parser.parse_args(
                ["compress", str(source_dir), "--output-dir", str(output_dir)]
            )
        )

        lookup_parser = _build_parser()
        exit_code = _run_lookup(
            lookup_parser.parse_args(["lookup", str(output_dir), "nonexistent"])
        )

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "no aparece" in output


class TestRunBenchmark:
    def test_reports_missing_index_dir(self, tmp_path: Path) -> None:
        parser = _build_parser()
        args = parser.parse_args(["benchmark", str(tmp_path / "missing")])

        assert _run_benchmark(args) == 2

    def test_prints_latency_and_ratio(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "compressed"
        _write_source_index(source_dir)
        compress_parser = _build_parser()
        _run_compress(
            compress_parser.parse_args(
                ["compress", str(source_dir), "--output-dir", str(output_dir)]
            )
        )

        benchmark_parser = _build_parser()
        exit_code = _run_benchmark(
            benchmark_parser.parse_args(["benchmark", str(output_dir), "--sample-size", "10"])
        )

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Términos consultados: 2" in output
        assert "Latencia media por lookup" in output
        assert "Ratio de compresión" in output
