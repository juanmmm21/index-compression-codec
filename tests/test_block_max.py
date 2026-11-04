import json
import math
import random
from pathlib import Path

import pytest

from index_compression_codec.__main__ import main
from index_compression_codec.block_max import (
    BlockMaxParams,
    bm25_tf_part,
    decode_block,
    decode_block_directory,
    decode_block_postings_list,
    encode_block_postings_list,
    float32_ceil,
)
from index_compression_codec.models import Posting, PostingsList
from index_compression_codec.pipeline import CompressionPipeline
from index_compression_codec.serialization import CompressedIndexReader, read_manifest

PARAMS = BlockMaxParams()


def _random_postings(seed: int, count: int) -> tuple[PostingsList, dict[int, int]]:
    rng = random.Random(seed)
    doc_id = rng.randint(0, 5)
    postings = []
    lengths: dict[int, int] = {}
    for _ in range(count):
        tf = rng.randint(1, 6)
        positions = tuple(sorted(rng.sample(range(200), tf)))
        postings.append(Posting(doc_id=doc_id, term_frequency=tf, positions=positions))
        lengths[doc_id] = rng.randint(tf, 400)
        doc_id += rng.randint(1, 40)
    return PostingsList("t", len(postings), tuple(postings)), lengths


@pytest.mark.parametrize("count", [1, 5, 64, 65, 300])
def test_round_trip_and_bounds_hold_for_every_posting(count: int) -> None:
    postings_list, lengths = _random_postings(count, count)
    data, term_max = encode_block_postings_list(
        postings_list, lengths, 120.0, params=PARAMS, block_size=64, bitpack_threshold=32
    )
    assert decode_block_postings_list("t", count, data, block_size=64) == postings_list

    directory = decode_block_directory(data)
    assert len(directory.blocks) == math.ceil(count / 64)
    start = 0
    for block in directory.blocks:
        chunk = postings_list.postings[start : start + block.num_postings]
        start += block.num_postings
        assert block.last_doc_id == chunk[-1].doc_id
        for posting in chunk:
            exact = bm25_tf_part(posting.term_frequency, lengths[posting.doc_id], 120.0, PARAMS)
            assert block.max_score >= exact
            # La cota independiente de parámetros también acota.
            assert (
                bm25_tf_part(block.max_term_frequency, block.min_document_length, 120.0, PARAMS)
                >= exact
            )
    assert math.isclose(term_max, max(block.max_score for block in directory.blocks))


def test_single_block_can_be_decoded_without_the_rest() -> None:
    postings_list, lengths = _random_postings(7, 200)
    data, _ = encode_block_postings_list(postings_list, lengths, 80.0, params=PARAMS, block_size=64)
    directory = decode_block_directory(data)
    offsets = directory.block_offsets()
    third = decode_block(
        data, offsets[2], directory.blocks[2], directory.blocks[1].last_doc_id, block_size=64
    )
    assert third == list(postings_list.postings[128:192])


def test_corrupt_block_is_rejected() -> None:
    postings_list, lengths = _random_postings(3, 10)
    data, _ = encode_block_postings_list(postings_list, lengths, 80.0, params=PARAMS, block_size=64)
    with pytest.raises(ValueError):
        decode_block_postings_list("t", 11, data, block_size=64)
    with pytest.raises(ValueError):
        decode_block_directory(data[:4])


def test_unknown_document_length_fails_loudly() -> None:
    postings_list, _ = _random_postings(3, 10)
    with pytest.raises(ValueError, match="documents.jsonl"):
        encode_block_postings_list(postings_list, {}, 80.0, params=PARAMS)


def test_float32_ceil_never_rounds_down() -> None:
    for value in (0.1, 1.0 / 3.0, 2.2250738585072014e-308, 123456.789, -0.1, 2.318840579710145):
        assert float32_ceil(value) >= value


def _write_source_index(root: Path) -> Path:
    index_dir = root / "index"
    index_dir.mkdir()
    documents = [{"doc_id": i, "url": f"u{i}", "title": "", "length": 10 + i} for i in range(150)]
    (index_dir / "documents.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in documents), encoding="utf-8"
    )
    postings = [
        {
            "term": "common",
            "document_frequency": 150,
            "postings": [
                {"doc_id": i, "term_frequency": 1 + i % 3, "positions": list(range(1 + i % 3))}
                for i in range(150)
            ],
        },
        {
            "term": "rare",
            "document_frequency": 1,
            "postings": [{"doc_id": 7, "term_frequency": 2, "positions": [0, 4]}],
        },
    ]
    (index_dir / "postings.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in postings), encoding="utf-8"
    )
    stats = {
        "total_documents": 150,
        "vocabulary_size": 2,
        "total_postings": 151,
        "average_document_length": 84.5,
    }
    (index_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return index_dir


def test_v2_index_decodes_exactly_like_v1(tmp_path: Path) -> None:
    source = _write_source_index(tmp_path)
    CompressionPipeline().compress(source, tmp_path / "v1")
    CompressionPipeline(block_size=64, block_max=PARAMS).compress(source, tmp_path / "v2")

    assert read_manifest(tmp_path / "v1")["format_version"] == 1
    manifest = read_manifest(tmp_path / "v2")
    assert manifest["format_version"] == 2
    assert manifest["block_max"] == {
        "scoring": "bm25",
        "k1": 1.5,
        "b": 0.75,
        "average_document_length": 84.5,
    }

    with CompressedIndexReader(tmp_path / "v1") as v1, CompressedIndexReader(tmp_path / "v2") as v2:
        for term in ("common", "rare"):
            assert v1.lookup(term) == v2.lookup(term)
        assert v1.term_max_score("common") is None
        assert v2.block_max_params == PARAMS
        common_max = v2.term_max_score("common")
        assert common_max is not None and common_max > 0
        directory = v2.block_directory("common")
        assert directory is not None and len(directory.blocks) == 3
        with pytest.raises(ValueError):
            v1.block_directory("common")


def test_unsupported_format_version_still_fails(tmp_path: Path) -> None:
    source = _write_source_index(tmp_path)
    CompressionPipeline().compress(source, tmp_path / "out")
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no soportado"):
        read_manifest(tmp_path / "out")


def test_cli_block_max_flag(tmp_path: Path) -> None:
    source = _write_source_index(tmp_path)
    assert main(["compress", str(source), "--output-dir", str(tmp_path / "o"), "--block-max"]) == 0
    manifest = read_manifest(tmp_path / "o")
    assert manifest["format_version"] == 2 and manifest["block_size"] == 64
    assert main(["lookup", str(tmp_path / "o"), "rare"]) == 0
