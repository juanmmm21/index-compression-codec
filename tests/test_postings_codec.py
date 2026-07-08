import pytest

from index_compression_codec.models import EncodingMode, Posting, PostingsList
from index_compression_codec.postings_codec import decode_postings_list, encode_postings_list


def _postings_list(term: str, postings: tuple[Posting, ...]) -> PostingsList:
    return PostingsList(term=term, document_frequency=len(postings), postings=postings)


class TestRoundTrip:
    def test_empty_postings_list(self) -> None:
        postings_list = _postings_list("empty", ())

        encoded = encode_postings_list(postings_list)
        decoded = decode_postings_list("empty", 0, encoded)

        assert decoded == postings_list

    def test_single_posting_single_position(self) -> None:
        postings_list = _postings_list(
            "term", (Posting(doc_id=7, term_frequency=1, positions=(3,)),)
        )

        encoded = encode_postings_list(postings_list)
        decoded = decode_postings_list("term", 1, encoded)

        assert decoded == postings_list

    def test_single_posting_multiple_positions(self) -> None:
        postings_list = _postings_list(
            "term", (Posting(doc_id=0, term_frequency=3, positions=(0, 5, 42)),)
        )

        encoded = encode_postings_list(postings_list)
        decoded = decode_postings_list("term", 1, encoded)

        assert decoded == postings_list

    def test_posting_with_no_recorded_positions(self) -> None:
        postings_list = _postings_list("term", (Posting(doc_id=0, term_frequency=0, positions=()),))

        encoded = encode_postings_list(postings_list)
        decoded = decode_postings_list("term", 1, encoded)

        assert decoded == postings_list

    def test_multiple_postings_below_bitpack_threshold_use_varint(self) -> None:
        postings = tuple(
            Posting(doc_id=doc_id, term_frequency=1, positions=(0,)) for doc_id in range(5)
        )
        postings_list = _postings_list("term", postings)

        encoded = encode_postings_list(postings_list, bitpack_threshold=64)

        assert encoded[0] == int(EncodingMode.VARINT)
        assert decode_postings_list("term", 5, encoded) == postings_list

    def test_multiple_postings_above_bitpack_threshold_use_packed(self) -> None:
        postings = tuple(
            Posting(doc_id=doc_id * 2, term_frequency=1, positions=(0,)) for doc_id in range(200)
        )
        postings_list = _postings_list("term", postings)

        encoded = encode_postings_list(postings_list, bitpack_threshold=64, block_size=16)

        assert encoded[0] == int(EncodingMode.PACKED)
        assert decode_postings_list("term", 200, encoded, block_size=16) == postings_list

    def test_very_long_postings_list_round_trips(self) -> None:
        postings = tuple(
            Posting(
                doc_id=doc_id,
                term_frequency=(doc_id % 7) + 1,
                positions=tuple(range(doc_id % 5 + 1)),
            )
            for doc_id in range(0, 10_000, 3)
        )
        postings_list = _postings_list("frequent", postings)

        encoded = encode_postings_list(postings_list, bitpack_threshold=64)
        decoded = decode_postings_list("frequent", len(postings), encoded)

        assert decoded == postings_list

    def test_non_contiguous_doc_id_gaps_round_trip(self) -> None:
        postings_list = _postings_list(
            "term",
            (
                Posting(doc_id=1, term_frequency=1, positions=(0,)),
                Posting(doc_id=100, term_frequency=2, positions=(1, 9)),
                Posting(doc_id=101, term_frequency=1, positions=(0,)),
            ),
        )

        encoded = encode_postings_list(postings_list, bitpack_threshold=2)
        decoded = decode_postings_list("term", 3, encoded)

        assert decoded == postings_list


class TestValidation:
    def test_rejects_doc_ids_not_strictly_ascending(self) -> None:
        postings_list = _postings_list(
            "term",
            (
                Posting(doc_id=5, term_frequency=1, positions=(0,)),
                Posting(doc_id=5, term_frequency=1, positions=(0,)),
            ),
        )

        with pytest.raises(ValueError):
            encode_postings_list(postings_list)

    def test_rejects_doc_ids_out_of_order(self) -> None:
        postings_list = _postings_list(
            "term",
            (
                Posting(doc_id=5, term_frequency=1, positions=(0,)),
                Posting(doc_id=3, term_frequency=1, positions=(0,)),
            ),
        )

        with pytest.raises(ValueError):
            encode_postings_list(postings_list)

    def test_rejects_positions_not_strictly_ascending(self) -> None:
        postings_list = _postings_list(
            "term", (Posting(doc_id=0, term_frequency=2, positions=(3, 3)),)
        )

        with pytest.raises(ValueError):
            encode_postings_list(postings_list)

    def test_decode_rejects_document_frequency_mismatch(self) -> None:
        postings_list = _postings_list(
            "term", (Posting(doc_id=0, term_frequency=1, positions=(0,)),)
        )
        encoded = encode_postings_list(postings_list)

        with pytest.raises(ValueError):
            decode_postings_list("term", 2, encoded)

    def test_decode_rejects_empty_blob(self) -> None:
        with pytest.raises(ValueError):
            decode_postings_list("term", 0, b"")
