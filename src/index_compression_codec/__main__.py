"""CLI de `index-compression-codec`: `compress` (comprime un índice de
`inverted-index-builder`), `stats` (métricas de compresión ya calculadas),
`lookup` (postings de un término concreto vía acceso aleatorio, sin
descomprimir el índice entero) y `benchmark` (ratio de compresión y latencia
de decodificación medidas sobre un índice ya comprimido).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from index_compression_codec.bitpacking import DEFAULT_BLOCK_SIZE
from index_compression_codec.pipeline import CompressionPipeline
from index_compression_codec.postings_codec import DEFAULT_BITPACK_THRESHOLD
from index_compression_codec.serialization import CompressedIndexReader, read_compression_stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="index-compression-codec",
        description="Comprime y consulta listas de postings de inverted-index-builder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser(
        "compress", help="Comprime un índice producido por inverted-index-builder"
    )
    compress_parser.add_argument(
        "source_index_dir", type=Path, help="Directorio de índice sin comprimir"
    )
    compress_parser.add_argument("--output-dir", type=Path, required=True)
    compress_parser.add_argument(
        "--bitpack-threshold",
        type=int,
        default=DEFAULT_BITPACK_THRESHOLD,
        help="document_frequency mínima para usar bitpacking en vez de varint",
    )
    compress_parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Tamaño de bloque de bitpacking",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="Muestra las métricas de compresión de un índice ya comprimido"
    )
    stats_parser.add_argument("index_dir", type=Path)

    lookup_parser = subparsers.add_parser(
        "lookup", help="Decodifica los postings de un término concreto (acceso aleatorio)"
    )
    lookup_parser.add_argument("index_dir", type=Path)
    lookup_parser.add_argument("term", type=str)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Mide latencia de decodificación aleatoria sobre el índice"
    )
    benchmark_parser.add_argument("index_dir", type=Path)
    benchmark_parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Número máximo de términos a decodificar para medir latencia",
    )

    return parser


def _run_compress(args: argparse.Namespace) -> int:
    source_index_dir: Path = args.source_index_dir
    if not source_index_dir.exists():
        print(f"Error: no existe {source_index_dir}", file=sys.stderr)
        return 2

    pipeline = CompressionPipeline(
        bitpack_threshold=args.bitpack_threshold, block_size=args.block_size
    )
    stats = pipeline.compress(source_index_dir, args.output_dir)

    print(
        f"Índice comprimido: {stats.total_documents} documentos, "
        f"{stats.vocabulary_size} términos, {stats.total_postings} postings, "
        f"{stats.uncompressed_bytes} -> {stats.compressed_bytes} bytes "
        f"(ratio {stats.compression_ratio:.2f}x)"
    )
    return 0


def _run_stats(args: argparse.Namespace) -> int:
    index_dir: Path = args.index_dir
    if not index_dir.exists():
        print(f"Error: no existe {index_dir}", file=sys.stderr)
        return 2

    stats = read_compression_stats(index_dir)
    print(f"Documentos: {stats.total_documents}")
    print(f"Vocabulario: {stats.vocabulary_size}")
    print(f"Postings totales: {stats.total_postings}")
    print(f"Bytes sin comprimir: {stats.uncompressed_bytes}")
    print(f"Bytes comprimidos: {stats.compressed_bytes}")
    print(f"Ratio de compresión: {stats.compression_ratio:.2f}x")
    return 0


def _run_lookup(args: argparse.Namespace) -> int:
    index_dir: Path = args.index_dir
    if not index_dir.exists():
        print(f"Error: no existe {index_dir}", file=sys.stderr)
        return 2

    with CompressedIndexReader(index_dir) as reader:
        postings_list = reader.lookup(args.term)
        if postings_list is None:
            print(f"'{args.term}' no aparece en el índice")
            return 0

        print(f"'{postings_list.term}' — document_frequency={postings_list.document_frequency}")
        for posting in postings_list.postings:
            print(
                f"  doc_id={posting.doc_id} tf={posting.term_frequency} "
                f"positions={list(posting.positions)}"
            )
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    index_dir: Path = args.index_dir
    if not index_dir.exists():
        print(f"Error: no existe {index_dir}", file=sys.stderr)
        return 2

    stats = read_compression_stats(index_dir)
    with CompressedIndexReader(index_dir) as reader:
        sample_terms = reader.terms()[: args.sample_size]
        if not sample_terms:
            print("El índice no tiene términos que consultar")
            return 0

        start = time.perf_counter()
        for term in sample_terms:
            reader.lookup(term)
        elapsed_seconds = time.perf_counter() - start

    average_microseconds = (elapsed_seconds / len(sample_terms)) * 1_000_000
    print(f"Términos consultados: {len(sample_terms)}")
    print(f"Tiempo total: {elapsed_seconds * 1000:.2f} ms")
    print(f"Latencia media por lookup: {average_microseconds:.2f} µs")
    print(f"Ratio de compresión del índice: {stats.compression_ratio:.2f}x")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "compress":
        return _run_compress(args)
    if args.command == "stats":
        return _run_stats(args)
    if args.command == "lookup":
        return _run_lookup(args)
    if args.command == "benchmark":
        return _run_benchmark(args)

    parser.error(f"Comando desconocido: {args.command}")
    return 2  # inalcanzable: parser.error() termina el proceso con sys.exit()


if __name__ == "__main__":
    sys.exit(main())
