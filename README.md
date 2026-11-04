# index-compression-codec

**Project 4/10 of the [`beacon-search-engine`](https://github.com/juanmmm21/beacon-search-engine) ecosystem** — *Indexing* category.
Repository: [`github.com/juanmmm21/index-compression-codec`](https://github.com/juanmmm21/index-compression-codec)

A from-scratch postings-list compression codec: delta encoding over ascending
`doc_id`s, LEB128 varint encoding for variable-length integers, and block-based
bitpacking for the longest lists. No `zlib`, no `gzip`, no general-purpose
compression library — the whole point is implementing the integer-encoding
techniques real search engines (Lucene, Elasticsearch) use internally for
postings, not delegating the problem to a byte-oriented compressor that knows
nothing about the structure of an inverted index.

## What problem it solves

An uncompressed inverted index stores every `doc_id`, term frequency and token
position as a full-width integer (4 or 8 bytes each), even though the values
that actually matter for search — how far apart two consecutive documents or
positions are — are almost always small. [`inverted-index-builder`](https://github.com/juanmmm21/inverted-index-builder)
already guarantees that postings are sorted by ascending `doc_id`, which is
exactly the precondition delta encoding needs: instead of storing absolute
`doc_id`s, store the *gaps* between consecutive ones, which are typically one
or two orders of magnitude smaller. Combined with varint encoding (small
integers use fewer bytes) and block-based bitpacking (each block of gaps uses
only the exact number of bits its largest value needs, not a fixed byte
width), this is what lets a real inverted index scale to millions of documents
without postings lists dominating disk and memory — the requirement stated
explicitly in `beacon-search-engine`'s engineering standards (`../CLAUDE.md`,
section 2.E). Just as important as the compression ratio itself: the format is
designed for **random access** — decoding the postings of a single term never
requires reading or decompressing the rest of the index.

## Role in `beacon-search-engine`

```text
inverted-index-builder
        │ documents.jsonl, postings.jsonl, stats.json (uncompressed)
        ▼
┌────────────────────────────┐
│  index-compression-codec    │   (this project)
│  delta + varint + bitpacking │
└─────────────┬────────────────┘
              │ documents.jsonl, stats.json (copied through),
              │ terms.jsonl, postings.bin, compression_stats.json
              ▼
   bm25-ranking-engine ── pagerank-link-analysis ── distributed-index-sharding
              │
              ▼
     (converges in beacon-search-console)
```

This project sits directly on top of `inverted-index-builder` in the indexing
stage of the pipeline: it consumes the uncompressed index that project
produces and hands downstream ranking modules (`bm25-ranking-engine`,
`pagerank-link-analysis`, `distributed-index-sharding`) a drop-in replacement
index directory — same `documents.jsonl` and `stats.json`, with postings now
compressed and randomly addressable by term.

## Goal and skills demonstrated

- **Delta encoding over sorted integers**: exploiting the ascending-`doc_id`
  invariant `inverted-index-builder` already guarantees to turn large absolute
  IDs into small, cheaply-encoded gaps — the single biggest compression win in
  this codec, applied before any bit-level trick.
- **LEB128 varint encoding from scratch**: a from-scratch implementation of
  the variable-length integer encoding used by Protocol Buffers, SQLite and
  Lucene, as the universal fallback for every integer this codec writes.
- **Block-based bitpacking**: a from-scratch frame-of-reference bitpacking
  scheme — not a fixed 16/32-bit width, but the *minimum* bit width each
  128-value block actually needs — applied selectively to the longest postings
  lists, where its one-byte-per-block header overhead is worth paying.
  Deliberately not Elias-Fano: bitpacking with per-block frame-of-reference
  gives most of the same benefit (near-optimal bits for small deltas) with a
  much simpler, more debuggable implementation, which fits `beacon-search-engine`'s
  stated preference for simplicity over premature over-engineering
  (`../AGENTS.md`, section 1).
- **A self-describing binary format built for random access**: every encoded
  postings list is a fully self-contained blob (its own header declares the
  encoding mode and length), addressed by a term directory of `(offset,
  length)` pairs — so looking up one term's postings never touches the rest of
  the index, the central requirement for this piece of the ecosystem.
- **Streaming compression with constant memory overhead**: the encoder never
  materializes the whole index in memory — it reads `postings.jsonl` line by
  line and writes directly into the compressed binary blob, which is what
  actually makes "scales to millions of documents" true rather than aspirational.

## How it works

1. **Reading** (`pipeline.py`): `CompressionPipeline` streams `postings.jsonl`
   from an `inverted-index-builder` index directory line by line — each line
   is one term's full postings list, parsed into this package's own `PostingsList`
   dataclass (no cross-repo Python import, per `../AGENTS.md`).
2. **Per-list encoding** (`postings_codec.py`): for each list, the `doc_id`s
   are split into a first absolute value plus ascending gaps; frequencies and
   per-posting positions are similarly delta-encoded (positions) or kept as-is
   (frequencies, which aren't monotonic). The `doc_id` gaps are encoded with
   varint if the list is shorter than `bitpack_threshold` postings (default 64),
   or with block-based bitpacking otherwise — the mode is written into the
   blob's header so decoding never needs to know which threshold built it.
3. **Integer encoding** (`varint.py`, `bitpacking.py`): varint is plain LEB128.
   Bitpacking splits a sequence into fixed-size blocks (default 128 values),
   computes the minimum bit width the largest value in each block needs, and
   packs every value in that block at exactly that width.
4. **Writing** (`pipeline.py` + `serialization.py`): each list's encoded bytes
   are appended to `postings.bin`, and its `(term, offset, length,
   document_frequency)` is recorded in `terms.jsonl`. `documents.jsonl` and
   `stats.json` are copied through unmodified, and a `manifest.json` records
   the format version and the exact `block_size`/`bitpack_threshold` used, so
   the format stays self-describing across future parameter changes.
5. **Random-access decoding** (`serialization.py`): `CompressedIndexReader`
   loads only the term directory into memory (proportional to vocabulary
   size, not to postings count), keeps one open file handle to `postings.bin`,
   and on `lookup(term)` seeks directly to that term's byte range and decodes
   only those bytes.

## Architecture

```text
src/index_compression_codec/
├── models.py            # Posting, PostingsList, EncodingMode, TermEntry, CompressionStats
├── protocols.py          # IntegerBlockCodec interface
├── varint.py               # LEB128 varint primitives + VarintBlockCodec
├── bitpacking.py             # block-based bitpacking primitives + PackedBlockCodec
├── postings_codec.py           # binary format of a single postings list (encode/decode)
├── block_max.py                 # v2 block layout with per-block BM25 upper bounds
├── pipeline.py                   # CompressionPipeline: streams postings.jsonl -> postings.bin
├── serialization.py                # on-disk format (manifest, terms directory) + CompressedIndexReader
└── __main__.py                       # CLI (compress, stats, lookup, benchmark)
```

## Requirements and installation

- Python `>=3.11`
- No runtime dependencies — varint encoding and bitpacking are implemented
  entirely on the standard library's integer and `bytearray` operations, by
  design (see `../AGENTS.md`: no `zlib`/`gzip` shortcuts).

```bash
git clone https://github.com/juanmmm21/index-compression-codec.git
cd index-compression-codec
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage (CLI)

```bash
# Compress an index produced by inverted-index-builder
index-compression-codec compress ./index --output-dir ./index-compressed

# Tune the bitpacking threshold and block size
index-compression-codec compress ./index --output-dir ./index-compressed \
    --bitpack-threshold 32 --block-size 256

# Compression metrics for an already-compressed index
index-compression-codec stats ./index-compressed

# Random-access lookup of a single term's postings — doc_id, term frequency, positions
index-compression-codec lookup ./index-compressed search

# Decode latency benchmark over a sample of terms
index-compression-codec benchmark ./index-compressed --sample-size 5000
```

Try it end-to-end against the tiny fixture checked into `samples/tiny_example/`
(a 3-document, 82-term index already built by `inverted-index-builder`):

```bash
index-compression-codec compress samples/tiny_example/index --output-dir /tmp/compressed-example
index-compression-codec stats /tmp/compressed-example
index-compression-codec lookup /tmp/compressed-example search
```

It can also be invoked as a module: `python -m index_compression_codec compress ...`.

## Data formats

**Input** — an index directory produced by `inverted-index-builder`:
`documents.jsonl`, `postings.jsonl` (postings sorted by ascending `doc_id`
within each term) and `stats.json` (see that project's README for the exact
schema).

**Output** — `--output-dir` produces a compressed index directory with six files:

**`manifest.json`** — format version, file names and the exact codec
parameters used, so a future incompatible format change fails loudly instead
of being decoded silently wrong:

```json
{
  "format_version": 1,
  "documents_file": "documents.jsonl",
  "stats_file": "stats.json",
  "terms_file": "terms.jsonl",
  "postings_file": "postings.bin",
  "compression_stats_file": "compression_stats.json",
  "block_size": 128,
  "bitpack_threshold": 64
}
```

**`documents.jsonl`**, **`stats.json`** — copied verbatim from the source
index; every downstream module that reads them doesn't need to know this
project exists.

**`terms.jsonl`** — the term directory that makes random access possible: one
line per term, ordered alphabetically, pointing into `postings.bin`:

```json
{"term": "search", "offset": 15823, "length": 47, "document_frequency": 2}
```

**`postings.bin`** — the concatenation of one self-contained binary blob per
term, in the exact byte layout documented in `postings_codec.py`:

```text
byte    encoding_mode        0 = VARINT, 1 = PACKED
varint  num_postings
-- only if num_postings > 0 --
varint  first_doc_id
...     doc_id_gaps          num_postings - 1 values, varint or bitpacked
...     term_frequencies     num_postings values, always varint
-- per posting, same order as doc_id_gaps --
    varint  num_positions
    -- only if num_positions > 0 --
    varint  first_position
    ...     position_gaps    num_positions - 1 values, always varint
```

`encoding_mode` only governs the `doc_id` gaps: that's the section long,
frequent-term postings lists dominate, and the only one for which per-block
bitpacking's one-byte header overhead reliably pays for itself. Frequencies
and positions stay varint by deliberate scope decision — they're short
per-posting sequences where a bitpacking block rarely amortizes its overhead,
and frequencies in particular aren't monotonic, so there's no meaningful delta
to take. The first `doc_id` (and first position) of each list is stored
separately from its gap block rather than as "a gap from zero": folding it
into the first bitpacked block would force that whole block to use the bit
width of the raw absolute value — usually far larger than the real gaps
between consecutive postings — inflating the block for no benefit.

### Block-max layout (format version 2)

Added in `0.2.0` and opt-in: `index-compression-codec compress ... --block-max`
writes `format_version: 2`; without the flag the output is exactly the v1
format above, so every pre-0.2.0 reader keeps working. `CompressedIndexReader`
reads both versions.

Each term's posting list is split into blocks of `block_size` postings (64 by
default with `--block-max`), and a block directory is written in front of the
data. For every block it records where the block ends (`last_doc_id`), how
long it is, and upper bounds on what any document in it can score. That is
what Block-Max WAND in `bm25-ranking-engine` needs: skip a whole block without
decoding it when its bound cannot reach the current top-k threshold, and find
the next block without reading the current one.

The manifest records the BM25 parameters the bounds were computed with:

```json
{
  "format_version": 2,
  "block_size": 64,
  "bitpack_threshold": 64,
  "block_max": {"scoring": "bm25", "k1": 1.5, "b": 0.75, "average_document_length": 187.5},
  "...": "same file names as v1"
}
```

`terms.jsonl` lines gain a `max_score` field — the bound for the whole term —
so WAND can use it without opening the blob:

```json
{"term": "search", "offset": 15823, "length": 61, "document_frequency": 2, "max_score": 1.8211}
```

Each term blob in `postings.bin`:

```text
varint  num_postings
varint  num_blocks
-- block directory, one entry per block --
varint  last_doc_id_delta     block's last doc_id minus the previous block's (first: minus 0)
varint  block_postings
varint  block_byte_length
varint  max_term_frequency
varint  min_document_length
4 bytes max_score             float32 little-endian, rounded up
-- blocks, same order --
byte    encoding_mode         0 = VARINT, 1 = PACKED (doc_id gaps only)
varint  first_doc_id_delta    first doc_id minus the previous block's last (first block: doc_id + 1)
...     doc_id_gaps           block_postings - 1 values
...     term_frequencies      block_postings values, varint
-- per posting: positions exactly as in v1 --
```

What the bound means: BM25 for one term is `idf(t) * tf_part(tf, len)` with
`tf_part = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl))`. `idf`
depends on the whole corpus (for a sharded index, the global corpus, not the
shard), so it is not baked in: `max_score` is the exact maximum of `tf_part`
over the block's real postings, and the reader multiplies by its own idf. A
reader using different `k1`/`b` cannot use `max_score`, but can use
`tf_part(max_term_frequency, min_document_length)`, which bounds the block for
any parameters (`tf_part` grows with `tf` and shrinks with `len`). `max_score`
is rounded **up** to float32: a bound that underestimates by rounding would
make Block-Max WAND drop a document that belonged in the top-k.

**`compression_stats.json`** — corpus-wide compression metrics:

```json
{
  "total_documents": 20000,
  "vocabulary_size": 30000,
  "total_postings": 3132833,
  "uncompressed_bytes": 192851405,
  "compressed_bytes": 15508032,
  "compression_ratio": 12.436690843050087
}
```

## Programmatic usage

```python
from pathlib import Path

from index_compression_codec.pipeline import CompressionPipeline
from index_compression_codec.serialization import CompressedIndexReader

stats = CompressionPipeline().compress(Path("./index"), Path("./index-compressed"))
print(f"{stats.compression_ratio:.2f}x compression")

# Random access: only this term's bytes are read and decoded.
with CompressedIndexReader(Path("./index-compressed")) as reader:
    postings_list = reader.lookup("search")
    if postings_list is not None:
        print(postings_list.document_frequency, [p.doc_id for p in postings_list.postings])
```

## Development

```bash
pytest
ruff check .
mypy --strict src/
```

Tests cover the varint primitives (boundary values, truncated buffers), the
bitpacking codec (single and multi-block round trips, per-block bit-width
independence, truncation errors), the postings codec (round trips for empty,
single-posting and very long lists, both the varint and packed paths,
malformed-input rejection), the compression pipeline (deterministic rebuilds,
manifest correctness, verbatim passthrough of `documents.jsonl`/`stats.json`)
and, critically, an end-to-end round trip: for every term in a compressed
index, `CompressedIndexReader.lookup` must reproduce the exact original
postings from the source `postings.jsonl` — the core round-trip guarantee this
repo exists to provide.

**Benchmark**, measured with this project's own CLI on a synthetic
Zipf-distributed reference index (20,000 documents, 30,000 terms, 3,132,833
postings, generated with the default `bitpack_threshold=64` and
`block_size=128`):

```text
$ index-compression-codec compress ./reference-index --output-dir ./reference-compressed
Índice comprimido: 20000 documentos, 30000 términos, 3132833 postings,
192851405 -> 15508032 bytes (ratio 12.44x)

$ index-compression-codec benchmark ./reference-compressed --sample-size 5000
Términos consultados: 5000
Tiempo total: 493.52 ms
Latencia media por lookup: 98.70 µs
Ratio de compresión del índice: 12.44x
```

12.44x compression against the raw JSONL postings, and ~99 µs of average
random-access decode latency per term, on ordinary hardware with no
warm-caching tricks — a single open file handle, one `seek`, one `read`, one
decode per lookup.

## Troubleshooting

- **`ValueError: ... no están ordenados estrictamente ascendentes`** during
  `compress`: the source `postings.jsonl` has a term whose postings aren't
  sorted by strictly ascending `doc_id`, or a posting whose positions aren't
  strictly ascending. This codec trusts the ordering contract documented in
  `inverted-index-builder`'s README and fails loudly rather than silently
  delta-encoding negative or zero gaps — regenerate the source index with a
  compliant version of that tool.
- **`ValueError: inconsistencia de índice ...` during `lookup`**: the
  `document_frequency` recorded in `terms.jsonl` doesn't match the
  `num_postings` encoded inside the corresponding `postings.bin` blob — the
  compressed index directory is corrupted or was assembled from mismatched
  files (e.g. `terms.jsonl` from one compression run paired with
  `postings.bin` from another). Recompress from the source index.
- **`ValueError: formato de índice comprimido no soportado`**: `manifest.json`
  declares a `format_version` this version of the library doesn't know how to
  decode. Recompress with a matching version of `index-compression-codec`.
- **Compression ratio lower than expected on a small or synthetic corpus**:
  the fixed per-blob header (mode byte + `num_postings` varint) and the
  per-block bitpacking header (one byte per 128 values) are a larger fraction
  of very short postings lists — the ratio improves as the corpus grows and
  more terms cross the `bitpack_threshold`. This is expected and is the same
  trade-off real systems like Lucene make with block-based codecs.
- **`mypy` fails under `tests/`**: expected — only `src/` is type-checked in
  `--strict` mode; test fixtures use deliberately looser typing.

## License

MIT — see [`LICENSE`](./LICENSE).
