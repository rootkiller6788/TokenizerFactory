"""TokenizerFactory —— 同一语料，什么 Tokenization Strategy 最合适？

形状是 **扇形展开再收拢**：

    Corpus → Normalize → Train BPE/Unigram/WordPiece → Encode Benchmark
        → Measure → Compare → Search Vocab/Algorithm → Select → Export

三个分词算法**全部手写**（零第三方依赖）。这不是「不想装依赖」，
而是本项目要研究的问题本身要求打开黑盒：黑盒只能给出
「v1024 的 unigram 比 bpe 少 3% token」，给不出**为什么** ——
是合并顺序不同、是剪枝剪掉了某个词缀、还是续接前缀把词表劈成两半。
而「为什么」正是本项目的验收标准之一（用 Tokenizer Diff 解释差异来源）。

快速开始::

    python main.py run --config configs/default.json --out out
    python main.py compare
    python main.py frontier
    python main.py diff
    python main.py explain bpe-v1024
    python main.py encode bpe-v1024 "中华人民共和国"
"""

from __future__ import annotations

from .compare import (
    assemble,
    dominated_count,
    dominators,
    frontier_key,
    frontier_pairs,
    hypervolume_of,
    knee_point,
    non_dominated,
    reference_point,
    select,
    to_min_form,
)
from .config import DEFAULTS, load_config, load_text_file, resolve_path, validate
from .corpus import (
    PRE_TOKENIZERS,
    SPACE_MARKER,
    alphabet_of,
    char_class,
    corpus_stats,
    detokenize_words,
    pre_tokenize,
    word_counts,
)
from .metrics import (
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    measure,
    measure_speed,
    metric_line,
    point_of,
)
from .normalize import FORMS, describe as describe_normalization, normalize, normalize_words
from .pipeline import CorpusBundle, expand_specs, run_pipeline, write_outputs
from .schema import (
    CODE_VERSION,
    UNK_TOKEN,
    CorpusStats,
    Tokenization,
    TokenizerMetrics,
    TokenizerResult,
    TokenizerRun,
    TokenizerSpec,
    Version,
    sha256_text,
    stable_id,
    utcnow,
)
from .tokenizers import (
    ALGORITHMS,
    TOKENIZERS,
    BPETokenizer,
    Tokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
    load_tokenizer,
    minimum_vocab_size,
    train_tokenizer,
    vocab_digest,
)

__version__ = "0.1.0"

__all__ = [
    "ALGORITHMS",
    "CODE_VERSION",
    "CorpusBundle",
    "CorpusStats",
    "DEFAULTS",
    "FORMS",
    "METRIC_DIRECTIONS",
    "METRIC_LABELS",
    "PRE_TOKENIZERS",
    "SPACE_MARKER",
    "TOKENIZERS",
    "UNK_TOKEN",
    "BPETokenizer",
    "Tokenizer",
    "Tokenization",
    "TokenizerMetrics",
    "TokenizerResult",
    "TokenizerRun",
    "TokenizerSpec",
    "UnigramTokenizer",
    "Version",
    "WordPieceTokenizer",
    "__version__",
    "alphabet_of",
    "assemble",
    "char_class",
    "corpus_stats",
    "describe_normalization",
    "detokenize_words",
    "dominated_count",
    "dominators",
    "expand_specs",
    "frontier_key",
    "frontier_pairs",
    "hypervolume_of",
    "knee_point",
    "load_config",
    "load_text_file",
    "load_tokenizer",
    "measure",
    "measure_speed",
    "metric_line",
    "minimum_vocab_size",
    "non_dominated",
    "normalize",
    "normalize_words",
    "point_of",
    "pre_tokenize",
    "reference_point",
    "resolve_path",
    "run_pipeline",
    "select",
    "sha256_text",
    "stable_id",
    "to_min_form",
    "train_tokenizer",
    "utcnow",
    "validate",
    "vocab_digest",
    "word_counts",
    "write_outputs",
]
