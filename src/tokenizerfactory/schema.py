"""核心对象。

本项目的形状是**「直线流水线里嵌一个搜索阶段」**：

    Corpus → Normalize → Train BPE/Unigram/WordPiece → Encode → Measure
        → Compare → Search Vocab/Algorithm → Select → Export

前半段（Normalize）是直线流水线，中段（Train × N）是**并行分支**，
后半段（Compare → Search → Select）是一次**在离散设计空间上的搜索**，
产物是「多算法 × 多词表规模」的对比矩阵 + 一条 Pareto 前沿 + 一份可导出的 Artifact。

因此核心对象分四层：

    CorpusStats      语料画像（字符构成 / 词数 / 唯一词数）—— 「问题是什么」
    TokenizerSpec    一个分词器配置 = 算法 × 词表规模 × 预分词规则 × 归一化选项
    Tokenization     一次编码的结果（token 序列 + id 序列 + 未知计数）
    TokenizerRun     一个 spec 训练 + 评测的完整结果（词表 + 五维度量 + 样本切分）
    TokenizerResult  整条流水线的产物：全部 run + 对比矩阵 + 前沿 + Diff + 版本

三个刻意的设计选择
------------------

1. **``TokenizerSpec.spec_id`` 必须唯一且稳定**（``{algorithm}-v{vocab_size}``）。
   DAG 节点 id、对比矩阵的行名、前沿点的标签全用它 —— 一旦重复，
   报告里就会出现两个看起来一样的点，而读者无法分辨。
2. **词表本身要进 digest**，不能只对 spec 取摘要。两个不同算法在同一个
   ``vocab_size`` 下产出不同词表，若 digest 只看 spec，缓存与版本就会混淆。
3. **指标一律以「原始量」存储**（tokens/char、tokens/sec 的绝对值），
   归一化只发生在算 Pareto 的时候。理由与 DataMixtureFactory 一致：
   报告必须给出能对回现实的数字，否则读者无法判断「快 2 倍」是 1e5 还是 1e6。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: 参与 digest 的代码版本。口径变了就改这个值，否则新旧结果会被当成同一批。
CODE_VERSION = "tokenizerfactory/0.1.0"

#: 未知 token 的显示形式
UNK_TOKEN = "[UNK]"

#: 续接符号前缀（WordPiece 约定：非词首的子词带这个前缀）
CONTINUATION_PREFIX = "##"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, n: int = 12) -> str:
    """稳定短 id。

    **不用内置 ``hash()``** —— 它按 ``PYTHONHASHSEED`` 加盐，跨进程不稳定。
    """
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:n]


# ----------------------------------------------------------------------
# 语料画像
# ----------------------------------------------------------------------
@dataclass
class CorpusStats:
    """语料画像。它决定「该关注哪个评测维度」。

    例如 CJK 占比高的语料，预分词规则的选择对 fertility 影响巨大
    （中文没有空格，若不做单字切分，一整句会变成一个「词」）；
    而符号占比高的语料，未知率才是关键维度。报告第一章会用它来解释
    「为什么这次对比里某个维度特别重要」。
    """

    name: str
    n_chars: int
    n_words: int
    n_unique_words: int
    #: 字符类别占比：cjk / latin / digit / space / symbol / emoji / other
    script_shares: dict[str, float] = field(default_factory=dict)
    max_word_len: int = 0
    mean_word_len: float = 0.0

    def dominant_script(self) -> str:
        if not self.script_shares:
            return "other"
        return max(sorted(self.script_shares), key=lambda k: self.script_shares[k])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_chars": self.n_chars,
            "n_words": self.n_words,
            "n_unique_words": self.n_unique_words,
            "script_shares": {k: round(v, 6) for k, v in sorted(self.script_shares.items())},
            "max_word_len": self.max_word_len,
            "mean_word_len": round(self.mean_word_len, 4),
        }


# ----------------------------------------------------------------------
# 分词器配置
# ----------------------------------------------------------------------
@dataclass
class TokenizerSpec:
    """一个分词器配置。

    ``vocab_size`` 是**目标**规模：BPE / WordPiece 是「合并到这么多」，
    Unigram 是「剪枝到这么多」。三者都保证**实际词表不超过目标**，
    否则「同等词表预算下谁压得更狠」这个对比就不成立。
    """

    algorithm: str
    vocab_size: int
    #: 预分词规则名（见 corpus.PRE_TOKENIZERS）
    pre_tokenizer: str = "unicode"
    #: 是否先小写化（英文场景有用，但会让「Apple」与「apple」共享 token）
    lowercase: bool = False
    #: 算法特有参数（如 Unigram 的 seed_max_len / prune_ratio）
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        """稳定且唯一的短标识。DAG 节点 id / 矩阵行名 / 前沿标签都用它。"""
        suffix = ""
        if self.lowercase:
            suffix += "+lower"
        if self.pre_tokenizer != "unicode":
            suffix += f"+{self.pre_tokenizer}"
        return f"{self.algorithm}-v{self.vocab_size}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "pre_tokenizer": self.pre_tokenizer,
            "lowercase": self.lowercase,
            "extra": dict(sorted(self.extra.items())),
        }

    @property
    def digest(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))


# ----------------------------------------------------------------------
# 编码结果
# ----------------------------------------------------------------------
@dataclass
class Tokenization:
    """一次编码的结果。

    ``n_unknown`` 单独记而不是从 tokens 里数 ``[UNK]``：因为词表里
    可能**合法地**含有 ``[UNK]`` 这个字符串（如果语料里真的出现过），
    数 tokens 会把它算成未知，而它其实是已知 token。
    """

    tokens: list[str]
    ids: list[int]
    n_unknown: int = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": len(self.tokens),
            "n_unknown": self.n_unknown,
            "tokens": list(self.tokens),
            "ids": list(self.ids),
        }


# ----------------------------------------------------------------------
# 五维度量
# ----------------------------------------------------------------------
@dataclass
class TokenizerMetrics:
    """README「评测维度」那一节的五个维度，逐条落成字段。

    | 维度 | 字段 | 方向 |
    | --- | --- | --- |
    | Compression | ``tokens_per_char`` | 越小越省上下文 |
    | Fertility | ``tokens_per_word`` | 越小越省上下文（跨语言比较的关键） |
    | Coverage | ``vocab_coverage`` | 越大越好（benchmark 里被词表覆盖的字符占比） |
    | Speed | ``encode_tokens_per_sec`` / ``decode_tokens_per_sec`` | 越大越好 |
    | Unknown | ``unknown_rate`` / ``hard_unknown_rate`` | 越小越好 |
    """

    #: 压缩：每个字符平均产生多少 token（README 的 Compression）
    tokens_per_char: float
    #: 生育率：每个词平均切出多少 token（README 的 Fertility）
    tokens_per_word: float
    #: 词表覆盖率：benchmark 里被词表覆盖的字符占比
    vocab_coverage: float
    #: 未知率：benchmark 里落到 [UNK] 的 token 占比
    unknown_rate: float
    #: 未知率（hard 语料：emoji / 罕见 CJK / 代码符号）
    hard_unknown_rate: float
    #: 编码吞吐（token/秒）
    encode_tokens_per_sec: float
    #: 解码吞吐（token/秒）
    decode_tokens_per_sec: float
    #: 原始计数，报告里要能对回现实
    n_tokens: int
    n_chars: int
    n_words: int

    def get(self, name: str) -> float:
        if not hasattr(self, name):
            raise KeyError(f"未知指标 {name!r}")
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_per_char": round(self.tokens_per_char, 6),
            "tokens_per_word": round(self.tokens_per_word, 6),
            "vocab_coverage": round(self.vocab_coverage, 6),
            "unknown_rate": round(self.unknown_rate, 6),
            "hard_unknown_rate": round(self.hard_unknown_rate, 6),
            "encode_tokens_per_sec": round(self.encode_tokens_per_sec, 2),
            "decode_tokens_per_sec": round(self.decode_tokens_per_sec, 2),
            "n_tokens": self.n_tokens,
            "n_chars": self.n_chars,
            "n_words": self.n_words,
        }


# ----------------------------------------------------------------------
# 一次训练 + 评测
# ----------------------------------------------------------------------
@dataclass
class TokenizerRun:
    """一个 spec 的完整结果。

    ``sample`` 保存 benchmark 前若干个词的切分结果 —— 它是
    ``explain`` 与 ``diff`` 的数据来源。**必须在训练时就存下来**，
    因为训练完的 tokenizer 对象不落盘（只落词表），事后重建就要重训。
    """

    spec: TokenizerSpec
    vocab: list[str]
    vocab_digest: str
    metrics: TokenizerMetrics
    train_seconds: float
    #: benchmark 前 N 个词的切分，形如 [("tokenization", [token, ...]), ...]
    sample: list[tuple[str, list[str]]] = field(default_factory=list)
    #: 算法特有的诊断信息（BPE 的合并次数 / Unigram 的剪枝轮数 …）
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    @property
    def algorithm(self) -> str:
        return self.spec.algorithm

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "spec_id": self.spec_id,
            "algorithm": self.algorithm,
            "vocab_size_actual": len(self.vocab),
            "vocab_digest": self.vocab_digest,
            "metrics": self.metrics.to_dict(),
            "train_seconds": round(self.train_seconds, 4),
            "meta": dict(sorted(self.meta.items())),
        }


# ----------------------------------------------------------------------
# 版本
# ----------------------------------------------------------------------
@dataclass
class Version:
    """内容寻址版本：digest 由**入选词表**决定。

    注意口径：这里「内容」是最终选中的分词器词表，而不是整条流水线的全部产物。
    只有词表变了版本才该变；换了算法但词表恰好相同，产物其实是等价的。
    """

    version_id: str
    created_at: str
    digest: str
    tag: str
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "digest": self.digest,
            "tag": self.tag,
            "counts": dict(sorted(self.counts.items())),
        }


# ----------------------------------------------------------------------
# 流水线产物
# ----------------------------------------------------------------------
@dataclass
class TokenizerResult:
    """整条流水线的产物。"""

    corpus: CorpusStats
    runs: list[TokenizerRun] = field(default_factory=list)
    #: 对比矩阵的行（每个 spec 一行，含五维度量），由 compare 模块生成
    matrix: list[dict[str, Any]] = field(default_factory=list)
    #: Pareto 前沿上的 spec_id（按目标对分组的字典）
    frontiers: dict[str, list[str]] = field(default_factory=dict)
    #: Tokenizer Diff 的量化结果
    diffs: dict[str, Any] = field(default_factory=dict)
    #: 最终入选的 spec_id（Select 阶段的产物）
    selected: str = ""
    dag: Any = None
    version: Version | None = None
    counters: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    # -- 便捷查询 ----------------------------------------------------
    def find(self, spec_id: str) -> TokenizerRun | None:
        for r in self.runs:
            if r.spec_id == spec_id:
                return r
        return None

    def by_algorithm(self, algorithm: str) -> list[TokenizerRun]:
        return [r for r in self.runs if r.algorithm == algorithm]

    def algorithms(self) -> list[str]:
        return sorted({r.algorithm for r in self.runs})

    def vocab_sizes(self) -> list[int]:
        return sorted({r.spec.vocab_size for r in self.runs})

    def frontier_of(self, key: str) -> list[TokenizerRun]:
        return [
            r
            for sid in self.frontiers.get(key, [])
            if (r := self.find(sid)) is not None
        ]

    def known_spec_ids(self) -> list[str]:
        return sorted(r.spec_id for r in self.runs)

    def best_by(self, metric: str, direction: str = "min") -> TokenizerRun | None:
        if not self.runs:
            return None
        key = (lambda r: r.metrics.get(metric))
        return (min if direction == "min" else max)(self.runs, key=key)
