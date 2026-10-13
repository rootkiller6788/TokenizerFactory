"""分词算法注册表。

本项目手写三个算法，**刻意不引入黑盒 tokenizer**。理由不是「不想装依赖」，
而是这个项目要研究的问题本身就要求打开黑盒：

    同一份语料，什么分词策略最合适？

黑盒只能给出「v1024 的 unigram 比 bpe 少 3% token」这种结论，
给不出**为什么** —— 是合并顺序不同？是剪枝把某个词缀剪掉了？
还是续接前缀让词表被劈成两半？而「为什么」恰好是本项目要交付的东西
（README 的验收标准第二条：能用 Tokenizer Diff **解释差异来源**）。

``pyproject.toml`` 里留了 ``hf`` 可选依赖，用途只有一个：
拿 HuggingFace ``tokenizers`` 训练同一个配置，对照**一致性** ——
手写实现如果和工业实现在同一份语料上给出差得很远的结果，
那说明手写的那份有 bug，而不是「工业实现更聪明」。
"""

from __future__ import annotations

from typing import Any

from ..schema import TokenizerSpec
from .base import Tokenizer, vocab_digest
from .bpe import BPETokenizer
from .unigram import UnigramTokenizer
from .wordpiece import WordPieceTokenizer

#: 算法名 → 实现。配置里 ``specs.algorithms`` 的取值必须在这里。
TOKENIZERS: dict[str, type[Tokenizer]] = {
    "bpe": BPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
}

#: 排序后的算法名元组（用于报错信息与稳定遍历）
ALGORITHMS: tuple[str, ...] = tuple(sorted(TOKENIZERS))

__all__ = [
    "ALGORITHMS",
    "TOKENIZERS",
    "BPETokenizer",
    "Tokenizer",
    "UnigramTokenizer",
    "WordPieceTokenizer",
    "load_tokenizer",
    "minimum_vocab_size",
    "train_tokenizer",
    "vocab_digest",
]


def tokenizer_class(name: str) -> type[Tokenizer]:
    if name not in TOKENIZERS:
        raise ValueError(
            f"未知算法 {name!r}；可用：{', '.join(ALGORITHMS)}"
        )
    return TOKENIZERS[name]


def minimum_vocab_size(algorithms: list[str], alphabet: set[str]) -> int:
    """给定基础字符集，这批算法**至少要**多大的词表才装得下。

    取所有算法里的最大值：配置是全局的，最小的那个 ``vocab_size``
    必须对每个算法都够用，否则会有算法在训练到一半才失败。

    差异主要来自 WordPiece —— 它需要每个字符的 ``##`` 续接形式，
    基础词表是别人的两倍。这个数字必须在**配置加载期**算出来，
    因为「训练一个 256 词表的 WordPiece」会在跑了几十秒之后才炸，
    而那时前两个算法已经白跑完了。
    """
    sizes = [len(tokenizer_class(a).base_vocab(alphabet)) for a in algorithms]
    return max(sizes) if sizes else 0


def train_tokenizer(spec: TokenizerSpec, counts: dict[str, int]) -> Tokenizer:
    """按 spec 训练一个分词器。``counts`` 必须是**同一份**预分词后的词频。"""
    if not counts:
        raise ValueError("词频为空，无法训练分词器")
    return tokenizer_class(spec.algorithm).train(spec, counts)


def load_tokenizer(payload: dict[str, Any]) -> Tokenizer:
    """从 ``export_dict()`` 的产物重新载入一个**可用**的分词器。

    有测试守着「导出 → 载入 → 编码结果完全一致」这条往返等式。
    没有它，导出的 JSON 就只是个快照，而不是一个分词器 ——
    而 README 承诺的产物是「可直接使用的 Tokenizer Artifact」。
    """
    cls = tokenizer_class(str(payload["algorithm"]))
    return cls.from_export(payload)
