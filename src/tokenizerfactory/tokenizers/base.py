"""分词器基类：三个算法的公共骨架。

三个算法（BPE / WordPiece / Unigram）在**训练**上完全不同，但在**编码**上
共享同一套外壳：

    encode(words) → 逐词 encode_word → Tokenization(tokens, ids, n_unknown)

外壳里有四件事是**必须共享**的，否则算法之间的对比就不公平：

1. **未知字符的处理粒度**。真实 BERT 的 WordPiece 遇到未登录字符时会把
   **整个词**退化成 ``[UNK]``；本项目刻意改成**字符级回退** —— 未登录字符
   各自一个 UNK，同一个词里已知的部分照常切分。理由很实际：整词回退会让
   ``tokens_per_char`` 因为**信息丢失**而虚高（一个词塌成一个 token，
   压缩率当然好看）。那样「谁压得更狠」这条主轴上就会混进「谁丢得更多」，
   而五个维度里的 Unknown 那一维也就失去意义了 —— 它本来就该负责这件事。
   这个改动是刻意的，报告里会单列。

2. **基础字符集**。三个算法的初始词表都必须包含语料里出现过的每个字符
   （否则它连一个词都编不出来），因此「词表规模预算」里要先扣掉这部分。
   扣多少**因算法而异**：WordPiece 还需要每个字符的续接形式 ``##c``，
   因为它的非词首子词一律带这个前缀。这不是额外发明 —— 真实 WordPiece
   训练时的初始符号序列就是 ``[c0, ##c1, ##c2, ...]``，初始词表自然包含
   两套形式。``minimum_vocab_size`` 是这条约束的可执行形式，
   配置加载期会用它挡掉「训练一个 256 词表的 WordPiece」这类配置。

3. **词表顺序**。``[UNK]`` 恒为 id 0，然后是**排序过**的基础字符，最后是
   按学习顺序追加的合并 token。排序这一步不能省：集合遍历顺序在 Python 里
   是确定的但依赖插入顺序，换个语料读取方式就会变，于是 ``vocab_digest``
   会无缘无故地变 —— 而版本号是由它决定的。

4. **未知 token 单独计数**。不从 tokens 里数 ``[UNK]`` 的个数：
   语料里可能真的出现过 ``[UNK]`` 这个字符串并被学进词表，那样数出来的
   是「未知 + 恰好长这样的已知 token」。所以 ``encode_word`` 直接返回
   ``(pieces, n_unknown)``，而不是让调用方事后猜。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence

from ..corpus import SPACE_MARKER
from ..schema import CONTINUATION_PREFIX, UNK_TOKEN, Tokenization, TokenizerSpec


def vocab_digest(vocab: Sequence[str]) -> str:
    """词表摘要。

    用 ``\\n`` 连接而不是 ``""``：否则 ``["ab","c"]`` 与 ``["a","bc"]``
    会撞成同一个摘要 —— 而它们是完全不同的分词器。
    """
    return hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()


class Tokenizer(ABC):
    """训练好的分词器。子类只需实现 ``base_vocab`` / ``train`` / ``_encode_run``。"""

    #: 算法名（与配置里 ``specs.algorithms`` 的取值一致）
    algorithm: ClassVar[str] = ""

    #: 导出格式标识。改了字段含义就要改它，否则旧产物会被当成新格式读
    EXPORT_FORMAT: ClassVar[str] = "tokenizerfactory/artifact/1"

    def __init__(
        self,
        spec: TokenizerSpec,
        vocab: Sequence[str],
        *,
        merges: Sequence[Sequence[str]] | None = None,
        scores: dict[str, float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.vocab: list[str] = list(vocab)

        if len(set(self.vocab)) != len(self.vocab):
            dupes = sorted({t for t in self.vocab if self.vocab.count(t) > 1})
            raise ValueError(
                f"词表出现重复 token：{dupes[:5]} —— 重复会让 id 映射指向前一个，"
                "且 vocab_digest 不再唯一标识这个分词器"
            )
        if UNK_TOKEN not in self.vocab:
            raise ValueError(f"词表必须包含 {UNK_TOKEN}")

        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.unk_id: int = self.token_to_id[UNK_TOKEN]
        #: 单字符 token 集合 —— 判断「这个字符训练时见过吗」的唯一依据
        self.chars: frozenset[str] = frozenset(t for t in self.vocab if len(t) == 1)
        self.merges: list[tuple[str, str]] = [tuple(m) for m in (merges or [])]
        self._merge_rank: dict[tuple[str, str], int] = {
            m: i for i, m in enumerate(self.merges)
        }
        self.scores: dict[str, float] = dict(scores or {})
        self.meta: dict[str, Any] = dict(meta or {})

    # ------------------------------------------------------------------
    # 子类契约
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def base_vocab(cls, alphabet: set[str]) -> list[str]:
        """初始词表（在学任何合并之前）。必须含 ``[UNK]`` 与全部基础字符。"""

    @classmethod
    @abstractmethod
    def train(cls, spec: TokenizerSpec, counts: dict[str, int]) -> "Tokenizer":
        """在词频上训练。``counts`` 是**已经预分词 + 可选小写化**后的词频。"""

    @abstractmethod
    def _encode_run(self, run: str) -> list[str]:
        """把一个「每个字符都已知」的片段切成 token。

        **不得返回** ``[UNK]`` —— 片段里没有未知字符。这条契约有测试守着，
        因为违反它只会表现为未知率悄悄偏高，不会报错。
        """

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------
    def _split_known_runs(self, word: str) -> list[tuple[str, bool]]:
        """把词切成 ``(片段, 是否已知)`` 的交替序列；未知片段长度恒为 1。"""
        runs: list[tuple[str, bool]] = []
        buf: list[str] = []
        for ch in word:
            if ch in self.chars:
                buf.append(ch)
                continue
            if buf:
                runs.append(("".join(buf), True))
                buf = []
            runs.append((ch, False))
        if buf:
            runs.append(("".join(buf), True))
        return runs

    def encode_word(self, word: str) -> tuple[list[str], int]:
        """编码单个词，返回 ``(pieces, n_unknown)``。

        ``n_unknown`` 由这里给出而不是让调用方数 ``[UNK]``，原因见模块文档第 4 条。
        """
        if self.spec.lowercase:
            # 小写化在这里而不是预分词之前：它只该影响词表内容，
            # 不该影响「词是怎么切开的」（见 normalize.normalize_words）
            word = word.lower()
        if not word:
            return [], 0

        pieces: list[str] = []
        n_unknown = 0
        for run, known in self._split_known_runs(word):
            if not known:
                pieces.append(UNK_TOKEN)
                n_unknown += 1
                continue
            sub = self._encode_run(run)
            for t in sub:
                if t == UNK_TOKEN:
                    # 契约要求不该发生；真发生了就如实计数，而不是假装没丢信息
                    n_unknown += 1
            pieces.extend(sub)
        return pieces, n_unknown

    def pieces_of(self, word: str) -> list[str]:
        """只要切分结果（``diff`` / 报告展示用）。"""
        return self.encode_word(word)[0]

    def encode(self, words: Sequence[str]) -> Tokenization:
        """编码一串词。返回的 ``ids`` 与 ``tokens`` 等长且一一对应。"""
        tokens: list[str] = []
        n_unknown = 0
        for w in words:
            pieces, k = self.encode_word(w)
            tokens.extend(pieces)
            n_unknown += k
        ids = [self.token_to_id[t] for t in tokens]
        return Tokenization(tokens=tokens, ids=ids, n_unknown=n_unknown)

    def plain_piece(self, token: str) -> str:
        """token 在原文里对应的**字面**内容。WordPiece 要剥掉 ``##``。

        公开而不是私有：``diff`` 模块要用它算 token 的字符边界，
        而边界必须跨算法可比 —— 带 ``##`` 的 token 长度会多 2，
        直接拿 ``len(token)`` 去算会让 WordPiece 的边界全部错位。
        """
        return token

    def decode_tokens(self, tokens: Sequence[str]) -> str:
        """token 序列 → 文本。

        ``[UNK]`` 一律丢弃：它代表「这里的信息已经丢了」，把字面量 ``[UNK]``
        拼回去只会制造一个假的「还原成功」。代价是：若语料里真的出现过
        ``[UNK]`` 字符串并被学进词表，那个 token 在 decode 时也会被丢掉 ——
        这是本约定的已知代价，比反过来（假装还原成功）可接受。

        结果会比原文多一个前导空格，见 ``corpus`` 模块的「词首标记」一节。
        """
        parts = [self.plain_piece(t) for t in tokens if t != UNK_TOKEN]
        return "".join(parts).replace(SPACE_MARKER, " ")

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_tokens([self.vocab[i] for i in ids])

    # ------------------------------------------------------------------
    # 导出 / 载入
    # ------------------------------------------------------------------
    @property
    def vocab_digest(self) -> str:
        return vocab_digest(self.vocab)

    def export_dict(self) -> dict[str, Any]:
        """可落盘、**可重新载入**的产物。

        只落词表是不够的：BPE / WordPiece 重新编码还需要合并表，
        Unigram 还需要每个 piece 的对数概率。少任何一样，「导出的分词器」
        就只是一个 JSON 快照而不是一个能用的分词器。
        """
        return {
            "format": self.EXPORT_FORMAT,
            "algorithm": self.algorithm,
            "spec": self.spec.to_dict(),
            "unk_token": UNK_TOKEN,
            "space_marker": SPACE_MARKER,
            "continuation_prefix": (
                CONTINUATION_PREFIX if self.algorithm == "wordpiece" else None
            ),
            "vocab": list(self.vocab),
            "merges": [list(m) for m in self.merges] or None,
            "scores": dict(self.scores) or None,
            "meta": dict(sorted(self.meta.items())),
            "vocab_digest": self.vocab_digest,
        }

    @classmethod
    def from_export(cls, payload: dict[str, Any]) -> "Tokenizer":
        spec_dict = dict(payload["spec"])
        spec_dict.pop("spec_id", None)
        spec = TokenizerSpec(**spec_dict)
        return cls(
            spec,
            payload["vocab"],
            merges=payload.get("merges"),
            scores=payload.get("scores"),
            meta=payload.get("meta"),
        )

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.spec.spec_id} "
            f"vocab={len(self.vocab)} merges={len(self.merges)}>"
        )
