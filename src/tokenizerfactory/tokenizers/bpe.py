"""BPE（Byte-Pair Encoding）—— 手写实现，不是黑盒调用。

算法本体只有一句：**反复把语料里最高频的相邻符号对合并成一个新符号**，
每合并一次就往词表里加一个 token，直到用满预算或没有 pair 可合并。

    the  →  t h e
    th   →  th e        （合并 t+h）
    the  →  the         （合并 th+e）

三处必须说清楚的地方
--------------------

1. **计数按词频加权**。``the`` 出现 100 次，那它贡献的 pair 计数是 100 而不是 1。
   不加权会让「只在一次罕见长词里出现的 pair」压过「每个句子都出现的 pair」，
   词表会学出一堆没用的长串。

2. **合并结果已存在时必须跳过**。``a+b`` 与 ``a'+b'`` 可能拼出同一个字符串
   （``ab``+``c`` 与 ``a``+``bc`` 都是 ``abc``）。不跳过就会在词表里放两个
   ``abc``：``token_to_id`` 指向后一个、``len(vocab)`` 虚增、
   ``vocab_digest`` 也不再唯一标识这个分词器 —— 而且全程不报错。

3. **编码时按「合并的先后顺序」而不是「频次」施加合并**。这是 BPE 的
   定义性约定：先学到的合并优先级更高。按频次编码会得到一个不同的、
   与训练不一致的分词器，压缩率看起来还行，但和词表对不上。

不做的事：不处理字节级回退（GPT-2 的 ``byte-level BPE``）。本项目用字符级
BPE + 字符级未知回退，因为要在**同一套未知处理**下比较三个算法；
换成字节级 BPE，未登录字符会自动变成 2~3 个字节 token，未知率恒为 0，
「Unknown」那一维就没法比了。
"""

from __future__ import annotations

from typing import ClassVar

from ..schema import UNK_TOKEN, TokenizerSpec
from .base import Tokenizer
from .merge import train_merges


def bpe_merge(a: str, b: str) -> str:
    return a + b


def bpe_score(pair: tuple[str, str], count: float, sym_freq) -> float:  # noqa: ANN001
    """BPE 只按 pair 频次排序。``sym_freq`` 用不到，但签名要与 WordPiece 一致。"""
    return float(count)


class BPETokenizer(Tokenizer):
    algorithm: ClassVar[str] = "bpe"

    @classmethod
    def base_vocab(cls, alphabet: set[str]) -> list[str]:
        return [UNK_TOKEN] + sorted(alphabet)

    @classmethod
    def train(cls, spec: TokenizerSpec, counts: dict[str, int]) -> "BPETokenizer":
        alphabet: set[str] = set()
        for w in counts:
            alphabet |= set(w)
        base = cls.base_vocab(alphabet)
        target = int(spec.vocab_size)
        if len(base) > target:
            raise ValueError(
                f"词表预算 {target} 装不下基础字符集（{len(base)} 个："
                f"{UNK_TOKEN} + {len(alphabet)} 个字符）；请调大 vocab_size"
            )

        words = sorted(counts)
        freqs = [counts[w] for w in words]
        budget = target - len(base)
        merges = train_merges(
            words,
            freqs,
            budget,
            make_seq=list,
            merge_fn=bpe_merge,
            score_fn=bpe_score,
            existing=set(base),
            use_heap=True,
        )
        vocab = base + [a + b for a, b in merges]
        return cls(
            spec,
            vocab,
            merges=merges,
            meta={
                "n_merges": len(merges),
                "merge_budget": budget,
                "n_alphabet": len(alphabet),
                # 「没合并到预算」是一个有信息量的结果：说明这个语料的
                # 可合并 pair 已经用尽，再大的词表也学不出新东西了。
                "stop_reason": "budget" if len(merges) == budget else "no_more_pairs",
            },
        )

    def _encode_run(self, run: str) -> list[str]:
        """按合并的先后顺序，反复合并「优先级最高」的相邻 pair。"""
        seq = list(run)
        while len(seq) >= 2:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(seq) - 1):
                rank = self._merge_rank.get((seq[i], seq[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i < 0:
                break
            merged = seq[best_i] + seq[best_i + 1]
            seq = seq[:best_i] + [merged] + seq[best_i + 2 :]
        return seq
