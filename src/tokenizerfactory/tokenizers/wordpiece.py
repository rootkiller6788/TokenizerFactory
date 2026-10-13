"""WordPiece —— 手写实现，不是黑盒调用。

WordPiece 与 BPE 的差别**只在选 pair 的依据**上：

    BPE:       分数 = count(pair)
    WordPiece: 分数 = count(pair) / (count(左) × count(右))

直觉是这样更聪明：``(t, h)`` 各出现 1000 次、一起出现 900 次，
比 ``(q, z)`` 各出现 2 次、一起出现 2 次更值得合并 —— 前者是「t 和 h 本来
就很常见，凑巧挨着」，后者是「q 和 z 几乎只为彼此存在」。
按 BPE 的纯频次，前者赢；按 WordPiece 的比值，后者赢。
这也是为什么 WordPiece 在小词表下往往能更快学到有意义的词缀。

续接前缀 ``##``
---------------
WordPiece 用 ``##`` 标记「这不是词的第一个子词」：

    playing  →  play + ##ing

于是 ``##ing`` 只可能出现在词中，``play`` 只可能出现在词首 —— 词表被
**无歧义地**分成两半，同一个字符串不会同时扮演两种角色。

初始词表因此必须包含**两套**单字符形式（``c`` 与 ``##c``）。这不是额外发明：
真实 WordPiece 训练时的初始符号序列就是 ``[c0, ##c1, ##c2, ...]``，
初始词表自然包含两套。代价是基础词表占用翻倍 —— 配置加载期会据此校验
「词表预算装不装得下」，而不是等训练完才发现。

合并时保留**前一个符号**的前缀：``##l`` + ``##l`` → ``##ll``，
``h`` + ``##e`` → ``he``。于是 ``##`` 恰好落在「这个词的第二个字符及以后」，
语义不会被合并破坏。
"""

from __future__ import annotations

from typing import ClassVar

from ..schema import CONTINUATION_PREFIX, UNK_TOKEN, TokenizerSpec
from .base import Tokenizer
from .merge import train_merges

PREFIX = CONTINUATION_PREFIX


def wp_make_seq(word: str) -> list[str]:
    """首字符保持原样，其余字符带 ``##`` 前缀。"""
    if not word:
        return []
    return [word[0]] + [PREFIX + c for c in word[1:]]


def wp_merge(a: str, b: str) -> str:
    """保留 ``a`` 的前缀，拼上 ``b`` 去掉前缀后的本体。"""
    prefix = PREFIX if a.startswith(PREFIX) else ""
    body_a = a[len(prefix) :]
    body_b = b[len(PREFIX) :] if b.startswith(PREFIX) else b
    return prefix + body_a + body_b


def wp_score(pair: tuple[str, str], count: float, sym_freq) -> float:  # noqa: ANN001
    """``count(pair) / (count(左) × count(右))``。

    分母取 ``max(1, ...)`` 而不是直接相乘：符号频次理论上不会为 0
    （能出现在 pair 里就说明至少出现过一次），但这一步是除法的唯一防线，
    换成 0 会让整个训练以 ``ZeroDivisionError`` 收场。
    """
    a, b = pair
    return count / (max(1, sym_freq.get(a, 0)) * max(1, sym_freq.get(b, 0)))


class WordPieceTokenizer(Tokenizer):
    algorithm: ClassVar[str] = "wordpiece"

    @classmethod
    def base_vocab(cls, alphabet: set[str]) -> list[str]:
        # 两套单字符形式：词首形式 + 续接形式。见模块文档。
        return (
            [UNK_TOKEN]
            + sorted(alphabet)
            + sorted(PREFIX + c for c in alphabet)
        )

    @classmethod
    def train(cls, spec: TokenizerSpec, counts: dict[str, int]) -> "WordPieceTokenizer":
        alphabet: set[str] = set()
        for w in counts:
            alphabet |= set(w)
        base = cls.base_vocab(alphabet)
        target = int(spec.vocab_size)
        if len(base) > target:
            raise ValueError(
                f"词表预算 {target} 装不下 WordPiece 的基础词表（{len(base)} 个："
                f"{UNK_TOKEN} + {len(alphabet)} 个字符 + 同样多的 ## 续接形式）；"
                f"请调大 vocab_size 或换更小的语料字符集"
            )

        words = sorted(counts)
        freqs = [counts[w] for w in words]
        budget = target - len(base)
        merges = train_merges(
            words,
            freqs,
            budget,
            make_seq=wp_make_seq,
            merge_fn=wp_merge,
            score_fn=wp_score,
            existing=set(base),
            # 分数依赖符号频次，而符号频次的变化会波及不含刚合并 pair 的词，
            # 惰性堆无法可靠失效 → 走全扫描（pair 表仍是增量维护的）
            use_heap=False,
        )
        vocab = base + [wp_merge(a, b) for a, b in merges]
        return cls(
            spec,
            vocab,
            merges=merges,
            meta={
                "n_merges": len(merges),
                "merge_budget": budget,
                "n_alphabet": len(alphabet),
                "n_continuation_base": len(alphabet),
                "stop_reason": "budget" if len(merges) == budget else "no_more_pairs",
            },
        )

    def plain_piece(self, token: str) -> str:
        return token[len(PREFIX) :] if token.startswith(PREFIX) else token

    def _encode_run(self, run: str) -> list[str]:
        """贪心最长匹配：每个位置优先取词表里最长的那个候选。

        真实 WordPiece 就是这么编的（BERT 的 ``WordpieceTokenizer``），
        所以这里不引入任何近似。
        """
        pieces: list[str] = []
        start = 0
        n = len(run)
        while start < n:
            end = n
            matched: str | None = None
            while end > start:
                candidate = run[start:end]
                if start > 0:
                    candidate = PREFIX + candidate
                if candidate in self.token_to_id:
                    matched = candidate
                    break
                end -= 1
            if matched is None:
                # 契约要求不得返回 UNK。已知片段里首字符有词首形式、
                # 其余字符有 ## 形式（都在基础词表里），所以这里不可达；
                # 真到了就逐字符兜底，至少不丢信息。
                return list(run)
            pieces.append(matched)
            start = end
        return pieces
