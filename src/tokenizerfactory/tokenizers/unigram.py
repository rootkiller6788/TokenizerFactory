"""Unigram 语言模型分词器 —— 手写实现，不是黑盒调用。

与前两个算法的根本差别：BPE / WordPiece 从**字符**出发往上合并，
Unigram 从**一大袋子候选子串**出发往下剪枝。它的世界观是概率的：

    每个 piece 有自己的概率 p；一个词的切分概率是各 piece 概率之积
    取概率最大的那种切分（Viterbi），训练时最大化语料的总似然（EM）

四段流程
--------

1. **种子词表**：语料里长度 ≤ ``seed_max_len``、出现次数 ≥ ``seed_min_count``
   的所有子串。这个集合通常有几千到几万个，远大于目标词表。
2. **EM（前向-后向）**：E 步算每个 piece 的**期望使用次数**，M 步重估概率。
3. **剪枝**：把「损失」最小的 piece 剪掉，再重跑 EM。损失的经典定义是
   ``期望次数 × (长度 - 1)`` —— 直觉是「去掉它之后，那些位置平均要多出
   这么多 token」。单字符**永不剪**，否则会有字符失去全部覆盖路径。
4. **编码**：Viterbi 求最大概率切分。

三个必须说清楚的实现选择
------------------------

1. **E 步用真实边际，不是 Viterbi 近似**。常见简化做法是「E 步只用最优切分
   的计数」，跑得快但会把全部质量压到一条路径上，长词的概率被系统性高估。
   这里用前向-后向把**所有切分路径**的概率加权求和 —— 代码只多二十行，
   但它是 Unigram 这个词的本意（uni-gram 边缘化）。

2. **剪枝的并列打破必须用 rng**。piece 的损失大量并列（次数 1、长度 2 的
   子串损失全是 1），若按字典序决定谁先被剪，就会**系统性地**先剪掉
   「字典序靠后」的那些，词表被一个与语言无关的因素扭曲。用 rng 随机打破
   才没有偏向 —— 这就是本项目引入 ``rng`` 模块的第一个理由。

3. **概率取对数存**。语料里一个 20 字的词，路径概率是几十个小数相乘，
   直接连乘会下溢成 0，于是「所有切分概率都是 0」而**不报错**。
   全部在 log 空间做加/减，前向用 ``logsumexp``、Viterbi 用 max。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import ClassVar, Iterable

from ..rng import make_rng
from ..schema import UNK_TOKEN, TokenizerSpec
from .base import Tokenizer

NEG_INF = float("-inf")

#: 概率下限。剪枝后可能留下「期望次数为 0」的 piece（单字符被强制保留时就
#: 会出现这种情况），给它一个极小但非零的概率，保证它仍在 lattice 里可达。
#: 用 0 会让 ``log(0) = -inf`` 把整条路径变成不可达。
MIN_PROB = 1e-12

#: 基础字符集之外的 piece 上限倍数（防止种子词表爆炸）
SEED_CAP_FACTOR = 4


# ----------------------------------------------------------------------
# log 空间的基本运算
# ----------------------------------------------------------------------
def logadd(a: float, b: float) -> float:
    """``log(exp(a) + exp(b))``，溢出安全。"""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _build_lattice(word: str, lp: dict[str, float], max_len: int) -> list[list[tuple[int, float]]]:
    """``edges[i]`` = 从位置 i 出发的 ``(结束位置, 该 piece 的对数概率)``。"""
    n = len(word)
    edges: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for length in range(1, min(max_len, n - i) + 1):
            value = lp.get(word[i : i + length])
            if value is not None:
                edges[i].append((i + length, value))
    return edges


# ----------------------------------------------------------------------
# 种子词表
# ----------------------------------------------------------------------
def seed_pieces(
    counts: dict[str, int], max_len: int, min_count: int, cap: int
) -> set[str]:
    """长度 ≤ ``max_len``、频次 ≥ ``min_count`` 的全部子串（含单字符）。

    单字符**无条件保留**，即使频次低于 ``min_count`` —— 否则语料里
    只出现一次的那个罕见字符就没有任何覆盖路径，含它的词会整段变成 UNK。
    """
    tally: Counter = Counter()
    for word, freq in counts.items():
        n = len(word)
        for i in range(n):
            for length in range(1, min(max_len, n - i) + 1):
                tally[word[i : i + length]] += freq

    pieces = {p for p, c in tally.items() if c >= min_count}
    chars = {c for w in counts for c in w}
    pieces |= chars

    if len(pieces) > cap:
        # 按 (频次降序, 字典序) 取前 cap 个。字典序在这里只是为了让结果确定，
        # 不承担语义 —— 真正的筛选依据是频次。
        keep = set(chars)
        for piece in sorted(pieces, key=lambda p: (-tally.get(p, 0), p)):
            if len(keep) >= cap:
                break
            keep.add(piece)
        pieces = keep
    return pieces


# ----------------------------------------------------------------------
# EM
# ----------------------------------------------------------------------
def em_fit(
    pieces: Iterable[str],
    counts: dict[str, int],
    iters: int,
    *,
    tol: float = 1e-7,
) -> tuple[dict[str, float], Counter]:
    """在词频上跑 EM，返回 ``(对数概率表, 期望次数表)``。

    收敛判据用**对数似然的变化**而不是参数变化：参数变化量在几千个 piece 上
    平均之后会小到看不出停滞，而似然是单调的，停没停一目了然。
    """
    piece_list = sorted(set(pieces))
    max_len = max((len(p) for p in piece_list), default=1)
    uniform = -math.log(len(piece_list)) if piece_list else 0.0
    lp: dict[str, float] = {p: uniform for p in piece_list}

    expected: Counter = Counter()
    prev_ll: float | None = None
    for _ in range(max(1, iters)):
        expected = Counter()
        total_ll = 0.0
        for word, freq in counts.items():
            n = len(word)
            if n == 0:
                continue
            edges = _build_lattice(word, lp, max_len)

            # 前向：alpha[i] = log P(前 i 个字符的所有切分)
            alpha = [NEG_INF] * (n + 1)
            alpha[0] = 0.0
            for i in range(n):
                if alpha[i] == NEG_INF:
                    continue
                base = alpha[i]
                for j, value in edges[i]:
                    alpha[j] = logadd(alpha[j], base + value)
            z = alpha[n]
            if z == NEG_INF:
                # 该词无法切分。理论上不可达（单字符都在），真发生了就跳过，
                # 而不是把 -inf 混进后面的除法里变成 nan。
                continue
            total_ll += freq * z

            # 后向：beta[i] = log P(从 i 到结尾的所有切分)
            beta = [NEG_INF] * (n + 1)
            beta[n] = 0.0
            for i in range(n - 1, -1, -1):
                for j, value in edges[i]:
                    if beta[j] == NEG_INF:
                        continue
                    beta[i] = logadd(beta[i], value + beta[j])

            # 期望次数：piece 在 (i, j) 上的后验概率之和
            for i in range(n):
                if alpha[i] == NEG_INF:
                    continue
                for j, value in edges[i]:
                    if beta[j] == NEG_INF:
                        continue
                    expected[word[i:j]] += freq * math.exp(alpha[i] + value + beta[j] - z)

        total = sum(expected.values())
        if total <= 0:
            break
        lp = {
            p: math.log(max(expected.get(p, 0.0) / total, MIN_PROB))
            for p in piece_list
        }
        if prev_ll is not None and abs(total_ll - prev_ll) <= tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = total_ll

    return lp, expected


# ----------------------------------------------------------------------
# 剪枝
# ----------------------------------------------------------------------
def prune_pieces(
    pieces: set[str],
    expected: Counter,
    keep_n: int,
    rng,
) -> set[str]:
    """剪掉 ``len(pieces) - keep_n`` 个损失最小的 piece（单字符除外）。

    损失 ``= 期望次数 × (长度 - 1)``：去掉它之后，原来用它吃掉的那些位置
    平均要多出这么多个 token。这与 SentencePiece 的剪枝判据一致。

    并列项用 ``rng`` 打乱后再截断。这一步不能省成「按字典序」——
    损失并列的量非常大（所有「次数 1、长度 2」的子串损失都是 1），
    按字典序会系统性地先剪掉字典序靠后的那些，等于用一个与语言无关的
    因素扭曲词表。
    """
    removable = [p for p in pieces if len(p) > 1]
    n_remove = len(pieces) - keep_n
    if n_remove <= 0:
        return set(pieces)
    if n_remove >= len(removable):
        # 要剪的比可剪的还多：只能把可剪的全剪掉，单字符保住。
        # 此时最终词表会大于目标 —— 如实返回，由调用方记录。
        return set(pieces) - set(removable)

    loss = {p: expected.get(p, 0.0) * (len(p) - 1) for p in removable}
    groups: dict[float, list[str]] = {}
    for piece in removable:
        groups.setdefault(loss[piece], []).append(piece)
    ordered: list[str] = []
    for key in sorted(groups):
        bucket = groups[key]
        rng.shuffle(bucket)
        ordered.extend(bucket)

    drop = set(ordered[:n_remove])
    return {p for p in pieces if p not in drop}


# ----------------------------------------------------------------------
# 分词器
# ----------------------------------------------------------------------
class UnigramTokenizer(Tokenizer):
    algorithm: ClassVar[str] = "unigram"

    def __init__(self, spec, vocab, *, merges=None, scores=None, meta=None) -> None:
        super().__init__(spec, vocab, merges=merges, scores=scores, meta=meta)
        self._max_piece_len = max((len(p) for p in self.scores), default=1)

    @classmethod
    def base_vocab(cls, alphabet: set[str]) -> list[str]:
        return [UNK_TOKEN] + sorted(alphabet)

    @classmethod
    def train(cls, spec: TokenizerSpec, counts: dict[str, int]) -> "UnigramTokenizer":
        extra = spec.extra or {}
        seed_max_len = int(extra.get("seed_max_len", 6))
        seed_min_count = int(extra.get("seed_min_count", 2))
        prune_rounds = int(extra.get("prune_rounds", 8))
        prune_ratio = float(extra.get("prune_ratio", 0.4))
        em_iters = int(extra.get("em_iters", 8))

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
        # 最终词表 = [UNK] + 基础字符 + 学到的多字 piece
        #          = 1 + |alphabet| + (|pieces| - |alphabet|) = 1 + |pieces|
        piece_target = target - 1

        rng = make_rng(spec.digest, "unigram-prune")
        cap = max(SEED_CAP_FACTOR * target, 4000)
        seeds = seed_pieces(counts, seed_max_len, seed_min_count, cap)
        n_seeds = len(seeds)

        pieces = set(seeds)
        lp, expected = em_fit(pieces, counts, em_iters)
        rounds_done = 0
        for _ in range(max(0, prune_rounds)):
            if len(pieces) <= piece_target:
                break
            keep_n = max(piece_target, int(len(pieces) * (1.0 - prune_ratio)))
            if keep_n >= len(pieces):
                break
            pieces = prune_pieces(pieces, expected, keep_n, rng)
            lp, expected = em_fit(pieces, counts, em_iters)
            rounds_done += 1

        trimmed_directly = 0
        if len(pieces) > piece_target:
            # 计划轮数跑完仍未到目标：直接剪到位，再收敛一次概率，
            # 否则留下来的 piece 的概率是在「还包含已删 piece」的 lattice 上估的
            trimmed_directly = len(pieces) - piece_target
            pieces = prune_pieces(pieces, expected, piece_target, rng)
            lp, _ = em_fit(pieces, counts, em_iters)

        # ``[UNK]`` 是保留 token（id 0），不能再作为一个普通 piece 进词表。
        #
        # 这个洞是测试抓出来的：若语料里**真的出现过**字面量 ``[UNK]``
        # （例如一份讨论分词器的文档），种子词表会把它当成一个普通子串学下来，
        # 于是最终词表里出现两个 ``[UNK]`` —— ``token_to_id`` 指向后一个、
        # ``len(vocab)`` 虚增、``vocab_digest`` 不再唯一标识这个分词器。
        #
        # BPE / WordPiece 不需要这个过滤：它们的合并循环把**整个基础词表**
        # 放进了 ``existing`` 集合，合并出 ``[UNK]`` 的那一步会被直接跳过。
        # Unigram 的剪枝路径没有这道闸，所以必须在这里显式排除。
        reserved = UNK_TOKEN in pieces
        learned = sorted(
            (p for p in pieces if p not in alphabet and p != UNK_TOKEN),
            key=lambda p: (-lp[p], p),
        )
        vocab = base + learned
        return cls(
            spec,
            vocab,
            scores={p: lp[p] for p in pieces if p != UNK_TOKEN},
            meta={
                "n_seeds": n_seeds,
                "seed_cap": cap,
                "n_pieces": len(pieces),
                "piece_target": piece_target,
                "prune_rounds_done": rounds_done,
                "prune_rounds_planned": prune_rounds,
                "prune_ratio": prune_ratio,
                "trimmed_directly": trimmed_directly,
                "em_iters": em_iters,
                "min_prob": MIN_PROB,
                # 语料里出现过字面量 [UNK] 时它会被丢掉 —— 记下来，
                # 否则「词表比 pieces 少一个」这件事无法解释
                "reserved_unk_piece_dropped": reserved,
            },
        )

    def _encode_run(self, run: str) -> list[str]:
        """Viterbi：求总对数概率最大的切分。"""
        n = len(run)
        if n == 0:
            return []
        best = [NEG_INF] * (n + 1)
        best[0] = 0.0
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for length in range(1, min(self._max_piece_len, i) + 1):
                value = self.scores.get(run[i - length : i])
                if value is None:
                    continue
                candidate = best[i - length] + value
                if candidate > best[i]:
                    best[i] = candidate
                    back[i] = i - length
        if best[n] == NEG_INF:
            # 不可达：单字符都在词表里（概率为 MIN_PROB，非 -inf），
            # 所以这里不会发生。真到了就逐字符返回，绝不返回 UNK
            # —— 契约要求已知片段里不得产生未知。
            return list(run)
        out: list[str] = []
        i = n
        while i > 0:
            j = back[i]
            out.append(run[j:i])
            i = j
        out.reverse()
        return out
