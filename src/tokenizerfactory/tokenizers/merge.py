"""BPE 与 WordPiece 共用的增量合并循环。

朴素实现每学一个合并就把整个语料重新数一遍 pair，复杂度是
``O(合并数 × 语料符号数)``。本项目的规模（几千个词、上千次合并）下这要几十秒，
而它会被「3 算法 × 4 词表规模」扫一遍 —— 不可接受。

增量做法：只更新**含这个 pair 的那些词**，其余词的贡献不变。

这类优化最怕的是「跑得通、结果却悄悄不同」，所以本模块同时保留一个
**朴素参照实现** ``train_merges_naive``，测试逐项比对两者的合并序列。
没有这个参照实现，增量版本的任何簿记错误都会表现为「词表略微不同」，
既不报错、也看不出来 —— 而那正好是分词器对比里最致命的一类错误。

两种算法在这里的差异只有三处，用回调注入：

| | 初始符号序列 | 合并结果 | 选 pair 的依据 |
| --- | --- | --- | --- |
| BPE | 逐字符 | ``a + b`` | pair 频次最高 |
| WordPiece | 首字符 + 其余带 ``##`` | 保留前一个符号的前缀 | ``频次 / (freq(a) × freq(b))`` |
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Callable, Sequence

Pair = tuple[str, str]
MergeFn = Callable[[str, str], str]
#: ``(pair, pair 频次, 符号频次表) -> 分数``（越大越先合并）
ScoreFn = Callable[[Pair, float, Counter], float]


def merge_sequence(seq: Sequence[str], pair: Pair, merge_fn: MergeFn) -> list[str]:
    """把序列里**所有**出现该 pair 的位置合并掉（一次扫完，左到右不重叠）。"""
    a, b = pair
    out: list[str] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(merge_fn(a, b))
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _best_pair_scan(
    pair_counts: Counter,
    sym_freq: Counter,
    score_fn: ScoreFn,
    existing: set[str],
    merge_fn: MergeFn,
) -> Pair | None:
    """全扫描选最优 pair；并列时取字典序最小的，保证确定性。

    ``existing`` 判定必须用 ``merge_fn`` 算合并结果，不能写 ``a + b`` ——
    WordPiece 的合并结果是保留前一个符号 ``##`` 前缀的，写成 ``a + b``
    会让「结果已存在」这条检查在 WordPiece 上完全失效。
    """
    best: Pair | None = None
    best_key: tuple[float, Pair] | None = None
    for pair, count in pair_counts.items():
        if merge_fn(*pair) in existing:
            # 合并结果已经存在 —— 再合并只会让词表里出现两个一样的 token。
            # 真实 BPE / WordPiece 实现都会跳过它（否则 token_to_id 会指向后一个，
            # 词表长度虚增，且 vocab_digest 不再唯一标识这个分词器）。
            continue
        key = (-score_fn(pair, count, sym_freq), pair)
        if best_key is None or key < best_key:
            best_key, best = key, pair
    return best


class MergeTrainer:
    """增量维护 pair 频次 / 符号频次 / pair→词 索引，逐步产出合并表。"""

    def __init__(
        self,
        words: Sequence[str],
        freqs: Sequence[int],
        *,
        make_seq: Callable[[str], list[str]],
        merge_fn: MergeFn,
        score_fn: ScoreFn,
        existing: set[str],
        use_heap: bool,
    ) -> None:
        self.freqs = list(freqs)
        self.seqs = [make_seq(w) for w in words]
        self.merge_fn = merge_fn
        self.score_fn = score_fn
        self.use_heap = use_heap
        #: 已经「存在」的符号串。合并结果若已在其中就必须跳过。
        #: 初始值不只是初始符号 —— 还包括**整个基础词表**：语料里若真的出现过
        #: ``[UNK]`` 这个字符串，BPE 完全可能把 ``[``+``U`` 一路合并成 ``[UNK]``，
        #: 而它已经在词表里了。
        self.existing: set[str] = set(existing)
        for seq in self.seqs:
            self.existing.update(seq)

        self.pair_counts: Counter = Counter()
        self.sym_freq: Counter = Counter()
        self.pair_words: dict[Pair, set[int]] = defaultdict(set)
        for i, seq in enumerate(self.seqs):
            f = self.freqs[i]
            for x in seq:
                self.sym_freq[x] += f
            for pair in zip(seq, seq[1:]):
                self.pair_counts[pair] += f
            for pair in set(zip(seq, seq[1:])):
                self.pair_words[pair].add(i)

        # BPE 的分数只取决于 pair 频次，所以「计数变了」等价于「分数过期了」，
        # 惰性堆可以精确失效。WordPiece 的分数还取决于符号频次，而符号频次的
        # 变化会波及**不含有刚合并那个 pair 的词**里的 pair（例如符号 a 变短了，
        # 词 j 里的 (a, z) 分数也跟着变，但 j 不在受影响词集合里），
        # 惰性堆无法可靠失效 —— 于是它走全扫描（pair 表本身仍是增量维护的）。
        self.heap: list[tuple[float, Pair]] = []
        if use_heap:
            self.heap = [(-score_fn(p, c, self.sym_freq), p) for p, c in self.pair_counts.items()]
            heapq.heapify(self.heap)

    # ------------------------------------------------------------------
    def _merge_result(self, pair: Pair) -> str:
        return self.merge_fn(*pair)

    def _pop_best(self) -> Pair | None:
        if not self.use_heap:
            return _best_pair_scan(
                self.pair_counts, self.sym_freq, self.score_fn, self.existing, self.merge_fn
            )
        while self.heap:
            neg, pair = heapq.heappop(self.heap)
            count = self.pair_counts.get(pair)
            if count is None or -neg != count:
                continue  # 过期条目（计数后来变过，后面会推入新条目）
            if self._merge_result(pair) in self.existing:
                continue  # 结果已存在 → 永久失效（existing 只增不减）
            return pair
        return None

    def _apply_to_word(self, i: int, pair: Pair) -> None:
        seq = self.seqs[i]
        f = self.freqs[i]
        old_pairs = Counter(zip(seq, seq[1:]))
        old_syms = Counter(seq)
        new_seq = merge_sequence(seq, pair, self.merge_fn)
        if new_seq == seq:
            return
        new_pairs = Counter(zip(new_seq, new_seq[1:]))
        new_syms = Counter(new_seq)

        for x in set(old_syms) | set(new_syms):
            delta = (new_syms.get(x, 0) - old_syms.get(x, 0)) * f
            if not delta:
                continue
            c = self.sym_freq.get(x, 0) + delta
            if c <= 0:
                self.sym_freq.pop(x, None)
            else:
                self.sym_freq[x] = c

        for p in set(old_pairs) | set(new_pairs):
            delta = (new_pairs.get(p, 0) - old_pairs.get(p, 0)) * f
            if delta:
                c = self.pair_counts.get(p, 0) + delta
                if c <= 0:
                    # 计数归零 ⇒ 没有任何词还含有它，索引表可以整体丢掉
                    self.pair_counts.pop(p, None)
                    self.pair_words.pop(p, None)
                else:
                    self.pair_counts[p] = c
                    if self.use_heap:
                        heapq.heappush(self.heap, (-self.score_fn(p, c, self.sym_freq), p))
            if new_pairs.get(p, 0):
                self.pair_words.setdefault(p, set()).add(i)
            else:
                self.pair_words.get(p, set()).discard(i)

        self.seqs[i] = new_seq

    def run(self, budget: int) -> list[Pair]:
        merges: list[Pair] = []
        while len(merges) < budget:
            pair = self._pop_best()
            if pair is None:
                break
            self.existing.add(self._merge_result(pair))
            merges.append(pair)
            # 先快照再遍历：_apply_to_word 可能会把 pair_words[pair] 整个丢掉
            for i in sorted(self.pair_words.get(pair, ())):
                self._apply_to_word(i, pair)
        return merges


def train_merges(
    words: Sequence[str],
    freqs: Sequence[int],
    budget: int,
    *,
    make_seq: Callable[[str], list[str]],
    merge_fn: MergeFn,
    score_fn: ScoreFn,
    existing: set[str],
    use_heap: bool = True,
) -> list[Pair]:
    """增量版：返回按学习顺序排列的合并表。"""
    trainer = MergeTrainer(
        words,
        freqs,
        make_seq=make_seq,
        merge_fn=merge_fn,
        score_fn=score_fn,
        existing=existing,
        use_heap=use_heap,
    )
    return trainer.run(budget)


def train_merges_naive(
    words: Sequence[str],
    freqs: Sequence[int],
    budget: int,
    *,
    make_seq: Callable[[str], list[str]],
    merge_fn: MergeFn,
    score_fn: ScoreFn,
    existing: set[str],
) -> list[Pair]:
    """朴素参照实现：每一步都把整个语料重数一遍。**只用于测试比对**。

    故意写得笨 —— 它的价值就在于「笨到不可能错」。
    """
    seqs = [make_seq(w) for w in words]
    seen = set(existing)
    for seq in seqs:
        seen.update(seq)
    merges: list[Pair] = []
    while len(merges) < budget:
        pair_counts: Counter = Counter()
        sym_freq: Counter = Counter()
        for seq, f in zip(seqs, freqs):
            for x in seq:
                sym_freq[x] += f
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += f
        pair = _best_pair_scan(pair_counts, sym_freq, score_fn, seen, merge_fn)
        if pair is None:
            break
        seen.add(merge_fn(*pair))
        merges.append(pair)
        seqs = [merge_sequence(seq, pair, merge_fn) for seq in seqs]
    return merges
