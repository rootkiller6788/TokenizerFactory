"""三个分词算法。

这是本项目最核心的一层，也是最容易「跑得通但结果悄悄不对」的一层。
所以这里有一类别处没有的测试：**朴素参照实现比对**。

增量合并（只更新受影响的词）比朴素实现（每步重数整个语料）快几十倍，
但它的簿记错误不会报错 —— 只会让词表略微不同。那正好是分词器对比里
最致命的一类错误：报告照常生成，结论悄悄变掉。
所以 ``train_merges`` 与 ``train_merges_naive`` 必须给出**完全相同**的合并序列。

语料与词频夹具（``train_text`` / ``counts`` / ``alphabet`` / ``big_counts``）
定义在 ``conftest.py``：度量与对比层的测试同样需要它们。
"""

from __future__ import annotations

import pytest

from tokenizerfactory.corpus import SPACE_MARKER, alphabet_of, pre_tokenize
from tokenizerfactory.normalize import normalize, normalize_words
from tokenizerfactory.schema import CONTINUATION_PREFIX, UNK_TOKEN, TokenizerSpec
from tokenizerfactory.tokenizers import (
    ALGORITHMS,
    TOKENIZERS,
    load_tokenizer,
    minimum_vocab_size,
    train_tokenizer,
)
from tokenizerfactory.tokenizers.bpe import bpe_merge, bpe_score
from tokenizerfactory.tokenizers.merge import train_merges, train_merges_naive
from tokenizerfactory.tokenizers.unigram import (
    MIN_PROB,
    em_fit,
    logadd,
    prune_pieces,
    seed_pieces,
)
from tokenizerfactory.tokenizers.wordpiece import wp_make_seq, wp_merge, wp_score


def _words_freqs(counts: dict[str, int]) -> tuple[list[str], list[int]]:
    words = sorted(counts)
    return words, [counts[w] for w in words]


# ----------------------------------------------------------------------
# 增量 vs 朴素 —— 最重要的三条
# ----------------------------------------------------------------------
@pytest.mark.parametrize("budget", [10, 60, 500])
def test_bpe_incremental_matches_naive(
    budget: int, big_counts: dict[str, int], alphabet: set[str]
) -> None:
    """增量合并（只更新受影响的词）必须与朴素实现给出**完全相同**的合并序列。

    这是本项目最贵的一条测试。增量版本比朴素快几十倍，但它的簿记错误
    **不会报错** —— 只会让词表略微不同，而那正好是分词器对比里最致命的
    一类错误：报告照常生成，结论悄悄变掉。所以必须有一个「笨到不可能错」
    的参照实现逐项比对。

    ``budget=500`` 那一档同时覆盖了「预算能被用满」的路径。
    """
    words, freqs = _words_freqs(big_counts)
    base = {UNK_TOKEN} | alphabet
    incremental = train_merges(
        words, freqs, budget,
        make_seq=list, merge_fn=bpe_merge, score_fn=bpe_score,
        existing=set(base), use_heap=True,
    )
    naive = train_merges_naive(
        words, freqs, budget,
        make_seq=list, merge_fn=bpe_merge, score_fn=bpe_score, existing=set(base),
    )
    assert incremental == naive
    assert len(incremental) == budget, "这个合成语料必须能支撑满预算"


def test_bpe_heap_and_scan_paths_agree(
    big_counts: dict[str, int], alphabet: set[str]
) -> None:
    """BPE 的惰性堆分支与全扫描分支必须一致。

    两条分支都对 BPE 成立（它的分数只取决于 pair 频次，堆可以精确失效），
    所以「用哪条」不该影响结果。不一致说明堆的失效判定写错了 ——
    那种错误表现为「词表少了几个合并」，同样不报错。
    """
    words, freqs = _words_freqs(big_counts)
    base = {UNK_TOKEN} | alphabet
    heap = train_merges(
        words, freqs, 300,
        make_seq=list, merge_fn=bpe_merge, score_fn=bpe_score,
        existing=set(base), use_heap=True,
    )
    scan = train_merges(
        words, freqs, 300,
        make_seq=list, merge_fn=bpe_merge, score_fn=bpe_score,
        existing=set(base), use_heap=False,
    )
    assert heap == scan


def test_wordpiece_incremental_matches_naive(
    big_counts: dict[str, int], alphabet: set[str]
) -> None:
    """WordPiece 走的是全扫描分支（它的分数依赖符号频次，惰性堆无法可靠失效）。

    两条路径都必须和朴素实现一致 —— 否则「WordPiece 比 BPE 差」这个结论
    可能只是增量簿记的产物。
    """
    words, freqs = _words_freqs(big_counts)
    base = set(TOKENIZERS["wordpiece"].base_vocab(alphabet))
    incremental = train_merges(
        words, freqs, 300,
        make_seq=wp_make_seq, merge_fn=wp_merge, score_fn=wp_score,
        existing=set(base), use_heap=False,
    )
    naive = train_merges_naive(
        words, freqs, 300,
        make_seq=wp_make_seq, merge_fn=wp_merge, score_fn=wp_score, existing=set(base),
    )
    assert incremental == naive
    assert len(incremental) == 300


def test_merge_trainer_skips_already_existing_results(counts: dict[str, int]) -> None:
    """合并结果已存在时必须跳过，否则词表会出现重复 token。

    ``ab``+``c`` 与 ``a``+``bc`` 都拼出 ``abc``。不跳过的话词表里会有两个
    ``abc``：``token_to_id`` 指向后一个、``len(vocab)`` 虚增、
    ``vocab_digest`` 不再唯一标识这个分词器 —— 而且全程不报错。
    """
    words, freqs = _words_freqs(counts)
    merges = train_merges(
        words, freqs, 300,
        make_seq=list, merge_fn=bpe_merge, score_fn=bpe_score,
        existing=set(), use_heap=True,
    )
    results = [bpe_merge(*m) for m in merges]
    assert len(set(results)) == len(results), "合并结果出现了重复"


def test_merge_trainer_stops_when_no_pairs_left() -> None:
    merges = train_merges(
        ["ab"], [1], 100,
        make_seq=list, merge_fn=bpe_merge, score_fn=bpe_score,
        existing=set(), use_heap=True,
    )
    assert merges == [("a", "b")]


# ----------------------------------------------------------------------
# 契约：已知片段绝不产生 UNK
# ----------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_encode_run_never_returns_unknown_for_known_runs(
    algorithm: str, counts: dict[str, int], alphabet: set[str]
) -> None:
    """``_encode_run`` 的契约。

    违反它只会表现为未知率悄悄偏高，不会报错 —— 所以必须有测试守着。
    """
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 128), counts)
    for word in sorted(counts)[:40]:
        for run, known in tokenizer._split_known_runs(word):
            if not known:
                continue
            pieces = tokenizer._encode_run(run)
            assert UNK_TOKEN not in pieces, (algorithm, word, run, pieces)
            assert "".join(tokenizer.plain_piece(p) for p in pieces) == run


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_known_text_round_trips_through_decode(
    algorithm: str, counts: dict[str, int], train_text: str
) -> None:
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 128), counts)
    text = normalize(train_text)
    words = pre_tokenize(text, "unicode")
    enc = tokenizer.encode(words)
    assert enc.n_unknown == 0
    assert tokenizer.decode(enc.ids) == " " + text


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_unknown_characters_fall_back_per_character(algorithm: str, counts: dict[str, int]) -> None:
    """未登录字符按**字符级**回退，而不是整词退化。

    整词退化会让 ``tokens_per_char`` 因为信息丢失而虚高（一个词塌成一个 token），
    于是「谁压得更狠」这条主轴上会混进「谁丢得更多」。这条断言把这个约定钉住。

    注意 ``▁`` 是**已知**字符，所以它会单独成为一个 token ——
    未知计数是 3（三个未登录汉字），而不是 4。
    """
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 160), counts)
    word = SPACE_MARKER + "龘靐齉"  # 三个未登录汉字
    pieces, n_unknown = tokenizer.encode_word(word)
    assert n_unknown == 3
    assert pieces.count(UNK_TOKEN) == 3
    assert SPACE_MARKER in "".join(tokenizer.plain_piece(p) for p in pieces)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_mixed_known_and_unknown_keeps_the_known_part(algorithm: str, counts: dict[str, int]) -> None:
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 128), counts)
    word = SPACE_MARKER + "a龘b"
    pieces, n_unknown = tokenizer.encode_word(word)
    assert n_unknown == 1
    assert UNK_TOKEN in pieces
    joined = "".join(tokenizer.plain_piece(p) for p in pieces if p != UNK_TOKEN)
    assert joined == SPACE_MARKER + "ab"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_unknown_count_is_not_derived_from_token_literals(
    algorithm: str, counts: dict[str, int]
) -> None:
    """``n_unknown`` 是单独计数的，不是数 tokens 里的 ``[UNK]``。

    语料里可能真的出现过 ``[UNK]`` 这个字符串并被学进词表。
    那样数 tokens 会把「恰好长这样的**已知** token」也算成未知 ——
    未知率会凭空偏高，而且看不出原因。

    构造方式：把一个字面量为 ``▁[UNK]`` 的「词」以高频放进词频表。
    它的每个字符都会进入基础字符集，因此它一定能被编码、且**一个未知都没有**；
    如果其中一部分恰好被学成了字面量 ``[UNK]`` 的 token，它仍然不是未知。
    """
    counts_with_literal = dict(counts)
    counts_with_literal[SPACE_MARKER + UNK_TOKEN] = 40
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 400), counts_with_literal)
    word = SPACE_MARKER + UNK_TOKEN
    pieces, n_unknown = tokenizer.encode_word(word)
    assert n_unknown == 0, "所有字符都在基础字符集里，不该产生未知"
    assert "".join(tokenizer.plain_piece(p) for p in pieces) == word


# ----------------------------------------------------------------------
# 词表规模与唯一性
# ----------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("size", [160, 256, 400])
def test_vocab_never_exceeds_the_target(algorithm: str, size: int, counts: dict[str, int]) -> None:
    """「同等词表预算下谁压得更狠」这个对比的前提。

    允许**小于**目标（算法提前用尽可合并的 pair / 可剪的子串），
    但不允许超过 —— 超过就意味着两个 spec 不是在同一预算下比较。
    """
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, size), counts)
    assert len(tokenizer.vocab) <= size
    assert len(set(tokenizer.vocab)) == len(tokenizer.vocab)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_vocab_contains_unk_first_and_all_characters(algorithm: str, counts: dict[str, int]) -> None:
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 160), counts)
    assert tokenizer.vocab[0] == UNK_TOKEN
    assert tokenizer.unk_id == 0
    assert alphabet_of(counts) <= set(tokenizer.vocab)


def test_vocab_digest_separates_algorithms_at_the_same_size(counts: dict[str, int]) -> None:
    """同一 vocab_size 下三个算法的词表不同 → 摘要必须不同。

    这是「词表本身要进 digest」这条设计决定的可执行形式：
    只对 spec 取摘要的话，三个算法在同一个 vocab_size 下会撞成同一个版本号。
    """
    digests = {
        a: train_tokenizer(TokenizerSpec(a, 160), counts).vocab_digest
        for a in ALGORITHMS
    }
    assert len(set(digests.values())) == len(ALGORITHMS)


def test_vocab_digest_is_newline_joined() -> None:
    """``["ab","c"]`` 与 ``["a","bc"]`` 是完全不同的分词器，摘要不能撞。"""
    from tokenizerfactory.tokenizers.base import vocab_digest

    assert vocab_digest(["ab", "c"]) != vocab_digest(["a", "bc"])


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_reserved_unk_never_appears_twice_in_the_vocab(
    algorithm: str, counts: dict[str, int]
) -> None:
    """语料里出现字面量 ``[UNK]`` 时，词表里不能有两个 ``[UNK]``。

    **这条测试抓出过一个真 bug**：Unigram 的种子词表会把字面量 ``[UNK]``
    当成一个普通子串学下来，于是最终词表里 ``[UNK]`` 出现两次 ——
    ``token_to_id`` 指向后一个、``len(vocab)`` 虚增、``vocab_digest``
    不再唯一标识这个分词器。全程不报错，只有 ``Tokenizer.__init__``
    的去重检查会拦住它，而那个检查是「不该走到」的兜底。

    BPE / WordPiece 靠合并循环的 ``existing`` 集合挡住了（整个基础词表
    都在里面，合并出 ``[UNK]`` 的那一步被跳过）；Unigram 必须显式排除。
    """
    counts_with_literal = dict(counts)
    counts_with_literal[SPACE_MARKER + UNK_TOKEN] = 40
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 400), counts_with_literal)
    assert tokenizer.vocab.count(UNK_TOKEN) == 1
    assert len(set(tokenizer.vocab)) == len(tokenizer.vocab)
    assert tokenizer.unk_id == 0


# ----------------------------------------------------------------------
# 导出 / 载入往返
# ----------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_export_load_round_trip_is_identical(
    algorithm: str, counts: dict[str, int], train_text: str
) -> None:
    """导出的产物必须是一个**能用的分词器**，不是一个 JSON 快照。

    只落词表是不够的：BPE / WordPiece 重新编码还需要合并表，
    Unigram 还需要每个 piece 的对数概率。
    """
    tokenizer = train_tokenizer(TokenizerSpec(algorithm, 160), counts)
    payload = tokenizer.export_dict()
    reloaded = load_tokenizer(payload)
    words = normalize_words(pre_tokenize(normalize(train_text), "unicode"), False)
    original = tokenizer.encode(words)
    assert reloaded.encode(words).tokens == original.tokens
    assert reloaded.encode(words).ids == original.ids
    assert reloaded.vocab == tokenizer.vocab
    assert reloaded.decode(original.ids) == tokenizer.decode(original.ids)


def test_load_tokenizer_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError):
        load_tokenizer({"algorithm": "nope", "spec": {}, "vocab": []})


# ----------------------------------------------------------------------
# 各算法的特有性质
# ----------------------------------------------------------------------
def test_bpe_encoding_applies_merges_in_learned_order(counts: dict[str, int]) -> None:
    """编码按**合并的先后顺序**施加，不按频次。

    按频次编码会得到一个与训练不一致的分词器：压缩率看着还行，
    但和词表对不上。构造一个「高频 pair 晚学到」的例子来区分两种做法。
    """
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 200), counts)
    ranks = [tokenizer._merge_rank[m] for m in tokenizer.merges]
    assert ranks == sorted(ranks), "合并表本身必须按学习顺序排列"
    # 编码结果里的每个多字 token 都必须在词表里
    for word in list(counts)[:30]:
        for piece in tokenizer.pieces_of(word):
            assert piece in tokenizer.token_to_id


def test_wordpiece_continuation_prefix_semantics(counts: dict[str, int]) -> None:
    """``##`` 恰好落在「词的第二个字符及以后」。"""
    assert wp_make_seq("abc") == ["a", "##b", "##c"]
    assert wp_make_seq("") == []
    assert wp_merge("a", "##b") == "ab"
    assert wp_merge("##a", "##b") == "##ab"
    assert wp_merge("##a", "b") == "##ab"

    tokenizer = train_tokenizer(TokenizerSpec("wordpiece", 160), counts)
    for word in list(counts)[:30]:
        for piece in tokenizer.pieces_of(word):
            if piece == UNK_TOKEN:
                continue
            assert piece in tokenizer.token_to_id
    # 词首 token 不带前缀，续接 token 一定带（除非它是整词）
    pieces = tokenizer.pieces_of(SPACE_MARKER + "tokenizer")
    assert pieces[0].startswith(SPACE_MARKER)
    for piece in pieces[1:]:
        assert piece.startswith(CONTINUATION_PREFIX), pieces


def test_wordpiece_base_vocab_is_twice_the_alphabet(alphabet: set[str]) -> None:
    base = TOKENIZERS["wordpiece"].base_vocab(alphabet)
    assert len(base) == 1 + 2 * len(alphabet)
    assert len(TOKENIZERS["bpe"].base_vocab(alphabet)) == 1 + len(alphabet)


def test_minimum_vocab_size_takes_the_max_over_algorithms(alphabet: set[str]) -> None:
    assert minimum_vocab_size(["bpe"], alphabet) == 1 + len(alphabet)
    assert minimum_vocab_size(["wordpiece"], alphabet) == 1 + 2 * len(alphabet)
    assert minimum_vocab_size(list(ALGORITHMS), alphabet) == 1 + 2 * len(alphabet)
    assert minimum_vocab_size([], alphabet) == 0


def test_train_tokenizer_rejects_empty_counts() -> None:
    with pytest.raises(ValueError):
        train_tokenizer(TokenizerSpec("bpe", 128), {})


# -- Unigram ------------------------------------------------------------
def test_logadd_handles_neg_inf() -> None:
    assert logadd(float("-inf"), 0.0) == 0.0
    assert logadd(0.0, float("-inf")) == 0.0
    assert abs(logadd(0.0, 0.0) - 0.6931471805599453) < 1e-12


def test_em_log_likelihood_does_not_decrease(counts: dict[str, int]) -> None:
    """EM 的单调性：每轮迭代的对数似然不应下降。

    这是 EM 正确性的一个必要条件。下降了说明 E 步或 M 步的边际化写错了 ——
    而那种错误只会让分词质量略差，不会报错。
    """
    import math

    pieces = seed_pieces(counts, max_len=4, min_count=1, cap=2000)

    def total_ll(lp: dict[str, float]) -> float:
        total = 0.0
        for word, freq in counts.items():
            best = [float("-inf")] * (len(word) + 1)
            best[0] = 0.0
            for i in range(1, len(word) + 1):
                for length in range(1, min(4, i) + 1):
                    value = lp.get(word[i - length : i])
                    if value is None or best[i - length] == float("-inf"):
                        continue
                    best[i] = max(best[i], best[i - length] + value)
            if best[len(word)] == float("-inf"):
                return float("-inf")
            total += freq * best[len(word)]
        return total

    history = []
    for iters in range(1, 6):
        lp, _ = em_fit(pieces, counts, iters)
        history.append(total_ll(lp))
    assert all(math.isfinite(v) for v in history), history
    for prev, cur in zip(history, history[1:]):
        assert cur >= prev - 1e-6, history


def test_em_gives_zero_mass_pieces_the_floor_probability(counts: dict[str, int]) -> None:
    """期望次数为 0 的 piece 必须拿到概率下限，而不是 0。

    用 0 会让 ``log(0) = -inf`` 把整条路径变成不可达，于是含它的词整段变 UNK。

    构造方式：往种子词表里塞一个**不可能是任何词的子串**的 piece
    （``ZZZZ`` —— ``Z`` 不在语料的字符集里），它的期望次数必然是 0。
    这比「靠剪枝恰好造出零期望 piece」可靠得多。
    """
    import math

    pieces = seed_pieces(counts, max_len=4, min_count=1, cap=2000) | {"ZZZZ"}
    lp, expected = em_fit(pieces, counts, 3)
    assert set(lp) == pieces
    assert expected.get("ZZZZ", 0.0) == 0.0
    assert lp["ZZZZ"] == math.log(MIN_PROB)
    assert all(math.isfinite(v) for v in lp.values()), "存在 -inf 概率"


def test_seed_pieces_always_keeps_single_characters(counts: dict[str, int]) -> None:
    """单字符无条件保留 —— 否则会有字符失去全部覆盖路径。"""
    pieces = seed_pieces(counts, max_len=2, min_count=99, cap=100000)
    assert alphabet_of(counts) <= pieces


def test_seed_pieces_respects_cap_and_min_count() -> None:
    counts = {"aaa": 10, "aab": 1}
    pieces = seed_pieces(counts, max_len=3, min_count=2, cap=100000)
    assert "aab" not in pieces
    assert "aa" in pieces
    assert {"a", "b"} <= pieces


def test_prune_pieces_never_removes_single_characters(counts: dict[str, int]) -> None:
    pieces = seed_pieces(counts, max_len=4, min_count=1, cap=2000)
    _, expected = em_fit(pieces, counts, 2)
    from tokenizerfactory.rng import make_rng

    keep = prune_pieces(pieces, expected, keep_n=len(alphabet_of(counts)), rng=make_rng("t"))
    assert alphabet_of(counts) <= keep


def test_prune_pieces_is_deterministic_given_the_same_rng(counts: dict[str, int]) -> None:
    from tokenizerfactory.rng import make_rng

    pieces = seed_pieces(counts, max_len=4, min_count=1, cap=2000)
    _, expected = em_fit(pieces, counts, 2)
    a = prune_pieces(pieces, expected, 100, make_rng("seed", 1))
    b = prune_pieces(pieces, expected, 100, make_rng("seed", 1))
    assert a == b


def test_prune_ties_are_broken_by_rng_not_by_lexicographic_order() -> None:
    """并列的损失必须用 rng 打破，不能按字典序。

    损失并列的量非常大（所有「次数 1、长度 2」的子串损失都是 1）。
    按字典序会系统性地先剪掉字典序靠后的那些 —— 用一个与语言无关的因素
    扭曲词表。这里构造一个**完全对称**的场景：四个长度 2 的候选损失相同，
    剪掉其中一个。若按字典序，被剪的永远是 "bb"；用 rng 则随种子变化。
    """
    from collections import Counter

    from tokenizerfactory.rng import make_rng

    pieces = {"a", "b", "aa", "ab", "ba", "bb"}
    expected: Counter = Counter({"aa": 1.0, "ab": 1.0, "ba": 1.0, "bb": 1.0})

    dropped = set()
    for seed in range(24):
        kept = prune_pieces(pieces, expected, keep_n=5, rng=make_rng("sym", seed))
        assert {"a", "b"} <= kept, "单字符必须留下"
        dropped |= set(pieces) - kept
    assert len(dropped) > 1, (
        "不同种子下被剪掉的总是同一个 piece —— 说明并列是按固定顺序打破的，"
        f"而不是用 rng（实际只剪掉了 {dropped}）"
    )

    # 同一个种子必须给出同一个结果
    a = prune_pieces(pieces, expected, keep_n=3, rng=make_rng("sym", 7))
    b = prune_pieces(pieces, expected, keep_n=3, rng=make_rng("sym", 7))
    assert a == b


def test_unigram_encoding_is_viterbi_optimal(counts: dict[str, int]) -> None:
    """编码结果的总对数概率必须等于 Viterbi 的最优值。

    用一个独立写的暴力搜索做对照：枚举所有切分，取概率最大的一种。
    这能抓住「backpointer 记错位置」这类只在长词上暴露的错误。
    """
    from functools import lru_cache

    tokenizer = train_tokenizer(TokenizerSpec("unigram", 160), counts)

    def brute(run: str) -> float:
        @lru_cache(maxsize=None)
        def best(i: int) -> float:
            if i == len(run):
                return 0.0
            out = float("-inf")
            for j in range(i + 1, len(run) + 1):
                value = tokenizer.scores.get(run[i:j])
                if value is None:
                    continue
                sub = best(j)
                if sub != float("-inf"):
                    out = max(out, value + sub)
            return out

        return best(0)

    for word in sorted(counts)[:25]:
        for run, known in tokenizer._split_known_runs(word):
            if not known:
                continue
            pieces = tokenizer._encode_run(run)
            total = sum(tokenizer.scores[p] for p in pieces)
            assert abs(total - brute(run)) < 1e-9, (word, run, pieces, total, brute(run))


def test_unigram_meta_records_the_pruning_story(counts: dict[str, int]) -> None:
    tokenizer = train_tokenizer(TokenizerSpec("unigram", 160), counts)
    meta = tokenizer.meta
    assert meta["n_seeds"] >= meta["n_pieces"]
    assert meta["piece_target"] == 160 - 1
    assert meta["prune_rounds_done"] <= meta["prune_rounds_planned"]
    assert meta["trimmed_directly"] >= 0


def test_bpe_meta_records_stop_reason(counts: dict[str, int]) -> None:
    generous = train_tokenizer(TokenizerSpec("bpe", 100000), counts)
    assert generous.meta["stop_reason"] == "no_more_pairs"
    small = train_tokenizer(TokenizerSpec("bpe", 100), counts)
    assert small.meta["stop_reason"] == "budget"
