"""Tokenizer Diff —— 回答「差异从哪来」。

这一层最值得守的三条：

1. **边界必须跨算法可比**。WordPiece 的续接 token 带 ``##``，直接拿
   ``len(token)`` 算字符边界会让它的边界整体右移 2，于是算出来的
   「边界一致度」全是假的 —— 而它**不会报错**，只会让 WordPiece 在报告里
   显得比实际更不一致。所以必须走 ``plain_piece``。
2. **Diff 展示的切分必须与度量用的切分是同一条路径**。否则「用 Diff 解释
   矩阵里的差异」这句话就是空话。
3. **入选者与每个算法都必须出现在 Diff 里**。少一个算法，读者就无从判断
   「到底是算法的问题还是词表规模的问题」。
"""

from __future__ import annotations

import pytest

from tokenizerfactory.corpus import pre_tokenize
from tokenizerfactory.diff import (
    boundary_set,
    compute_diffs,
    diff_specs,
    jaccard,
    piece_length,
    tokenize_line,
)
from tokenizerfactory.normalize import normalize, normalize_words
from tokenizerfactory.schema import UNK_TOKEN, TokenizerSpec
from tokenizerfactory.tokenizers import train_tokenizer

# ----------------------------------------------------------------------
# 集合与长度：Diff 的两个原子操作
# ----------------------------------------------------------------------
def test_jaccard_basics() -> None:
    assert jaccard([], []) == 1.0  # 两个空集视为完全一致
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2}, {3, 4}) == 0.0
    assert jaccard({1, 2, 3, 4}, {3, 4, 5, 6}) == pytest.approx(2 / 6)


def test_jaccard_is_symmetric_and_ignores_duplicates() -> None:
    assert jaccard([1, 1, 2], [2, 3]) == jaccard([2, 3], [1, 1, 2])


def test_piece_length_counts_unknown_as_one_character(counts: dict[str, int]) -> None:
    """``[UNK]`` 按 1 个字符算。

    它代表的那段信息已经丢了；按字面量的 5 个字符算会把边界推出去，
    造出一批不存在的「不一致」。
    """
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), counts)
    assert piece_length(tokenizer, UNK_TOKEN) == 1
    assert piece_length(tokenizer, "ab") == 2


def test_piece_length_strips_the_wordpiece_continuation_prefix(
    counts: dict[str, int],
) -> None:
    tokenizer = train_tokenizer(TokenizerSpec("wordpiece", 160), counts)
    assert piece_length(tokenizer, "##ing") == 3


def test_boundary_set_excludes_the_final_position(counts: dict[str, int]) -> None:
    """末位恒等于总长，对所有 spec 都一样，没有信息量。"""
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), counts)
    tokens = ["ab", "c", "de"]
    assert boundary_set(tokenizer, tokens) == {2, 3}
    assert boundary_set(tokenizer, ["ab"]) == set()
    assert boundary_set(tokenizer, []) == set()


def test_boundary_set_is_identical_for_plain_pieces_across_algorithms(
    counts: dict[str, int],
) -> None:
    """把 WordPiece 的 token 换成字面量后，边界集合必须**逐位相同**。

    这条测试直接对着 ``plain_piece`` 的存在理由：若 ``piece_length`` 用了
    ``len(token)``，带 ``##`` 的 token 会各多算 2 个字符，边界集合整体错位。
    """
    wordpiece = train_tokenizer(TokenizerSpec("wordpiece", 400), counts)
    identity = train_tokenizer(TokenizerSpec("bpe", 160), counts)

    # ``tokenizational`` 会切成 ``▁tokenization`` + ``##a`` + ``##l``。
    # 续接 token 必须**不在末位**：``boundary_set`` 会丢掉最后一个位置，
    # 若 ``##`` token 恰好是最后一个，用不用 ``plain_piece`` 都算出同一个集合，
    # 这条测试就没有牙齿了（第一版就是这么写的，被这条断言抓出来）。
    line = "tokenizational"
    tokens = tokenize_line(wordpiece, line)
    assert len(tokens) >= 3 and tokens[1].startswith("##"), tokens

    via_plain = boundary_set(identity, [wordpiece.plain_piece(t) for t in tokens])
    assert boundary_set(wordpiece, tokens) == via_plain

    # 反证：直接用原始 token 去算，边界会不一样 —— 说明这条测试有牙齿
    assert boundary_set(identity, tokens) != via_plain


def test_boundary_set_positions_never_exceed_the_text_length(
    counts: dict[str, int], train_text: str
) -> None:
    for algorithm in ("bpe", "wordpiece", "unigram"):
        tokenizer = train_tokenizer(TokenizerSpec(algorithm, 400), counts)
        tokens = tokenize_line(tokenizer, train_text)
        total = len("".join(tokenizer.plain_piece(t) for t in tokens if t != UNK_TOKEN))
        assert all(0 < p <= total for p in boundary_set(tokenizer, tokens)), algorithm


# ----------------------------------------------------------------------
# 切分路径必须与度量一致
# ----------------------------------------------------------------------
def test_tokenize_line_uses_the_same_path_as_measure(counts: dict[str, int]) -> None:
    """归一化 → 预分词 → 可选小写化 → 编码，一步都不能少。"""
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 400), counts)
    line = "the cat sat on the mat. 数据 与 模型。"

    tokens = tokenize_line(tokenizer, line)
    words = normalize_words(
        pre_tokenize(normalize(line, form="NFC"), tokenizer.spec.pre_tokenizer),
        tokenizer.spec.lowercase,
    )
    assert tokens == tokenizer.encode(words).tokens


def test_tokenize_line_honours_the_pre_tokenizer_rule(counts: dict[str, int]) -> None:
    """Diff 也必须尊重 spec 的预分词规则，否则跨规则对比是假的。

    这一行是纯中文、无标点：``unicode`` 与 ``whitespace`` 给出**完全一样**的
    切分（都是按空白切），而 ``char`` 规则把每个汉字都切成一个词。
    顺带把「``unicode`` 与 ``whitespace`` 在纯中文下等价」这件事固定下来 ——
    它是 ``corpus`` 里那条实测结论的可执行版本。
    """
    line = "数据 与 模型 的 训练"
    unicode_tok = train_tokenizer(TokenizerSpec("bpe", 400, "unicode"), counts)
    whitespace_tok = train_tokenizer(TokenizerSpec("bpe", 400, "whitespace"), counts)
    char_tok = train_tokenizer(TokenizerSpec("bpe", 400, "char"), counts)

    unicode_tokens = tokenize_line(unicode_tok, line)
    assert tokenize_line(whitespace_tok, line) == unicode_tokens
    assert len(tokenize_line(char_tok, line)) > len(unicode_tokens)


def test_tokenize_line_keeps_every_character_when_nothing_is_unknown(
    counts: dict[str, int], train_text: str
) -> None:
    """已知文本必须无损：切分只是重新分组，不能丢字符。"""
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 400), counts)
    tokens = tokenize_line(tokenizer, train_text)
    assert UNK_TOKEN not in tokens
    words = normalize_words(pre_tokenize(normalize(train_text), "unicode"), False)
    assert "".join(tokenizer.plain_piece(t) for t in tokens) == "".join(words)


# ----------------------------------------------------------------------
# 选哪些 spec 进 Diff
# ----------------------------------------------------------------------
def test_diff_specs_always_contains_the_selection_and_every_algorithm(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    picked = diff_specs(result, tiny_cfg, max_specs=100)
    assert result.selected in picked
    assert set(result.algorithms()) <= {result.find(s).algorithm for s in picked}


def test_diff_specs_contains_the_whole_main_frontier(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    from tokenizerfactory.compare import frontier_key

    main = result.frontiers[frontier_key(tiny_cfg["pareto"]["objectives"])]
    picked = set(diff_specs(result, tiny_cfg, max_specs=100))
    assert set(main) <= picked


def test_diff_specs_is_sorted_by_budget_then_algorithm(tiny_result, tiny_cfg) -> None:
    """表要从上到下读起来像「预算从小到大」，不是一串随机 id。"""
    result, _ = tiny_result
    picked = diff_specs(result, tiny_cfg, max_specs=100)
    keys = [(result.find(s).spec.vocab_size, result.find(s).algorithm) for s in picked]
    assert keys == sorted(keys)


def test_diff_specs_respects_max_specs(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    picked = diff_specs(result, tiny_cfg, max_specs=3)
    assert len(picked) == 3
    # 截断也要保住入选者（它是要交付的东西）
    assert result.selected in picked


def test_diff_specs_puts_every_algorithm_in_frame_with_the_same_rule(
    tiny_result, tiny_cfg
) -> None:
    """每个算法都要出现在 Diff 里，**且优先与入选者同一个预分词规则**。

    不固定规则的话，实测下来 unigram 与 wordpiece 会被补上 ``whitespace`` 变体
    （它的 ``tokens_per_char`` 恰好略好一点），于是那张表一次动了两个变量 ——
    读者无法判断某一行变长是因为换了算法还是换了规则。
    """
    result, _ = tiny_result
    picked = diff_specs(result, tiny_cfg, max_specs=100)
    selected = result.find(result.selected)
    picked_runs = [result.find(sid) for sid in picked]

    algorithms = {r.algorithm for r in picked_runs}
    assert algorithms == set(result.algorithms())

    for algorithm in result.algorithms():
        same_rule = [
            r
            for r in picked_runs
            if r.algorithm == algorithm
            and r.spec.pre_tokenizer == selected.spec.pre_tokenizer
        ]
        assert same_rule, f"{algorithm} 没有与入选者同规则的列"


def test_diff_specs_isolates_the_pre_tokenizer_axis(tiny_result, tiny_cfg) -> None:
    """每条预分词规则都要有一列，且与入选者**同算法** —— 隔离「规则」这个轴。

    没有 ``char`` 这一列，「粒度」这个轴在表里就只有一个端点，
    「同一个词是 2 个 token 还是 8 个」这句话就没有参照。
    """
    result, _ = tiny_result
    picked = diff_specs(result, tiny_cfg, max_specs=100)
    selected = result.find(result.selected)
    picked_runs = [result.find(sid) for sid in picked]

    rules = {r.spec.pre_tokenizer for r in result.runs}
    for rule in rules:
        same_algorithm = [
            r
            for r in picked_runs
            if r.spec.pre_tokenizer == rule
            and r.algorithm == selected.spec.algorithm
        ]
        assert same_algorithm, f"规则 {rule} 没有与入选者同算法的列"


def test_diff_specs_keeps_the_selection_when_truncating(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    for limit in (1, 2, 3):
        picked = diff_specs(result, tiny_cfg, max_specs=limit)
        assert len(picked) <= limit
        assert result.selected in picked


def test_diff_specs_has_no_duplicates(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    picked = diff_specs(result, tiny_cfg, max_specs=100)
    assert len(picked) == len(set(picked))


# ----------------------------------------------------------------------
# 完整结果
# ----------------------------------------------------------------------
def test_compute_diffs_structure(tiny_result, tiny_cfg, tiny_diff_lines) -> None:
    result, tokenizers = tiny_result
    diffs = compute_diffs(
        result,
        tiny_cfg,
        tokenizers,
        lines=tiny_diff_lines,
        form=tiny_cfg["normalize"]["form"],
    )

    assert diffs["lines"] == tiny_diff_lines
    assert diffs["selected"] == result.selected
    assert diffs["specs"] == diff_specs(result, tiny_cfg, max_specs=100)
    assert len(diffs["table"]) == len(tiny_diff_lines)

    for sid in diffs["specs"]:
        assert sid in diffs["vocab_size"]
        assert sid in diffs["algorithm"]
        assert sid in diffs["mean_tokens_per_line"]
        assert sid in diffs["mean_agreement"]
        assert diffs["vocab_size"][sid] > 0

    n = len(diffs["specs"])
    assert len(diffs["agreement"]) == n * (n - 1) // 2
    for value in diffs["agreement"].values():
        assert 0.0 <= value <= 1.0


def test_compute_diffs_table_rows_are_consistent(tiny_result, tiny_cfg, tiny_diff_lines) -> None:
    result, tokenizers = tiny_result
    diffs = compute_diffs(
        result, tiny_cfg, tokenizers, lines=tiny_diff_lines, form="NFC"
    )
    for row in diffs["table"]:
        assert set(row["specs"]) == set(diffs["specs"])
        for payload in row["specs"].values():
            assert payload["n_tokens"] == len(payload["tokens"])
            assert payload["rendered"] == " | ".join(payload["tokens"])


def test_compute_diffs_mean_tokens_per_line_matches_the_table(
    tiny_result, tiny_cfg, tiny_diff_lines
) -> None:
    result, tokenizers = tiny_result
    diffs = compute_diffs(result, tiny_cfg, tokenizers, lines=tiny_diff_lines, form="NFC")
    for sid in diffs["specs"]:
        counts = [row["specs"][sid]["n_tokens"] for row in diffs["table"]]
        assert diffs["mean_tokens_per_line"][sid] == pytest.approx(
            sum(counts) / len(counts), abs=1e-4
        )


def test_compute_diffs_mean_agreement_is_the_mean_over_pairs(
    tiny_result, tiny_cfg, tiny_diff_lines
) -> None:
    result, tokenizers = tiny_result
    diffs = compute_diffs(result, tiny_cfg, tokenizers, lines=tiny_diff_lines, form="NFC")
    for sid in diffs["specs"]:
        keys = [k for k in diffs["agreement"] if sid in k.split("~")]
        assert keys, sid
        expected = sum(diffs["agreement"][k] for k in keys) / len(keys)
        assert diffs["mean_agreement"][sid] == pytest.approx(expected, abs=2e-6)


def test_jaccard_of_a_tokenization_with_itself_is_one(counts: dict[str, int]) -> None:
    """同一个切分和它自己比，边界一致度必须是 1.0。"""
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 400), counts)
    tokens = tokenize_line(tokenizer, "tokenizations 数据 与 模型")
    boundaries = boundary_set(tokenizer, tokens)
    assert jaccard(boundaries, boundaries) == 1.0
    assert boundaries  # 不是空集，否则上面那条恒真


def test_compute_diffs_rejects_a_partial_tokenizer_map(
    tiny_result, tiny_cfg, tiny_diff_lines
) -> None:
    """``tokenizers`` 少给几个 spec 时必须报错，而不是静默少算几个。

    静默跳过会让报告里的 Diff 少几个 spec，而读者不会知道少了什么。
    """
    result, tokenizers = tiny_result
    sid = result.selected
    with pytest.raises(ValueError, match="缺少这些 spec 的分词器"):
        compute_diffs(
            result, tiny_cfg, {sid: tokenizers[sid]}, lines=tiny_diff_lines, form="NFC"
        )


def test_compute_diffs_handles_empty_lines(tiny_result, tiny_cfg) -> None:
    """没有行时不能除零，也不能凭空造出「不一致」。"""
    result, tokenizers = tiny_result
    diffs = compute_diffs(result, tiny_cfg, tokenizers, lines=[], form="NFC")
    assert diffs["table"] == []
    assert all(v == 0.0 for v in diffs["mean_tokens_per_line"].values())
    assert all(v == 1.0 for v in diffs["agreement"].values())


def test_compute_diffs_marks_unknown_characters(tiny_result, tiny_cfg) -> None:
    """硬字符必须真的落到 ``[UNK]``，否则 Diff 展示的是一个不存在的世界。"""
    result, tokenizers = tiny_result
    sid = result.selected
    diffs = compute_diffs(
        result, tiny_cfg, tokenizers, lines=["龘靐齉"], form="NFC"
    )
    rendered = diffs["table"][0]["specs"][sid]["rendered"]
    assert UNK_TOKEN in rendered
