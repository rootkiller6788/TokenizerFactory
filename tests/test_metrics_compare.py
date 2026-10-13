"""度量、对比矩阵、Pareto 前沿、选型。

这一层最值得守的三条：

1. **口径**。``tokens_per_char`` 的分母是「归一化后文本的字符数」，
   不是「预分词结果的字符数之和」—— 后者会多出词首标记，把比率系统性压低。
   口径错了不会报错，只会让结论在小差距处翻面。
2. **坐标约定只有一套**。支配/排序/超体积全部走 ``min`` 形式。
   同族项目里出现过「超体积吃最大形式的点、内部却调 ``(quality, cost)``
   约定的支配函数」这种静默削掉半个前沿的 bug。
3. **超体积的阶梯和恒等于总量**。每个点的增量之和必须等于总超体积 ——
   这条恒等式能抓住阶梯求和写错方向（多算或少算面积）而**不报错**的情况。

坐标轴一律用**真实指标名**，不用 ``x`` / ``y`` 这种占位符。
``compare`` 层的每个函数都要查 ``METRIC_DIRECTIONS`` 才能知道方向，
这是刻意的（配置里写错指标名要在加载期就报错，而不是静默算出一个前沿）；
占位符会绕过这条约束，测出来的东西线上不成立。
"""

from __future__ import annotations

import math

import pytest

from tokenizerfactory.compare import (
    REFERENCE_MARGIN,
    assemble,
    build_matrix,
    dominated_count,
    dominators,
    frontier_key,
    frontier_pairs,
    hypervolume_2d,
    hypervolume_of,
    hv_contributions,
    knee_point,
    non_dominated,
    reference_point,
    select,
    to_min_form,
)
from tokenizerfactory.metrics import (
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    measure,
    measure_speed,
    metric_line,
    point_of,
)
from tokenizerfactory.schema import TokenizerMetrics, TokenizerRun, TokenizerSpec
from tokenizerfactory.tokenizers import train_tokenizer

#: 两个**都是 min 方向**的指标，于是「越小越好」对两者同时成立，
#: 手算前沿/超体积时不用在脑子里翻方向。
VOCAB = "vocab_size"
TCP = "tokens_per_char"
AXES = [VOCAB, TCP]


def pt(vocab: float, tokens_per_char: float) -> dict[str, float]:
    """构造一个点。两个轴都是 min 方向（见 ``AXES``）。"""
    return {VOCAB: vocab, TCP: tokens_per_char}


def make_run(
    spec_id_algorithm: str,
    vocab_size: int,
    tokens_per_char: float,
    *,
    speed: float = 1.0,
    unknown_rate: float = 0.0,
    coverage: float = 1.0,
    vocab_actual: int | None = None,
) -> TokenizerRun:
    """手造一个 run，用来测「纯算法性质」——不受真实训练结果干扰。"""
    spec = TokenizerSpec(spec_id_algorithm, vocab_size)
    vocab = [f"t{i}" for i in range(vocab_actual or vocab_size)]
    return TokenizerRun(
        spec=spec,
        vocab=vocab,
        vocab_digest=f"digest-{spec.spec_id}",
        metrics=TokenizerMetrics(
            tokens_per_char=tokens_per_char,
            tokens_per_word=tokens_per_char * 5,
            vocab_coverage=coverage,
            unknown_rate=unknown_rate,
            hard_unknown_rate=unknown_rate,
            encode_tokens_per_sec=speed,
            decode_tokens_per_sec=speed,
            n_tokens=int(tokens_per_char * 1000),
            n_chars=1000,
            n_words=200,
        ),
        train_seconds=0.1,
    )


# ----------------------------------------------------------------------
# 度量口径
# ----------------------------------------------------------------------
def test_metric_directions_and_labels_cover_the_same_keys() -> None:
    assert set(METRIC_DIRECTIONS) == set(METRIC_LABELS)
    assert set(METRIC_DIRECTIONS.values()) <= {"min", "max"}


def test_point_of_exposes_design_and_result_variables(counts: dict[str, int]) -> None:
    """设计变量（vocab_size 目标）与结果变量（vocab_size_actual）必须同时在。

    BPE 在 pair 用尽时会提前停下，``actual < target`` 本身就是一个结果；
    只给一个数的话，「目标 8192 实际 2985」这件事会被藏起来。
    """
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 100000), counts)
    run = make_run("bpe", 100000, 0.5, vocab_actual=len(tokenizer.vocab))
    point = point_of(run)
    assert point["vocab_size"] == 100000
    assert point["vocab_size_actual"] == len(tokenizer.vocab)
    for name in METRIC_DIRECTIONS:
        assert name in point, name
        assert math.isfinite(point[name])


def test_measure_uses_normalized_text_length_as_denominator(counts: dict[str, int]) -> None:
    """``tokens_per_char`` 的分母是归一化后原文的字符数。

    预分词会给第一个词加一个词首标记，于是 ``sum(len(w))`` 比文本长度多 1。
    用后者当分母等于把词首标记当成一个真实字符，比率会被系统性压低一点点 ——
    小到不会引起注意，但足以让「谁压得更狠」在接近的两个 spec 之间翻面。
    """
    from tokenizerfactory.corpus import pre_tokenize, word_counts
    from tokenizerfactory.normalize import normalize, normalize_words

    text = normalize("the cat sat on the mat. 数据 与 模型。")
    words = normalize_words(pre_tokenize(text, "unicode"), False)
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), word_counts(words))
    cfg = {"seed": 1, "metrics": {"speed_repeats": 1, "speed_sample_words": 5}}
    metrics = measure(
        tokenizer,
        words,
        words,
        bench_chars=len(text),
        hard_chars=len(text),
        cfg=cfg,
        tag="t",
    )
    assert metrics.n_chars == len(text)
    assert metrics.n_chars != sum(len(w) for w in words), "这两者必须不同，否则测不到口径"
    assert abs(metrics.tokens_per_char - metrics.n_tokens / len(text)) < 1e-12


def test_measure_coverage_is_word_level(counts: dict[str, int]) -> None:
    """覆盖率是**词级**：一个 UNK 都不产生的词才算被覆盖。"""
    from tokenizerfactory.corpus import pre_tokenize, word_counts
    from tokenizerfactory.normalize import normalize, normalize_words

    text = normalize("the cat sat on the mat")
    words = normalize_words(pre_tokenize(text, "unicode"), False)
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), word_counts(words))
    cfg = {"seed": 1, "metrics": {"speed_repeats": 1, "speed_sample_words": 5}}
    perfect = measure(
        tokenizer, words, words, bench_chars=len(text), hard_chars=len(text), cfg=cfg, tag="a"
    )
    assert perfect.vocab_coverage == 1.0

    broken = list(words) + ["龘龘龘"]
    worse = measure(
        tokenizer,
        broken,
        broken,
        bench_chars=len(text),
        hard_chars=len(text),
        cfg=cfg,
        tag="b",
    )
    assert worse.vocab_coverage < 1.0


def test_measure_speed_returns_positive_rates(counts: dict[str, int]) -> None:
    from tokenizerfactory.corpus import pre_tokenize, word_counts
    from tokenizerfactory.normalize import normalize, normalize_words

    words = normalize_words(pre_tokenize(normalize("the cat sat on the mat"), "unicode"), False)
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), word_counts(words))
    enc, dec = measure_speed(tokenizer, words, repeats=3, seed=1, tag="t")
    assert enc > 0 and dec > 0


def test_measure_speed_handles_empty_input(counts: dict[str, int]) -> None:
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), counts)
    assert measure_speed(tokenizer, [], repeats=2, seed=1, tag="t") == (0.0, 0.0)


def test_metric_line_is_single_line(counts: dict[str, int]) -> None:
    tokenizer = train_tokenizer(TokenizerSpec("bpe", 160), counts)
    run = make_run("bpe", 160, 0.5)
    line = metric_line(run)
    assert "\n" not in line and "bpe-v160" in line


# ----------------------------------------------------------------------
# 支配与前沿
# ----------------------------------------------------------------------
def test_to_min_form_flips_max_directions() -> None:
    point = {"tokens_per_char": 0.5, "encode_tokens_per_sec": 100.0}
    assert to_min_form(point, ["tokens_per_char", "encode_tokens_per_sec"]) == (0.5, -100.0)


def test_to_min_form_rejects_unknown_metric() -> None:
    """指标名写错必须炸掉。

    静默地把未知指标当成 min 方向，会让「配置里打错一个字」
    变成一个看起来正常、实际上方向反了的前沿。
    """
    with pytest.raises(KeyError):
        to_min_form({"tokens_per_char": 0.5}, ["tokens_per_chars"])


def test_dominance_basics() -> None:
    points = {
        "tiny_good": pt(64, 0.50),  # 词表最小、压缩最好 → 支配所有人
        "big_bad": pt(256, 0.90),  # 词表最大、压缩最差 → 被三个人支配
        "tiny_bad": pt(64, 0.90),  # 词表和 tiny_good 一样小，但压缩差
        "big_good": pt(256, 0.40),  # 词表最大，但压缩最好 → 不被支配
    }
    ids = sorted(points)
    # 注意 tiny_bad 也支配 big_bad：压缩一样，词表小得多 ——
    # 「支配」不要求某一维严格更好，只要不全差且至少一维更好。
    assert dominators("big_bad", ids, points, AXES) == ["big_good", "tiny_bad", "tiny_good"]
    assert non_dominated(ids, points, AXES) == ["big_good", "tiny_good"]
    assert dominated_count("tiny_good", ids, points, AXES) == 2
    assert dominated_count("big_bad", ids, points, AXES) == 0

    # 完全相同的两个点互相不支配（「至少一维严格更好」不成立）
    same = {"a": pt(64, 0.5), "e": pt(64, 0.5)}
    assert set(non_dominated(["a", "e"], same, AXES)) == {"a", "e"}


def test_dominance_respects_direction() -> None:
    """``max`` 方向的指标取负后参与比较。"""
    points = {
        "fast_bad": {"tokens_per_char": 1.0, "encode_tokens_per_sec": 1000.0},
        "slow_good": {"tokens_per_char": 0.5, "encode_tokens_per_sec": 100.0},
    }
    front = non_dominated(
        ["fast_bad", "slow_good"], points, ["tokens_per_char", "encode_tokens_per_sec"]
    )
    assert set(front) == {"fast_bad", "slow_good"}


def test_dominated_count_matches_dominators() -> None:
    points = {
        "a": pt(64, 0.3),
        "b": pt(128, 0.6),
        "c": pt(192, 0.9),
    }
    ids = sorted(points)
    assert dominated_count("a", ids, points, AXES) == 2
    assert dominated_count("c", ids, points, AXES) == 0
    assert dominators("c", ids, points, AXES) == ["a", "b"]


# ----------------------------------------------------------------------
# 超体积
# ----------------------------------------------------------------------
def test_hypervolume_of_a_single_point_is_its_box() -> None:
    assert hypervolume_2d([(2.0, 3.0)], (10.0, 10.0)) == pytest.approx(8 * 7)


def test_hypervolume_staircase_matches_manual_union() -> None:
    """两个点的并集面积：手算 66.0。

    这两个箱体是 [1,10]×[5,10] 与 [3,10]×[2,10]，并集面积 = 45 + 21 = 66。
    """
    assert hypervolume_2d([(1.0, 5.0), (3.0, 2.0)], (10.0, 10.0)) == pytest.approx(66.0)


def test_hypervolume_ignores_dominated_points() -> None:
    """传进来被支配的点不能让超体积变大。

    阶梯求和若直接信任输入，多算面积而**不报错** —— 所以函数自己会重算
    一遍非支配集。
    """
    clean = hypervolume_2d([(1.0, 5.0), (3.0, 2.0)], (10.0, 10.0))
    dirty = hypervolume_2d([(1.0, 5.0), (3.0, 2.0), (5.0, 8.0)], (10.0, 10.0))
    assert clean == pytest.approx(dirty)


def test_hv_contributions_sum_to_total() -> None:
    """每个前沿点的增量之和必须等于总超体积。"""
    points = {
        "a": pt(64, 0.9),
        "b": pt(128, 0.6),
        "c": pt(192, 0.3),
    }
    ids = sorted(points)
    contributions = hv_contributions(ids, points, AXES)
    total = hypervolume_of(ids, points, AXES)
    assert sum(contributions.values()) == pytest.approx(total)
    # 三个点在同一条权衡线上，各自都该有非零贡献
    assert all(v > 0 for v in contributions.values())


def test_hv_contributions_are_non_negative() -> None:
    points = {
        "a": pt(64, 0.9),
        "b": pt(128, 0.6),
        "c": pt(192, 0.3),
    }
    for value in hv_contributions(sorted(points), points, AXES).values():
        assert value >= 0.0


def test_reference_point_is_worst_corner_times_margin() -> None:
    points = {"a": pt(64, 0.5), "b": pt(192, 0.2)}
    ref = reference_point(sorted(points), points, AXES)
    assert ref == pytest.approx((192.0 * (1 + REFERENCE_MARGIN), 0.5 * (1 + REFERENCE_MARGIN)))


def test_reference_point_uses_all_candidates_not_just_the_frontier() -> None:
    """参考点取**全部候选点**的最差值，不取前沿点的。

    这样超体积才能在「同一候选集的不同前沿子集」之间比较 —— 而那正是它的用途。
    若参考点跟着前沿变，两个子集的超体积就不可比了。
    """
    points = {
        "a": pt(64, 0.5),
        "b": pt(192, 0.2),
        "bad": pt(1024, 5.0),  # 被支配，但它是候选集的一员
    }
    subset = reference_point(["a", "b"], points, AXES)
    everything = reference_point(sorted(points), points, AXES)
    assert subset != everything
    assert everything[0] > subset[0]


def test_adding_candidates_cannot_shrink_hypervolume() -> None:
    """候选集变大，超体积不会变小 —— 一个必须成立的性质。

    不成立就说明参考点算错了（例如跟着前沿走）。
    """
    base = {
        "a": pt(64, 0.5),
        "b": pt(192, 0.2),
    }
    smaller = hypervolume_of(sorted(base), base, AXES)
    extended = dict(base)
    extended["c"] = pt(128, 0.1)  # 明确支配 b 的点
    larger = hypervolume_of(sorted(extended), extended, AXES)
    assert larger >= smaller


def test_hypervolume_of_empty_set_is_zero() -> None:
    assert hypervolume_of([], {}, AXES) == 0.0


# ----------------------------------------------------------------------
# 拐点
# ----------------------------------------------------------------------
def test_knee_point_prefers_the_balanced_point() -> None:
    """拐点应当在「两头都不极端」的位置上。

    两个轴都是 min 方向，所以「词表小」与「压缩好」是一对真实取舍：
    词表越小，能装的子词越少，压缩越差。
    """
    points = {
        "small_vocab": pt(64, 4.0),  # 词表最省，但压缩最差
        "balanced": pt(128, 1.0),
        "big_vocab": pt(192, 0.0),  # 压缩最好，但词表最大
    }
    assert knee_point(sorted(points), points, AXES) == "balanced"


def test_knee_point_is_invariant_to_axis_rescaling() -> None:
    """把一个轴整体缩放不改变拐点。

    词表规模（10²~10⁴）与压缩率（0.3~1.0）量级差三四个数量级，
    若实现里混进任何一处「按原始量算距离」，换一份词表规模差十倍的语料
    拐点就会翻面。

    这里顺带把 ``knee_point`` 里那次归一化的作用**说准**：
    逐轴仿射缩放不会改变「点到直线距离」的排序（两个距离只差一个常数因子），
    所以归一化改变的不是结果，而是中间量的量级与数值条件；
    真正依赖归一化的是 ``start`` / ``end`` 的选取 —— 它按 ``(x, y)``
    做字典序比较，原始量下量级大的那个轴会单独决定选谁。
    """
    base = {
        "small_vocab": pt(64, 4.0),
        "balanced": pt(128, 1.0),
        "big_vocab": pt(192, 0.0),
    }
    scaled = {
        "small_vocab": pt(64_000, 4.0),
        "balanced": pt(128_000, 1.0),
        "big_vocab": pt(192_000, 0.0),
    }
    assert knee_point(sorted(base), base, AXES) == "balanced"
    assert knee_point(sorted(scaled), scaled, AXES) == "balanced"


def test_knee_point_handles_extreme_scale_ratios() -> None:
    """词表规模比压缩率大六个数量级时也要选对。"""
    points = {
        "small_vocab": pt(100, 1.0),  # 词表最小
        "balanced": pt(500_000, 0.4),
        "best_compression": pt(1_000_000, 0.0),  # 压缩最好
    }
    assert knee_point(sorted(points), points, AXES) == "balanced"


def test_knee_point_handles_degenerate_inputs() -> None:
    assert knee_point([], {}, AXES) is None
    one = {"a": pt(64, 1.0)}
    assert knee_point(["a"], one, AXES) == "a"
    two = {"a": pt(64, 1.0), "b": pt(128, 0.5)}
    assert knee_point(["a", "b"], two, AXES) in {"a", "b"}


def test_knee_point_handles_constant_axis() -> None:
    """一个轴完全不变时不能除零，也不能抛异常。"""
    flat = {
        "a": pt(2048, 0.9),
        "b": pt(2048, 0.6),
        "c": pt(2048, 0.3),
    }
    assert knee_point(sorted(flat), flat, AXES) in set(flat)


# ----------------------------------------------------------------------
# 组装与选型
# ----------------------------------------------------------------------
def _fake_result(runs: list[TokenizerRun], cfg: dict):
    from tokenizerfactory.schema import CorpusStats, TokenizerResult

    return TokenizerResult(
        corpus=CorpusStats(name="x", n_chars=1, n_words=1, n_unique_words=1),
        runs=runs,
        cfg=cfg,
    )


MINIMAL_CFG = {
    "seed": 1,
    "metrics": {"speed_repeats": 1, "speed_sample_words": 5, "sample_words": 2},
    "pareto": {
        "objectives": ["vocab_size", "tokens_per_char"],
        "extra_objectives": [["tokens_per_char", "encode_tokens_per_sec"]],
    },
    "select": {"rule": "max_compression_within_budget", "vocab_budget": 100},
}


def test_frontier_key_round_trips() -> None:
    assert frontier_key(["vocab_size", "tokens_per_char"]) == "vocab_size+tokens_per_char"
    assert frontier_pairs(MINIMAL_CFG)[0] == ["vocab_size", "tokens_per_char"]


def test_frontier_pairs_dedupes() -> None:
    cfg = {
        "pareto": {
            "objectives": ["a", "b"],
            "extra_objectives": [["a", "b"], ["b", "a"]],
        }
    }
    assert frontier_pairs(cfg) == [["a", "b"], ["b", "a"]]


def test_assemble_fills_matrix_frontiers_and_selection() -> None:
    runs = [
        make_run("bpe", 64, 0.60),
        make_run("bpe", 128, 0.50),
        make_run("wordpiece", 64, 0.90),
        make_run("wordpiece", 128, 0.70),
        # 同预算（64）、同词表规模，但压缩比 bpe 更好 —— 这一行就是
        # 「选错算法的代价」在最小规模上的样子。
        make_run("unigram", 64, 0.55),
    ]
    result = _fake_result(runs, MINIMAL_CFG)
    assemble(result, MINIMAL_CFG)

    assert len(result.matrix) == 5
    assert set(result.frontiers) == {
        "vocab_size+tokens_per_char",
        "tokens_per_char+encode_tokens_per_sec",
    }
    main = result.frontiers["vocab_size+tokens_per_char"]
    # unigram-v64 支配了 bpe-v64 / wordpiece-v64 / wordpiece-v128；
    # bpe-v128 与它互不支配（词表大一倍换 0.05 的压缩），所以前沿是这两个。
    assert set(main) == {"unigram-v64", "bpe-v128"}
    assert result.counters["n_specs"] == 5

    # 预算 100：bpe-v128 / wordpiece-v128 被排除在候选之外
    assert result.selected == "unigram-v64"

    baseline = result.counters["baseline"]
    # 基线 = 不比较就直接上的默认选择（bpe + 最小预算 + unicode）
    assert baseline["spec_id"] == "bpe-v64"
    # 它在最小词表那一端，但同预算下的 unigram 压缩更好 —— 它被支配了
    assert baseline["dominated_by"] == 1
    assert baseline["dominated_by_sample"] == ["unigram-v64"]
    assert baseline["best_at_same_vocab"]["spec_id"] == "unigram-v64"
    assert baseline["best_at_same_vocab"]["gain"] == pytest.approx(0.05)
    # 入选者与基线同预算，所以这个差值里**没有**混进「多花的预算」
    assert baseline["selected_vs_baseline"]["spec_id"] == "unigram-v64"
    assert baseline["selected_vs_baseline"]["same_budget"] is True
    assert baseline["selected_vs_baseline"]["gain"] == pytest.approx(0.05)


def test_baseline_marks_when_the_selection_costs_more_budget() -> None:
    """预算放大到能装下 v128 时，入选者换了，差值里就混进了「多花的预算」。

    这个标记必须出现 —— 否则「比基线好 0.1」会被读成纯粹的选型收益。
    """
    cfg = dict(MINIMAL_CFG)
    cfg["select"] = {"rule": "max_compression_within_budget", "vocab_budget": 128}
    runs = [make_run("bpe", 64, 0.60), make_run("bpe", 128, 0.50)]
    result = _fake_result(runs, cfg)
    assemble(result, cfg)

    assert result.selected == "bpe-v128"
    payload = result.counters["baseline"]["selected_vs_baseline"]
    assert payload["same_budget"] is False
    assert "混着" in payload["note"]


def test_matrix_rows_count_matches_runs() -> None:
    runs = [make_run("bpe", 64, 0.6), make_run("bpe", 128, 0.5)]
    result = _fake_result(runs, MINIMAL_CFG)
    assemble(result, MINIMAL_CFG)
    assert len(build_matrix(result, MINIMAL_CFG)) == len(runs)


def test_select_knee_rule_uses_the_main_frontier() -> None:
    cfg = dict(MINIMAL_CFG)
    cfg["select"] = {"rule": "knee", "vocab_budget": 128}
    runs = [
        make_run("bpe", 64, 1.0),
        make_run("bpe", 128, 0.5),
        make_run("bpe", 256, 0.0),
    ]
    result = _fake_result(runs, cfg)
    assemble(result, cfg)
    chosen, reason = select(result, cfg)
    assert chosen in {"bpe-v64", "bpe-v128", "bpe-v256"}
    assert "knee" in reason


def test_select_falls_back_when_budget_excludes_everything() -> None:
    cfg = dict(MINIMAL_CFG)
    cfg["select"] = {"rule": "max_compression_within_budget", "vocab_budget": 1}
    runs = [make_run("bpe", 64, 0.6), make_run("bpe", 128, 0.5)]
    result = _fake_result(runs, cfg)
    assemble(result, cfg)
    _, reason = select(result, cfg)
    assert "退化" in reason
    assert result.selected == "bpe-v128"


def test_select_breaks_ties_by_smaller_vocab_then_id() -> None:
    cfg = dict(MINIMAL_CFG)
    runs = [
        make_run("bpe", 128, 0.5),
        make_run("bpe", 64, 0.5),
    ]
    result = _fake_result(runs, cfg)
    assemble(result, cfg)
    assert result.selected == "bpe-v64"


def test_select_on_empty_result() -> None:
    result = _fake_result([], MINIMAL_CFG)
    assemble(result, MINIMAL_CFG)
    assert result.selected == ""
    assert "没有可选" in result.counters["selected_reason"]


# ----------------------------------------------------------------------
# 真实结果上的整体一致性
# ----------------------------------------------------------------------
def test_assemble_on_real_pipeline_result(tiny_result) -> None:
    result, _ = tiny_result
    points = {r.spec_id: point_of(r) for r in result.runs}
    ids = sorted(points)
    assert len(result.matrix) == len(result.runs)
    for key, members in result.frontiers.items():
        objectives = key.split("+")
        assert members == non_dominated(ids, points, objectives), key
        assert members, f"{key} 的前沿不该为空"
    # 矩阵里标注的前沿必须与 frontiers 一致
    for row in result.matrix:
        expected = sorted(
            key for key, members in result.frontiers.items() if row["spec_id"] in members
        )
        assert row["frontier"] == expected
    assert result.selected in ids
    assert result.counters["baseline"]["spec_id"] in ids


def test_hypervolume_matches_contribution_sum_on_real_result(tiny_result) -> None:
    result, _ = tiny_result
    points = {r.spec_id: point_of(r) for r in result.runs}
    for key, members in result.frontiers.items():
        objectives = key.split("+")
        contributions = hv_contributions(sorted(points), points, objectives)
        total = hypervolume_of(members, points, objectives)
        assert sum(contributions.get(m, 0.0) for m in members) == pytest.approx(total)


def test_counters_record_hypervolume_and_reference_for_every_pair(tiny_result) -> None:
    result, _ = tiny_result
    assert set(result.counters["hypervolume"]) == set(result.frontiers)
    assert set(result.counters["reference_point"]) == set(result.frontiers)
    assert set(result.counters["best_by"]) >= {
        "tokens_per_char",
        "tokens_per_word",
        "vocab_coverage",
        "unknown_rate",
        "encode_tokens_per_sec",
    }
