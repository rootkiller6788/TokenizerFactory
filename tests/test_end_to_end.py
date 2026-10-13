"""端到端：编排层 → 落盘 → 报告 → CLI。

这一层测的不是某个算法，而是**整条链的契约**：

1. **扇形展开再收拢**。一个语料分叉出规则、规则分叉出 spec，最后只有前沿上的
   那几个收拢到 ``sel`` / ``art``。收拢点的输入集合必须**恰好**是前沿候选 ——
   这是本项目 DAG 与前几个项目最不一样的地方，也是「一眼看出谁进了前沿」的
   唯一依据。
2. **只读子命令真的只读**。``--out`` 只属于 ``run``；其余命令跑完，
   产物目录必须**一个字节都没变**。观察者会改状态是最容易把人绕晕的一类设计。
3. **导出的 artifact 真的能用**。载入它重新编码，结果必须与内存里的分词器
   逐位相同 —— 否则 ``encode`` 这条命令就只是「读了一个 JSON 然后假装能用」。
4. **同一份配置跑两次结果必须一致**。训练里有随机性（Unigram 的剪枝并列打破、
   速度基准的顺序打乱），所以这条不是免费的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizerfactory import cli, report
from tokenizerfactory.compare import frontier_key, frontier_pairs
from tokenizerfactory.lineage import (
    CORPUS_NODE_ID,
    artifact_node_id,
    bench_node_id,
    norm_node_id,
    select_node_id,
    train_node_id,
)
from tokenizerfactory.metrics import (
    METRIC_DIRECTIONS,
    TIMING_METRICS,
    has_timing_axis,
    point_of,
)
from tokenizerfactory.pipeline import (
    SELECT_TAG,
    CorpusBundle,
    expand_specs,
    run_pipeline,
    write_outputs,
)
from tokenizerfactory.tokenizers import load_tokenizer


# ----------------------------------------------------------------------
# 设计空间展开
# ----------------------------------------------------------------------
def test_expand_specs_covers_the_cross_product(tiny_cfg) -> None:
    specs = expand_specs(tiny_cfg)
    s = tiny_cfg["specs"]
    expected = (
        len(s["algorithms"]) * len(s["vocab_sizes"]) * len(s["pre_tokenizers"])
    ) * (2 if s["lowercase_variants"] else 1)
    assert len(specs) == expected
    assert len({sp.spec_id for sp in specs}) == expected, "spec_id 必须唯一"


def test_expand_specs_marks_lowercase_variants(tiny_cfg) -> None:
    from tokenizerfactory.config import deep_merge

    cfg = deep_merge(tiny_cfg, {"specs": {"lowercase_variants": True}})
    specs = expand_specs(cfg)
    assert len(specs) == len(expand_specs(tiny_cfg)) * 2
    # 注意 ``+lower`` **不一定在末尾**：非 unicode 规则还会再挂一个 ``+规则``。
    # spec_id 的后缀顺序是 lowercase → pre_tokenizer，所以这里查「包含」。
    assert all("+lower" in sp.spec_id for sp in specs if sp.lowercase)
    assert not any("+lower" in sp.spec_id for sp in specs if not sp.lowercase)
    assert any(sp.spec_id.endswith("+lower+whitespace") for sp in specs)


def test_expand_specs_only_gives_unigram_its_own_params(tiny_cfg) -> None:
    """算法特有参数不能泄漏到别的算法上 —— 否则对比里混进了配置差异。"""
    for spec in expand_specs(tiny_cfg):
        if spec.algorithm == "unigram":
            assert spec.extra
        else:
            assert spec.extra == {}


# ----------------------------------------------------------------------
# 流水线
# ----------------------------------------------------------------------
def test_run_pipeline_produces_one_run_per_spec(tiny_result, tiny_cfg) -> None:
    result, tokenizers = tiny_result
    specs = expand_specs(tiny_cfg)
    assert len(result.runs) == len(specs)
    assert set(tokenizers) == {r.spec_id for r in result.runs}
    assert result.algorithms() == sorted(tiny_cfg["specs"]["algorithms"])


def test_every_run_stays_within_its_vocab_budget(tiny_result) -> None:
    """「同等词表预算下谁压得更狠」的前提：实际词表不能超过目标。"""
    result, _ = tiny_result
    for run in result.runs:
        assert 0 < len(run.vocab) <= run.spec.vocab_size, run.spec_id
        assert len(set(run.vocab)) == len(run.vocab), run.spec_id


def test_word_counts_are_shared_across_algorithms(tiny_cfg) -> None:
    """同一（规则, 小写）下三个算法必须拿到**同一份**词频。

    缓存是正确性的一部分：各算各的会让「同一份语料」这个前提在实现层面
    悄悄失效 —— 而它正是整个对比的前提。这里查的是对象同一性，
    不只是相等：相等但重算过，说明缓存没生效，某天两处的归一化漂移就会分家。
    """
    bundle = CorpusBundle(tiny_cfg)
    first = bundle.words_and_counts("unicode", False)
    again = bundle.words_and_counts("unicode", False)
    assert first is again

    other_rule = bundle.words_and_counts("whitespace", False)
    assert other_rule is not first


def test_corpus_bundle_uses_normalized_text_length_as_the_denominator(tiny_cfg) -> None:
    bundle = CorpusBundle(tiny_cfg)
    words, _ = bundle.words_and_counts("unicode", False)
    assert bundle.bench_chars == len(bundle.bench_text)
    assert bundle.bench_chars != sum(len(w) for w in words)


def test_run_pipeline_is_deterministic_on_everything_but_timing(tiny_cfg) -> None:
    """同一份配置跑两次：词表、摘要、非时间类指标、入选、版本必须逐位相同。

    **时间类指标除外**，而且这不是偷懒 —— 它是实测出来的性质：
    ``encode_tokens_per_sec`` 靠计时得到，两个速度接近的 spec 会在两次运行之间
    交换前沿成员。实测中 ``tokens_per_char+encode_tokens_per_sec`` 这条前沿的
    成员从 ``wordpiece-v128`` 变成了 ``wordpiece-v128+whitespace``，
    而另外三条前沿与全部词表摘要逐位相同。

    所以这里分两段断言：不含实测轴的部分要求严格相等，
    含实测轴的部分只要求「是一组合法的前沿点」。这条区别在
    ``metrics.TIMING_METRICS`` 与报告的「含实测轴」标注里是同一件事。
    """
    first, _ = run_pipeline(tiny_cfg)
    second, _ = run_pipeline(tiny_cfg)

    assert [r.spec_id for r in first.runs] == [r.spec_id for r in second.runs]
    for a, b in zip(first.runs, second.runs):
        assert a.vocab == b.vocab, a.spec_id
        assert a.vocab_digest == b.vocab_digest, a.spec_id
        # 用 point_of 而不是 metrics.get：``METRIC_DIRECTIONS`` 里有几个是
        # spec 派生量（vocab_size / vocab_size_actual），不在 TokenizerMetrics 上。
        pa, pb = point_of(a), point_of(b)
        for name in METRIC_DIRECTIONS:
            if name in TIMING_METRICS:
                continue
            assert pa[name] == pb[name], (a.spec_id, name)
    assert first.selected == second.selected
    assert first.version.version_id == second.version.version_id

    def canonical(rows: list[dict]) -> list[dict]:
        """抹掉实测字段，并只保留不含实测轴的前沿标注。"""
        out = []
        for row in rows:
            kept = {k: v for k, v in row.items() if k not in TIMING_METRICS}
            kept["frontier"] = sorted(
                key for key in row["frontier"] if not has_timing_axis(key.split("+"))
            )
            out.append(kept)
        return out

    assert canonical(first.matrix) == canonical(second.matrix)

    known = {r.spec_id for r in first.runs}
    for key, members in first.frontiers.items():
        assert members and set(members) <= known, key
        if not has_timing_axis(key.split("+")):
            assert members == second.frontiers[key], key


def test_version_digest_is_content_addressed(tiny_result) -> None:
    result, _ = tiny_result
    selected = result.find(result.selected)
    assert result.version is not None
    assert selected is not None
    assert result.version.tag == "tiny"
    assert result.version.version_id.startswith("tiny-")
    # 摘要由入选词表决定：换一个词表就必须换一个版本
    other = next(r for r in result.runs if r.spec_id != selected.spec_id)
    assert other.vocab_digest != selected.vocab_digest


# ----------------------------------------------------------------------
# DAG：扇形展开再收拢
# ----------------------------------------------------------------------
def test_dag_fans_out_and_merges(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    dag = result.dag
    assert dag is not None

    n_specs = len(result.runs)
    n_norms = len(
        {(r.spec.pre_tokenizer, r.spec.lowercase) for r in result.runs}
    )
    assert len(dag.by_kind("corpus")) == 1
    assert len(dag.by_kind("normalize")) == n_norms
    assert len(dag.by_kind("train")) == n_specs
    assert len(dag.by_kind("bench")) == n_specs
    assert dag.by_kind("select") == [select_node_id(SELECT_TAG)]
    assert dag.by_kind("artifact") == [artifact_node_id(SELECT_TAG)]

    # 每个 norm 节点都必须真的分叉出至少一个 train —— 否则那个规则白配了
    for nid in dag.by_kind("normalize"):
        assert dag.children(nid), nid


def test_every_spec_chain_is_a_real_path(tiny_result) -> None:
    """``corpus → norm:rule → train:spec → bench:spec`` 必须是图里的一条真路径。

    扁平列表与图一旦分家，``explain`` 会悄悄缺一块而且不报错 ——
    这条不变量在前几个项目里都救过场。
    """
    result, _ = tiny_result
    dag = result.dag
    for run in result.runs:
        bench = bench_node_id(run.spec_id)
        train = train_node_id(run.spec_id)
        assert bench in dag.nodes
        assert train in dag.nodes
        assert dag.parents(bench) == [train]

        norm = norm_node_id(run.spec.pre_tokenizer, run.spec.lowercase)
        assert dag.parents(train) == [norm]
        assert dag.parents(norm) == [CORPUS_NODE_ID]
        assert CORPUS_NODE_ID in dag.path_to_root(bench)


def test_select_node_inputs_are_exactly_the_frontier_candidates(tiny_result, tiny_cfg) -> None:
    """收拢点恰好接住全部前沿候选 —— 这是「一眼看出谁进了前沿」的依据。"""
    result, _ = tiny_result
    dag = result.dag
    key = frontier_key(tiny_cfg["pareto"]["objectives"])
    frontier = result.frontiers[key]

    sel = dag.nodes[select_node_id(SELECT_TAG)]
    expected = {bench_node_id(sid) for sid in frontier}
    expected.add(bench_node_id(result.selected))
    assert set(sel.inputs) == expected
    assert len(sel.inputs) > 1, "收拢点必须是真正的多输入节点"


def test_artifact_node_reads_the_selection(tiny_result) -> None:
    result, _ = tiny_result
    dag = result.dag
    art = dag.nodes[artifact_node_id(SELECT_TAG)]
    assert dag.parents(artifact_node_id(SELECT_TAG)) == [select_node_id(SELECT_TAG)]
    assert art.stats["spec_id"] == result.selected


def test_dag_is_acyclic_and_topological(tiny_result) -> None:
    result, _ = tiny_result
    order = result.dag.topological()
    assert len(order) == len(result.dag.nodes)
    position = {n: i for i, n in enumerate(order)}
    for a, b in result.dag.edges:
        assert position[a] < position[b], (a, b)


def test_frontier_header_has_no_duplicate_columns_for_any_metric_pair() -> None:
    """任意两个指标组合成目标对，表头都不得出现重复列名。

    这条测试是补出来的：小语料夹具的 ``extra_objectives`` 里没有
    ``vocab_size_actual``，所以「表头重复」这个 bug 只在**真实配置**上
    跑 ``frontier`` 命令时才露出来（出现了两个「实际词表」列）。
    穷举全部组合才能真正守住它。
    """
    from itertools import combinations

    for a, b in combinations(sorted(METRIC_DIRECTIONS), 2):
        header = report.frontier_header([a, b])
        assert len(header) == len(set(header)), (a, b, header)
        assert len(header) == len(report.frontier_header([b, a]))


def test_frontier_rows_match_the_header_for_any_metric_pair(tiny_result) -> None:
    """穷举组合也要保证行宽与表头一致 —— 表头与行是两处独立构造的，
    分家时不会有任何报错，只会让表格错位。"""
    from itertools import combinations

    result, _ = tiny_result
    for a, b in combinations(sorted(METRIC_DIRECTIONS), 2):
        key = f"{a}+{b}"
        header = report.frontier_header([a, b])
        result.frontiers[key] = sorted({r.spec_id for r in result.runs})
        try:
            rows = report.frontier_rows(result, key)
        finally:
            del result.frontiers[key]
        assert rows, key
        assert all(len(row) == len(header) for row in rows), key


def test_frontier_integer_axes_print_as_integers(tiny_result, tiny_cfg) -> None:
    """词表规模是整数计数，打印成 ``2048.0000`` 会让读者以为它是比率。"""
    result, _ = tiny_result
    key = frontier_key(tiny_cfg["pareto"]["objectives"])
    rows = report.frontier_rows(result, key)
    assert rows
    # 主目标对的第一轴是 vocab_size
    for row in rows:
        assert "." not in row[4], row
        assert "." in row[5], row


def test_selected_ancestry_dot_is_smaller_than_the_full_graph(tiny_result, tiny_cfg) -> None:
    """``--frontier-only`` 只画入选 spec 的祖先链 —— 3×N×M 个训练节点全画太密。"""
    result, _ = tiny_result
    dag = result.dag
    full = dag.to_dot()
    trimmed = dag.to_dot(only_selected_ancestry=True)
    assert trimmed.count("->") < full.count("->")
    assert "sel:main" in trimmed

    def node_ids(dot: str) -> set[str]:
        return {
            line.strip().split('"')[1]
            for line in dot.splitlines()
            if line.strip().startswith('"') and "[label=" in line
        }

    kept = node_ids(trimmed)
    main = frontier_key(tiny_cfg["pareto"]["objectives"])
    must_keep = {bench_node_id(s) for s in result.frontiers[main]}
    must_keep.add(bench_node_id(result.selected))
    assert must_keep <= kept

    dropped = [
        r.spec_id
        for r in result.runs
        if r.spec_id != result.selected and r.spec_id not in result.frontiers[main]
    ]
    assert dropped, "这份配置下应当有被淘汰的 spec，否则测不到裁剪"
    assert all(bench_node_id(sid) not in kept for sid in dropped)


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
EXPECTED_OUTPUTS = (
    "result.json",
    "matrix.csv",
    "frontier.csv",
    "runs.jsonl",
    "diffs.json",
    "lineage.json",
    "lineage.dot",
    "lineage_selected.dot",
    "version.json",
    "report.md",
    "artifacts",
)


def test_write_outputs_writes_everything(tiny_out, tiny_result) -> None:
    result, _ = tiny_result
    for name in EXPECTED_OUTPUTS:
        assert (tiny_out / name).exists(), name
    assert len(list((tiny_out / "artifacts").glob("*.json"))) == len(result.runs)


def test_result_json_omits_samples_but_runs_jsonl_keeps_them(tiny_out, tiny_result) -> None:
    """``result.json`` 刻意不存 ``sample``（会撑大文件），
    但 ``runs.jsonl`` 必须有 —— ``explain`` 要展示切分，少了它只会静默地不显示。"""
    result, _ = tiny_result
    payload = json.loads((tiny_out / "result.json").read_text(encoding="utf-8"))
    assert payload["runs"] and all("sample" not in run for run in payload["runs"])

    lines = [
        json.loads(line)
        for line in (tiny_out / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == len(result.runs)
    assert all("sample" in run for run in lines)


def test_matrix_csv_rows_match_the_matrix(tiny_out, tiny_result) -> None:
    import csv

    result, _ = tiny_result
    with (tiny_out / "matrix.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(result.matrix)
    assert {row["spec_id"] for row in rows} == {r.spec_id for r in result.runs}
    # 「前沿」是列表，必须被序列化成 JSON 而不是被丢掉
    assert all(row["frontier"].startswith("[") for row in rows)


def test_frontier_csv_covers_every_objective_pair(tiny_out, tiny_result) -> None:
    import csv

    result, _ = tiny_result
    with (tiny_out / "frontier.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["frontier"] for row in rows} == set(result.frontiers)
    for row in rows:
        assert row["spec_id"] in result.frontiers[row["frontier"]]
        # 目标对的每个指标都必须是列（_write_csv 合并了所有行的键）
        for name in row["frontier"].split("+"):
            assert name in row


def test_artifacts_are_loadable_and_reproduce_the_encoding(tiny_out, tiny_result, tiny_cfg) -> None:
    """导出的分词器必须**能用**：载入后编码结果与内存里的逐位相同。"""
    result, tokenizers = tiny_result
    bundle = CorpusBundle(tiny_cfg)
    for run in result.runs:
        payload = json.loads(
            (tiny_out / "artifacts" / f"{run.spec_id}.json").read_text(encoding="utf-8")
        )
        reloaded = load_tokenizer(payload)
        assert reloaded.vocab == run.vocab
        assert reloaded.vocab_digest == run.vocab_digest

        words = bundle.bench_words(run.spec.pre_tokenizer, run.spec.lowercase)
        original = tokenizers[run.spec_id].encode(words)
        assert reloaded.encode(words).tokens == original.tokens, run.spec_id
        assert reloaded.encode(words).ids == original.ids, run.spec_id


def test_lineage_json_round_trips(tiny_out) -> None:
    payload = json.loads((tiny_out / "lineage.json").read_text(encoding="utf-8"))
    assert payload["summary"]["nodes"] == len(payload["nodes"])
    assert payload["summary"]["edges"] == len(payload["edges"])
    ids = {node["node_id"] for node in payload["nodes"]}
    for edge in payload["edges"]:
        assert edge["from"] in ids and edge["to"] in ids


def test_dag_rebuild_from_json_is_equivalent(tiny_result, tiny_out) -> None:
    """重建的图必须与原图**逐项等价**。

    CLI 的 ``dag`` 命令靠这次重建来支持 ``--dot``；若重建出来的图与原来的不同，
    ``--dot`` 会画出另一张图而没有任何报错。最容易出错的一处是节点的
    ``outputs``：照抄一遍会被 ``add_node`` 再追加一次。
    """
    from tokenizerfactory.lineage import LineageDAG

    original = tiny_result[0].dag
    payload = json.loads((tiny_out / "lineage.json").read_text(encoding="utf-8"))
    rebuilt = LineageDAG.from_dict(payload)
    assert rebuilt.to_dict() == payload
    assert rebuilt.summary() == original.summary()
    assert rebuilt.to_dot() == original.to_dot()
    assert rebuilt.merge_points() == original.merge_points()
    assert rebuilt.topological() == original.topological()


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def test_matrix_rows_match_the_header_width(tiny_result) -> None:
    result, _ = tiny_result
    rows = report.matrix_rows(result)
    assert len(rows) == len(result.runs)
    assert all(len(row) == len(report.MATRIX_HEADER) for row in rows)


def test_frontier_rows_match_the_header_width(tiny_result, tiny_cfg) -> None:
    result, _ = tiny_result
    for pair in frontier_pairs(tiny_cfg):
        key = frontier_key(pair)
        rows = report.frontier_rows(result, key)
        assert len(rows) == len(result.frontiers[key])
        assert all(len(row) == len(report.frontier_header(pair)) for row in rows)


def test_frontier_header_has_no_duplicate_columns(tiny_cfg) -> None:
    """``objectives[0]`` 就是 ``vocab_size`` 时，标签本身就是「词表目标」——
    再加一列会造出两个同名表头。"""
    for pair in frontier_pairs(tiny_cfg):
        header = report.frontier_header(pair)
        assert len(header) == len(set(header)), header


def test_budget_sensitivity_has_one_row_per_vocab_size(tiny_result) -> None:
    result, _ = tiny_result
    rows = report.budget_sensitivity_rows(result)
    assert [row[0] for row in rows] == [str(s) for s in result.vocab_sizes()]
    # 词表预算 + 每个算法一格 + （赢家, 赢家压缩, 算法差距, 规则差距）
    width = 1 + len(result.algorithms()) + 4
    assert all(len(row) == width for row in rows)
    # 两条差距列必须分开 —— 合并成一列会得到不属于任何决策轴的数字
    for row in rows:
        assert row[-2].endswith("%") or row[-2] == "-"
        assert row[-1].endswith("%") or row[-1] == "-"


def test_budget_sensitivity_isolates_one_variable_at_a_time(tiny_result) -> None:
    """算法差距固定规则、规则差距固定算法。

    合并成「该档最大差距」会让 `char` 这个刻意做的粒度极端值决定数字，
    于是报告里出现「预算越大差距越大」与「预算越小选错的代价越大」两句矛盾的话。
    """
    result, _ = tiny_result
    for row in report.budget_sensitivity_rows(result):
        size = int(row[0])
        winner = result.find(row[len(result.algorithms()) + 1])
        assert winner.spec.vocab_size == size
        at_size = [r for r in result.runs if r.spec.vocab_size == size]

        same_rule = [
            r for r in at_size if r.spec.pre_tokenizer == winner.spec.pre_tokenizer
        ]
        worst_same_rule = max(same_rule, key=lambda r: r.metrics.tokens_per_char)
        expected = (
            worst_same_rule.metrics.tokens_per_char / winner.metrics.tokens_per_char - 1.0
        )
        assert row[-2] == f"{expected:+.1%}"

        same_algorithm = [r for r in at_size if r.algorithm == winner.spec.algorithm]
        worst_same_algorithm = max(
            same_algorithm, key=lambda r: r.metrics.tokens_per_char
        )
        expected = (
            worst_same_algorithm.metrics.tokens_per_char
            / winner.metrics.tokens_per_char
            - 1.0
        )
        assert row[-1] == f"{expected:+.1%}"


def test_pretokenizer_rows_cover_every_rule(tiny_result) -> None:
    result, _ = tiny_result
    rows = report.pretokenizer_rows(result)
    assert {row[0] for row in rows} == {r.spec.pre_tokenizer for r in result.runs}


def test_base_vocab_rows_show_wordpiece_costs_twice(tiny_result) -> None:
    result, _ = tiny_result
    rows = {row[0]: row for row in report.base_vocab_rows(result)}
    assert rows, "读不到语料时这张表会静默为空，测试必须发现这一点"
    alphabet = int(rows["bpe"][2])
    assert int(rows["bpe"][1]) == 1 + alphabet
    assert int(rows["wordpiece"][1]) == 1 + 2 * alphabet
    assert int(rows["unigram"][1]) == 1 + alphabet


def test_diff_rows_width_matches_header(tiny_result) -> None:
    result, _ = tiny_result
    spec_ids = list(result.diffs["specs"])
    rows = report.diff_rows(result)
    assert len(rows) == len(result.diffs["table"])
    assert all(len(row) == len(report.diff_header(spec_ids)) for row in rows)


def test_worst_alternative_holds_budget_and_rule_fixed(tiny_result) -> None:
    """「选错算法的代价」必须同预算 **且** 同预分词规则 —— 一次只动一个变量。"""
    result, _ = tiny_result
    spread = report.worst_alternative_at_same_budget(result)
    assert spread, "这份配置下应当算得出这个数字"
    selected = result.find(result.selected)
    worst = result.find(spread["worst_spec"])
    assert worst.spec.vocab_size == selected.spec.vocab_size == spread["budget"]
    assert worst.spec.pre_tokenizer == selected.spec.pre_tokenizer
    assert worst.metrics.tokens_per_char >= selected.metrics.tokens_per_char
    assert spread["ratio"] == pytest.approx(
        worst.metrics.tokens_per_char / selected.metrics.tokens_per_char - 1.0
    )


def test_summary_line_is_a_single_line(tiny_result) -> None:
    result, _ = tiny_result
    line = report.summary_line(result)
    assert "\n" not in line
    assert result.selected in line


def test_markdown_table_handles_empty_rows() -> None:
    assert report.markdown_table(("a", "b"), []) == "（空）"


def test_render_markdown_has_all_sections(tiny_result) -> None:
    result, _ = tiny_result
    text = report.render_markdown(result)
    for section in (
        "## 一、结论",
        "## 二、语料画像",
        "## 三、对比矩阵",
        "## 四、Pareto 前沿",
        "## 五、Tokenizer Diff",
        "## 六、每个维度上谁最强",
        "## 七、血缘与产物",
    ):
        assert section in text, section
    assert result.selected in text
    assert result.version.version_id in text


def test_report_keeps_the_coverage_caveat(tiny_result) -> None:
    """「覆盖率跨预分词规则不可比」这条口径限制必须留在报告里。

    删掉它，矩阵里那一列就会被读成「char 的词表更好」—— 一个错误结论，
    而且不会有人发现。
    """
    text = report.render_markdown(tiny_result[0])
    assert "覆盖率」跨预分词规则不可比" in text


def test_explain_known_spec_lists_facts(tiny_result) -> None:
    result, _ = tiny_result
    sid = result.selected
    text = report.explain(result, sid)
    run = result.find(sid)
    assert f"spec: {sid}" in text
    assert run.spec.algorithm in text
    assert "度量：" in text
    assert "词表片段" in text
    assert "切分样本" in text


def test_explain_reports_a_spec_that_is_not_on_any_frontier(tiny_result) -> None:
    result, _ = tiny_result
    on_frontier = {
        sid for members in result.frontiers.values() for sid in members
    }
    outside = [r.spec_id for r in result.runs if r.spec_id not in on_frontier]
    if not outside:
        pytest.skip("这份配置下每个 spec 都在某条前沿上")
    text = report.explain(result, outside[0])
    assert "不在任何目标对的前沿上" in text
    assert "被" in text


def test_explain_unknown_spec_says_so(tiny_result) -> None:
    assert "未知 spec_id" in report.explain(tiny_result[0], "nope-v1")


def test_meta_payload_is_json_serialisable(tiny_result) -> None:
    payload = report.meta_payload(tiny_result[0])
    json.dumps(payload, ensure_ascii=False)
    assert payload["selected"] == tiny_result[0].selected
    assert set(payload["frontiers"]) == set(tiny_result[0].frontiers)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_cli_run_then_readonly_commands(tiny_config_path, tmp_path, capsys) -> None:
    out = tmp_path / "cli-out"
    assert cli.main(["run", "--config", str(tiny_config_path), "--out", str(out), "--quiet"]) == 0
    capsys.readouterr()
    for name in EXPECTED_OUTPUTS:
        assert (out / name).exists(), name

    assert cli.main(["compare", "--out", str(out)]) == 0
    assert "spec" in capsys.readouterr().out
    assert cli.main(["list", "--out", str(out)]) == 0
    assert cli.main(["frontier", "--out", str(out)]) == 0
    assert "超体积" in capsys.readouterr().out
    assert cli.main(["diff", "--out", str(out)]) == 0
    assert "边界一致度" in capsys.readouterr().out
    assert cli.main(["dag", "--out", str(out)]) == 0
    assert cli.main(["dag", "--out", str(out), "--dot"]) == 0
    assert "digraph" in capsys.readouterr().out


def test_cli_readonly_commands_do_not_touch_the_outputs(tiny_out) -> None:
    """观察者不能改状态 —— ``--out`` 只属于 ``run``。"""
    before = sorted(p.relative_to(tiny_out) for p in tiny_out.rglob("*"))
    for argv in (
        ["compare", "--out", str(tiny_out)],
        ["list", "--out", str(tiny_out)],
        ["frontier", "--out", str(tiny_out)],
        ["diff", "--out", str(tiny_out)],
        ["dag", "--out", str(tiny_out)],
    ):
        assert cli.main(argv) == 0
    after = sorted(p.relative_to(tiny_out) for p in tiny_out.rglob("*"))
    assert before == after


def test_cli_frontier_can_filter_a_pair(tiny_out, tiny_cfg, capsys) -> None:
    key = frontier_key(tiny_cfg["pareto"]["objectives"])
    assert cli.main(["frontier", "--out", str(tiny_out), "--pair", key]) == 0
    out = capsys.readouterr().out
    assert f"## {key}" in out
    # 只该打印这一个目标对
    for pair in frontier_pairs(tiny_cfg)[1:]:
        assert f"## {frontier_key(pair)}" not in out


def test_cli_frontier_rejects_an_unknown_pair(tiny_out, capsys) -> None:
    assert cli.main(["frontier", "--out", str(tiny_out), "--pair", "a+b"]) == 1
    assert "未知目标对" in capsys.readouterr().err


def test_cli_explain_unknown_spec_returns_one_with_suggestions(tiny_out, capsys) -> None:
    assert cli.main(["explain", "bpe-v9999", "--out", str(tiny_out)]) == 1
    err = capsys.readouterr().err
    assert "未知 spec_id" in err
    assert "相近的可选项" in err


def test_cli_explain_known_spec_returns_zero(tiny_out, tiny_result, capsys) -> None:
    sid = tiny_result[0].selected
    assert cli.main(["explain", sid, "--out", str(tiny_out)]) == 0
    assert sid in capsys.readouterr().out


def test_cli_encode_uses_the_exported_artifact(tiny_out, tiny_result, capsys) -> None:
    """``encode`` 是「导出的产物真的能用」这条承诺的可执行检查。"""
    sid = tiny_result[0].selected
    assert cli.main(["encode", sid, "the cat sat on the mat", "--out", str(tiny_out)]) == 0
    out = capsys.readouterr().out
    assert f"spec: {sid}" in out
    assert "token 数：" in out
    assert "decode :" in out
    assert "the cat sat on the mat" in out


def test_cli_encode_unknown_spec_returns_one(tiny_out, capsys) -> None:
    assert cli.main(["encode", "nope-v1", "hello", "--out", str(tiny_out)]) == 1
    assert "找不到" in capsys.readouterr().err


def test_cli_missing_out_dir_returns_two(tmp_path, capsys) -> None:
    """缺产物目录不该抛栈，而应给一句能照着做的提示。"""
    missing = tmp_path / "nope"
    assert cli.main(["compare", "--out", str(missing)]) == 2
    assert "先跑一次 run" in capsys.readouterr().err


def test_cli_rejects_a_bad_config(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"specs": {"algorithms": ["nope"]}}), encoding="utf-8")
    assert cli.main(["run", "--config", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert "配置或参数错误" in capsys.readouterr().err


def test_cli_has_no_rounds_subcommand() -> None:
    """本项目不是一轮一轮迭代的，用「第几轮」描述会误导 —— 刻意不提供。"""
    parser = cli.build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    subcommands = set()
    for action in actions:
        subcommands |= set(action.choices)
    assert "rounds" not in subcommands
    assert {"run", "compare", "frontier", "diff", "explain", "list", "dag", "encode"} <= subcommands


def test_cli_out_option_belongs_to_run_only() -> None:
    """只读子命令当然也有 ``--out``（它要指出读哪里），但 ``run`` 是唯一
    会往里面写东西的 —— 这条约定由 ``test_cli_readonly_commands_do_not_touch_the_outputs``
    守着，这里只确认 ``run`` 的默认值是相对路径 ``out``。"""
    parser = cli.build_parser()
    ns = parser.parse_args(["run"])
    assert ns.out == "out"
    assert ns.config.endswith("configs/default.json") or ns.config == "configs/default.json"


def test_cli_selected_run_is_the_best_within_its_budget(tiny_result) -> None:
    """预算规则保证：入选者在自己那一档里压缩最好。

    若这条挂了，说明「选型规则」与「对比矩阵」的口径打架了 ——
    报告会同时说「按预算内压缩最好选的」和「另一个 spec 在同一档压缩更好」。
    """
    result, _ = tiny_result
    selected = result.find(result.selected)
    assert selected is not None
    same_budget = [
        r for r in result.runs if r.spec.vocab_size == selected.spec.vocab_size
    ]
    assert len(same_budget) > 1
    assert all(
        selected.metrics.tokens_per_char <= r.metrics.tokens_per_char
        for r in same_budget
    )


def test_report_markdown_file_matches_render_markdown(tiny_out, tiny_result) -> None:
    """``run`` 落盘的 ``report.md`` 必须与 ``render_markdown`` 同源 ——
    两处各写一遍必然会在某次改动后分家。"""
    written = (tiny_out / "report.md").read_text(encoding="utf-8")
    assert written == report.render_markdown(tiny_result[0])


# ----------------------------------------------------------------------
# 随项目发布的语料：它必须为 Diff 词做覆盖
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shipped_corpus_covers_every_diff_word_character() -> None:
    """Diff 词里的**每一个字符**都必须出现在训练语料的字符集里。

    这条测试是补出来的：旗舰演示词 ``中华人民共和国`` 一开始在 Diff 里全是
    ``[UNK]`` —— 训练语料里根本没有「人」「共」「华」「国」「民」这些字。
    这类错误**不会报错**，只会让 README 里那个旗舰演示变成一堆 ``[UNK]``，
    而人眼扫表格时很容易漏掉。所以它必须是一条测试。

    刻意走**真实仓库数据**（``configs/default.json`` + ``data/corpus/``），
    不用小语料夹具：要守的正是「随项目发布的这份语料本身是否够用」。
    只做字符集比对、不训练，所以很快。
    """
    from tokenizerfactory.config import load_config, load_text_file, resolve_path
    from tokenizerfactory.corpus import alphabet_of, pre_tokenize
    from tokenizerfactory.normalize import normalize

    cfg = load_config(REPO_ROOT / "configs" / "default.json")
    form = cfg["normalize"]["form"]
    train = normalize(load_text_file(cfg, "train"), form=form)

    alphabet: set[str] = set()
    for rule in [str(r) for r in cfg["specs"]["pre_tokenizers"]]:
        alphabet |= alphabet_of(pre_tokenize(train, rule))

    diff_path = resolve_path(cfg["_root"], cfg["corpus"]["diff_words"])
    lines = [
        line.strip()
        for line in diff_path.read_text(encoding="utf-8").splitlines()
    ]
    lines = [line for line in lines if line and not line.startswith("#")]
    assert lines, "Diff 词文件是空的"

    for line in lines:
        # 归一化后行内只剩内容字符（空白已被折叠）；空格由预分词转成 ▁，
        # 所以「空格」不需要在字符集里。
        missing = sorted(set(normalize(line, form=form)) - alphabet - {" "})
        assert not missing, f"{line!r} 里有训练语料未覆盖的字符：{missing}"


def test_shipped_corpus_is_not_empty_and_is_mixed_script() -> None:
    """语料要真的中英混排 —— 纯英文或纯中文会让一半结论失去意义。"""
    from tokenizerfactory.config import load_config, load_text_file
    from tokenizerfactory.corpus import char_class, corpus_stats, pre_tokenize
    from tokenizerfactory.normalize import normalize

    cfg = load_config(REPO_ROOT / "configs" / "default.json")
    form = cfg["normalize"]["form"]
    for key in ("train", "benchmark", "hard"):
        text = normalize(load_text_file(cfg, key), form=form)
        assert len(text) > 500, key

    train = normalize(load_text_file(cfg, "train"), form=form)
    stats = corpus_stats("train", train, "unicode")
    assert stats.n_chars > 1000
    assert stats.n_unique_words > 100
    assert stats.script_shares.get("cjk", 0.0) > 0.2
    assert stats.script_shares.get("latin", 0.0) > 0.2

    # ``char_class`` 的分桶：六个内容类别 + 词首标记 ``▁``（预分词引入的，
    # 不是语料自带的字符，所以它单独一类）。写全这个集合，
    # 换语料后若冒出第七类，这条测试会立刻发现。
    classes = {char_class(ch) for ch in pre_tokenize(train, "unicode")[0]}
    assert classes <= {
        "cjk",
        "latin",
        "digit",
        "symbol",
        "emoji",
        "other",
        "marker",
    }
    assert "marker" in classes, "预分词必须留下词首标记，否则「词边界」信息丢了"
