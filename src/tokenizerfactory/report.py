"""报告层：表格行构造 + Markdown 报告 + ``explain``。

行构造函数**公开**给 ``cli`` 复用（与 DataMixtureFactory 同一套做法）。
理由很实际：CLI 的 ``compare`` 与报告的「对比矩阵」必须是**同一个东西**，
两处各写一遍必然会在某次改动后悄悄分家 —— 而那种分家不会报错，
只会让「报告里说 A、CLI 里说 B」。

格式约定（三处刻意统一）
------------------------

1. **比率类指标一律带固定小数位**（压缩 4 位、覆盖率 3 位、未知率 4 位）。
   不固定的话 ``0.0`` 与 ``0.000012`` 在表里长得一样，
   而它们的含义完全不同（真的没有未知 vs 有极少量未知）。
2. **吞吐带千分位**。``1234567`` 和 ``123456`` 扫一眼分不出来。
3. **每个 spec 行都带「前沿」列**，值是它所在的目标对（可能多个）。
   只在前沿表里列出前沿点、矩阵里不标，读者就得来回对照两张表。
"""

from __future__ import annotations

from typing import Any, Sequence

from .compare import (
    dominated_count,
    dominators,
    frontier_key,
    frontier_pairs,
    hv_contributions,
    reference_point,
)
from .metrics import METRIC_DIRECTIONS, METRIC_LABELS, has_timing_axis, point_of
from .schema import UNK_TOKEN, TokenizerResult

#: 矩阵列（表头与行必须同步改，有测试守着列数一致）
MATRIX_HEADER: tuple[str, ...] = (
    "spec",
    "算法",
    "规则",
    "词表目标",
    "实际",
    "压缩(tok/字符)",
    "生育率(tok/词)",
    "覆盖率",
    "未知率",
    "硬未知率",
    "编码吞吐",
    "训练秒",
    "前沿",
)


def matrix_rows(result: TokenizerResult) -> list[tuple[str, ...]]:
    """矩阵的行。

    「前沿」列压成一个紧凑形式：``★`` 表示它在**主前沿**上，后面的数字是
    它一共进了几条前沿。不这么压的话，三个目标对的名字拼起来有七八十字，
    表格宽度直接失控；而读者真正需要一眼看出的只有「它在不在主前沿上」。
    """
    main_key = _main_key(result)
    rows: list[tuple[str, ...]] = []
    for row in result.matrix:
        keys = list(row["frontier"])
        mark = ("★" if main_key and main_key in keys else "") + str(len(keys))
        rows.append(
            (
                str(row["spec_id"]),
                str(row["algorithm"]),
                str(row["pre_tokenizer"]) + ("+lower" if row["lowercase"] else ""),
                str(row["vocab_size"]),
                str(row["vocab_size_actual"]),
                f"{float(row['tokens_per_char']):.4f}",
                f"{float(row['tokens_per_word']):.3f}",
                f"{float(row['vocab_coverage']):.3f}",
                f"{float(row['unknown_rate']):.4f}",
                f"{float(row['hard_unknown_rate']):.4f}",
                f"{float(row['encode_tokens_per_sec']):,.0f}",
                f"{float(row['train_seconds']):.2f}",
                mark,
            )
        )
    return rows


def _main_key(result: TokenizerResult) -> str:
    cfg = result.cfg or {}
    if not cfg.get("pareto"):
        return ""
    return frontier_key(cfg["pareto"]["objectives"])


def worst_alternative_at_same_budget(result: TokenizerResult) -> dict[str, Any]:
    """同预算、**同预分词规则**下最差的算法 —— 「选错算法要付多少代价」。

    这是本项目最可决策的一个数字。它把「算法选型」从一句定性的话
    （「BPE 更适合中文」）变成一个有量纲的代价（「多 71% 的序列长度」）。

    两个限定条件都不能省，否则这个数字会失去意义：

    - **同预算**：不同预算比出来的差值里混着「多花的预算」；
    - **同规则**：``char`` 规则是刻意做的生育率上界（每个码点一个词），
      拿它当「另一个算法」比，得到的是「预分词的代价」而不是「算法选型的代价」，
      数字会从 71% 虚涨到 86%。变量必须一次只动一个。
    """
    selected = result.find(result.selected)
    if selected is None:
        return {}
    budget = selected.spec.vocab_size
    same = [
        r
        for r in result.runs
        if r.spec.vocab_size == budget
        and r.spec.pre_tokenizer == selected.spec.pre_tokenizer
    ]
    if len(same) < 2:
        return {}
    worst = max(same, key=lambda r: (r.metrics.tokens_per_char, r.spec_id))
    best_value = selected.metrics.tokens_per_char
    if worst.spec_id == selected.spec_id or best_value <= 0:
        return {}
    return {
        "budget": budget,
        "pre_tokenizer": selected.spec.pre_tokenizer,
        "worst_spec": worst.spec_id,
        "worst_value": worst.metrics.tokens_per_char,
        "ratio": worst.metrics.tokens_per_char / best_value - 1.0,
    }


#: 这些指标是整数计数，表里按整数打印。``2048.0000`` 这种写法会让读者
#: 以为它是一个比率，而且比实际数字更难扫。
_INTEGER_METRICS = frozenset(
    {"vocab_size", "vocab_size_actual", "n_tokens", "n_chars", "n_words"}
)


def format_objective(name: str, value: float) -> str:
    """目标轴上的数值格式。整数类按整数打印，比率类固定 4 位小数。"""
    if name in _INTEGER_METRICS:
        return str(int(value))
    return f"{value:.4f}"


def frontier_header(objectives: Sequence[str]) -> tuple[str, ...]:
    """前沿表头。

    **「实际词表」这一列只在目标对里没有它的时候才补。**
    第一版无条件加了一列，于是当 ``objectives[0]`` 是 ``vocab_size_actual``
    （``METRIC_LABELS`` 里它就叫「实际词表」）时，表头出现了两个同名
    「实际词表」—— 一个显示 ``552``，另一个显示 ``552.0000``，读者分不清哪个是
    设计变量、哪个是结果。这个 bug 是在真实配置上跑 ``frontier`` 命令时
    看出来的（小语料夹具里没有这个目标对，所以测试没覆盖到），
    现在由「任意两个指标组合都不得出现重复列名」的测试守着。
    """
    labels = [METRIC_LABELS.get(o, o) for o in objectives]
    head = ["排名", "spec", "算法"]
    if "实际词表" not in labels:
        head.append("实际词表")
    return tuple(head + labels + ["支配数", "超体积增量"])


def _frontier_shows_actual_vocab(objectives: Sequence[str]) -> bool:
    return "实际词表" not in [METRIC_LABELS.get(o, o) for o in objectives]


def frontier_rows(result: TokenizerResult, key: str) -> list[tuple[str, ...]]:
    """前沿表的行。``key`` 形如 ``vocab_size+tokens_per_char``。"""
    objectives = key.split("+")
    ids = result.frontiers.get(key, [])
    if not ids:
        return []
    points = {r.spec_id: point_of(r) for r in result.runs}
    all_ids = sorted(points)
    contributions = hv_contributions(all_ids, points, objectives)
    ordered = sorted(ids, key=lambda i: (points[i][objectives[0]], i))
    show_actual = _frontier_shows_actual_vocab(objectives)

    rows: list[tuple[str, ...]] = []
    for rank, sid in enumerate(ordered, 1):
        run = result.find(sid)
        algorithm = run.algorithm if run else "?"
        mark = " ★" if sid == result.selected else ""
        cells = [str(rank), sid + mark, algorithm]
        if show_actual:
            cells.append(str(int(points[sid]["vocab_size_actual"])))
        cells += [format_objective(o, points[sid][o]) for o in objectives]
        cells += [
            str(dominated_count(sid, all_ids, points, objectives)),
            f"{contributions.get(sid, 0.0):,.1f}",
        ]
        rows.append(tuple(cells))
    return rows


def budget_sensitivity_rows(result: TokenizerResult) -> list[tuple[str, ...]]:
    """「换个词表预算，结论会不会变」—— 这是配置里把 ``specs`` 单列出来要回答的问题。

    每个预算一行：三个算法各自最好的压缩、该档赢家，以及**两条分开的差距**。

    为什么差距要拆成「算法」与「规则」两列？第一版只有一列「该档最大差距」，
    取该预算下**最差 spec 比最好 spec** 差多少 —— 实测下来它恒定在 +86%~+89%，
    而那个数字其实由 ``char`` 规则决定（``char`` 是刻意做的粒度极端值，
    ``tokens_per_char`` 恒为 1.0005），跟「选哪个算法」没关系。
    于是同一份报告里出现了自相矛盾的两句话：表格说「预算越大差距越大」，
    正文说「预算越小选错的代价越大」。

    所以这里按项目自己的纪律（**变量一次只动一个**）拆开：

    - ``该档算法差距``：固定赢家的预分词规则，最差的算法差多少 —— 这才是
      「算法选型有多重要」；
    - ``该档规则差距``：固定赢家的算法，最差的规则差多少 —— 这是
      「预分词规则有多重要」。

    拆开之后结论才站得住：算法差距随预算从 71.3% 降到 10.5%
    （预算越大，WordPiece 被 ``##`` 吃掉的那一半额度越摊越薄），
    而规则差距恒定在 ~89%（``char`` 是粒度极端值，与预算无关）。
    """
    rows: list[tuple[str, ...]] = []
    algorithms = result.algorithms()
    for size in result.vocab_sizes():
        at_size = [r for r in result.runs if r.spec.vocab_size == size]
        if not at_size:
            continue
        best_by_algo: dict[str, Any] = {}
        for algorithm in algorithms:
            pool = [r for r in at_size if r.algorithm == algorithm]
            if pool:
                best_by_algo[algorithm] = min(
                    pool, key=lambda r: (r.metrics.tokens_per_char, r.spec_id)
                )
        winner = min(at_size, key=lambda r: (r.metrics.tokens_per_char, r.spec_id))
        best = winner.metrics.tokens_per_char

        def gap(pool: list[Any]) -> str:
            if not pool or best <= 0:
                return "-"
            worst = max(pool, key=lambda r: (r.metrics.tokens_per_char, r.spec_id))
            return f"{worst.metrics.tokens_per_char / best - 1.0:+.1%}"

        same_rule = [
            r for r in at_size if r.spec.pre_tokenizer == winner.spec.pre_tokenizer
        ]
        same_algorithm = [r for r in at_size if r.algorithm == winner.spec.algorithm]
        rows.append(
            (str(size),)
            + tuple(
                (
                    f"{best_by_algo[a].metrics.tokens_per_char:.4f}"
                    f" ({best_by_algo[a].spec_id.split('-v')[0]}"
                    f"{'' if best_by_algo[a].spec.pre_tokenizer == 'unicode' else '+' + best_by_algo[a].spec.pre_tokenizer})"
                    if a in best_by_algo
                    else "-"
                )
                for a in algorithms
            )
            + (
                winner.spec_id,
                f"{best:.4f}",
                gap(same_rule),
                gap(same_algorithm),
            )
        )
    return rows


def pretokenizer_rows(result: TokenizerResult) -> list[tuple[str, ...]]:
    """每种预分词规则的最好结果。

    刻意把**压缩率与生育率并排放**：它们会给出不同的结论，而这个不一致
    本身就是结论（见报告第三节的说明）。
    """
    notes = {
        "unicode": "CJK 成串 / 拉丁词 / 数字串 / 表情 / 其余每符号一个 —— 默认真实可用",
        "whitespace": "只按空白切 —— 中文整句成一个词，生育率立刻暴露",
        "char": "每个码点一个词 —— 「词」的最细粒度，`tokens_per_word` 恒为 1.0",
    }
    rows: list[tuple[str, ...]] = []
    for rule in sorted({r.spec.pre_tokenizer for r in result.runs}):
        pool = [r for r in result.runs if r.spec.pre_tokenizer == rule]
        if not pool:
            continue
        best = min(pool, key=lambda r: (r.metrics.tokens_per_char, r.spec_id))
        rows.append(
            (
                rule,
                f"{best.metrics.tokens_per_char:.4f}",
                best.spec_id,
                f"{best.metrics.tokens_per_word:.3f}",
                notes.get(rule, ""),
            )
        )
    return rows


def diff_header(spec_ids: Sequence[str]) -> tuple[str, ...]:
    return ("文本",) + tuple(spec_ids) + ("最短/最长",)


def base_vocab_rows(result: TokenizerResult) -> list[tuple[str, ...]]:
    """各算法的基础词表占用。

    这是「词表预算」上的一个隐形约束，而它**因算法而异**。只在报告里给一个
    总词表数字，读者会以为三个算法是从同一起跑线开始的 —— 它们不是。
    """
    from .corpus import alphabet_of, pre_tokenize
    from .normalize import normalize
    from .tokenizers import TOKENIZERS

    if not result.cfg.get("corpus") or not result.cfg.get("_root"):
        return []
    try:
        from .config import load_text_file

        train = normalize(
            load_text_file(result.cfg, "train"),
            form=result.cfg["normalize"]["form"],
        )
    except Exception:  # pragma: no cover - 只读 CLI 缺 _root 时静默跳过
        return []
    rules = [str(r) for r in result.cfg["specs"]["pre_tokenizers"]]
    alphabet: set[str] = set()
    for rule in rules:
        alphabet |= alphabet_of(pre_tokenize(train, rule))

    rows: list[tuple[str, ...]] = []
    for algorithm in result.algorithms():
        cls = TOKENIZERS.get(algorithm)
        if cls is None:
            continue
        size = len(cls.base_vocab(alphabet))
        rows.append(
            (
                algorithm,
                str(size),
                str(len(alphabet)),
                f"{UNK_TOKEN} + {len(alphabet)} 个字符"
                + (
                    f" + {len(alphabet)} 个 ## 续接形式"
                    if algorithm == "wordpiece"
                    else ""
                ),
            )
        )
    return rows


def diff_rows(result: TokenizerResult) -> list[tuple[str, ...]]:
    """Diff 表的行：每行文本一行，每列一个 spec 的切分。"""
    data = result.diffs or {}
    spec_ids = list(data.get("specs", []))
    if not spec_ids:
        return []
    rows: list[tuple[str, ...]] = []
    for entry in data.get("table", []):
        counts = [entry["specs"][sid]["n_tokens"] for sid in spec_ids]
        rows.append(
            (entry["line"],)
            + tuple(
                f"{entry['specs'][sid]['rendered']} ({entry['specs'][sid]['n_tokens']})"
                for sid in spec_ids
            )
            + (f"{min(counts)}/{max(counts)}",)
        )
    return rows


def agreement_rows(result: TokenizerResult) -> list[tuple[str, ...]]:
    data = result.diffs or {}
    return [
        (spec_id, f"{value:.4f}")
        for spec_id, value in sorted((data.get("mean_agreement") or {}).items())
    ]


# ----------------------------------------------------------------------
# explain
# ----------------------------------------------------------------------
def explain(result: TokenizerResult, spec_id: str, *, max_tokens: int = 24) -> str:
    """把「为什么是这个结果」写清楚。未知 spec 由 CLI 负责提示相近项。"""
    run = result.find(spec_id)
    if run is None:
        return f"未知 spec_id：{spec_id}"

    points = {r.spec_id: point_of(r) for r in result.runs}
    ids = sorted(points)
    lines: list[str] = []
    lines.append(f"spec: {run.spec_id}")
    lines.append(
        f"  算法={run.spec.algorithm}  词表目标={run.spec.vocab_size}  "
        f"实际={len(run.vocab)}  预分词={run.spec.pre_tokenizer}  "
        f"小写={run.spec.lowercase}"
    )
    lines.append(f"  词表摘要={run.vocab_digest[:16]}  训练={run.train_seconds:.3f}s")
    if run.spec.extra:
        lines.append(
            "  算法参数=" + ", ".join(f"{k}={v}" for k, v in sorted(run.spec.extra.items()))
        )
    if run.meta:
        lines.append(
            "  训练诊断=" + ", ".join(f"{k}={v}" for k, v in sorted(run.meta.items()))
        )

    lines.append("")
    lines.append("度量：")
    metrics = point_of(run)
    for name in (
        "tokens_per_char",
        "tokens_per_word",
        "vocab_coverage",
        "unknown_rate",
        "hard_unknown_rate",
        "encode_tokens_per_sec",
        "decode_tokens_per_sec",
    ):
        value = metrics[name]
        if name.endswith("_per_sec"):
            text = f"{value:,.0f}"
        elif name == "tokens_per_word":
            text = f"{value:.4f}"
        else:
            text = f"{value:.6f}"
        lines.append(f"  {METRIC_LABELS.get(name, name):<18} {text}")

    on = sorted(key for key, members in result.frontiers.items() if spec_id in members)
    lines.append("")
    if on:
        lines.append(f"前沿：在 {len(on)} 个目标对上（{', '.join(on)}）")
        for key in on:
            objectives = key.split("+")
            dom = dominated_count(spec_id, ids, points, objectives)
            lines.append(f"  [{key}] 支配了 {dom} 个其它 spec")
    else:
        lines.append("前沿：不在任何目标对的前沿上")
        for key in result.frontiers:
            objectives = key.split("+")
            who = dominators(spec_id, ids, points, objectives)
            if who:
                lines.append(
                    f"  [{key}] 被 {len(who)} 个 spec 支配，例如：{', '.join(who[:4])}"
                )

    lines.append("")
    lines.append(f"词表片段（前 {max_tokens} 个）：")
    lines.append("  " + " ".join(repr(t) for t in run.vocab[:max_tokens]))
    multi = [t for t in run.vocab if len(t) > 1][:max_tokens]
    lines.append(f"学到的多字 piece（前 {max_tokens} 个）：")
    lines.append("  " + " ".join(repr(t) for t in multi))

    if run.sample:
        lines.append("")
        lines.append(f"切分样本（benchmark 前 {min(8, len(run.sample))} 个词）：")
        for word, pieces in run.sample[:8]:
            lines.append(f"  {word}  →  {' | '.join(pieces)}")

    data = result.diffs or {}
    if spec_id in (data.get("specs") or []):
        lines.append("")
        lines.append("在 Diff 选中的文本上的切分：")
        for entry in data.get("table", [])[:6]:
            cell = entry["specs"].get(spec_id)
            if cell:
                lines.append(f"  {entry['line']}  →  {cell['rendered']} ({cell['n_tokens']})")
        agreement = (data.get("mean_agreement") or {}).get(spec_id)
        if agreement is not None:
            lines.append(f"  与其它 spec 的平均边界一致度：{agreement:.4f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Markdown 报告
# ----------------------------------------------------------------------
def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "（空）"
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def summary_line(result: TokenizerResult) -> str:
    """一句话结论 —— 报告与 CLI 都用它，避免两处口径不同。

    刻意**不**把「比基线好多少」当成主结论。实测下来最常见的默认选择
    （bpe + 最小预算 + unicode）在这份语料上恰好就是该预算下最好的 ——
    于是「搜索带来 +0.00% 改进」这种话会读起来像「搜索没用」，
    而真相是**这份语料上默认选择是对的**，那是结果而不是前提。
    真正可决策的数字是「选错算法的代价」，所以把它放在结论里。
    """
    selected = result.find(result.selected)
    if selected is None:
        return "没有选出任何 spec。"
    parts = [
        f"入选 {selected.spec_id}"
        f"（{selected.spec.algorithm} / 词表 {len(selected.vocab) or selected.spec.vocab_size} / "
        f"{selected.spec.pre_tokenizer} 预分词）",
        f"压缩 {selected.metrics.tokens_per_char:.4f} tok/字符",
        f"未知率 {selected.metrics.unknown_rate:.4f}",
        f"硬语料未知率 {selected.metrics.hard_unknown_rate:.4f}",
    ]
    spread = worst_alternative_at_same_budget(result)
    if spread:
        parts.append(
            f"同预算（{spread['budget']}）下最差的算法 {spread['worst_spec']} 是 "
            f"{spread['worst_value']:.4f}，选错要多花 {spread['ratio']:.1%} 的序列长度"
        )
    return "；".join(parts) + "。"


def render_markdown(result: TokenizerResult) -> str:
    cfg = result.cfg or {}
    pairs = frontier_pairs(cfg) if cfg.get("pareto") else []
    counters = result.counters
    points = {r.spec_id: point_of(r) for r in result.runs}
    ids = sorted(points)

    out: list[str] = []
    out.append("# TokenizerFactory 运行报告")
    out.append("")
    if result.version:
        out.append(
            f"- 版本：`{result.version.version_id}`（摘要 `{result.version.digest[:16]}`）"
        )
    out.append(f"- 生成时间：{result.version.created_at if result.version else '-'}")
    out.append(f"- 代码版本：`{_code_version()}`")
    out.append("")
    out.append("## 一、结论")
    out.append("")
    out.append(summary_line(result))
    out.append("")
    spread = worst_alternative_at_same_budget(result)
    if spread:
        out.append(
            f"**本项目最可决策的一个数字是「选错算法的代价」**：在 "
            f"{spread['budget']} 这个预算档上，最好的 `{result.selected}` 与最差的 "
            f"`{spread['worst_spec']}` 相差 {spread['ratio']:.1%} 的序列长度。"
            "它不是一句定性的「BPE 更适合中文」，而是一个有量纲的代价 —— "
            "可以直接拿去和「换一个更大的模型」「加一层缓存」这类方案比较。"
        )
        out.append("")
    reason = counters.get("selected_reason")
    if reason:
        out.append(f"选型规则：{reason}。")
        out.append("")
        out.append(
            "> 注意：Pareto 前沿只说明「哪些配置不被别人支配」，**它不告诉你选哪个**。"
            "选型规则（词表预算 + 压缩最好）是一个额外的价值判断，不是从数据里推出来的。"
            "换个规则（例如 `knee`）会选出别的 spec，两者的取舍下面都列了。"
        )
        out.append("")

    out.append("## 二、语料画像")
    out.append("")
    corpus = result.corpus
    out.append(f"- 语料：`{corpus.name}`")
    out.append(f"- 字符数 {corpus.n_chars}，词数 {corpus.n_words}，唯一词 {corpus.n_unique_words}")
    out.append(
        f"- 最长词 {corpus.max_word_len} 字符，平均 {corpus.mean_word_len:.2f} 字符，"
        f"主导字符类 `{corpus.dominant_script()}`"
    )
    shares = "、".join(
        f"{k} {v:.1%}" for k, v in sorted(corpus.script_shares.items(), key=lambda kv: -kv[1])
    )
    out.append(f"- 字符构成：{shares}")
    out.append("")
    out.append(
        "字符构成决定了**该关注哪个维度**：CJK 占比高时预分词规则对生育率的影响最大"
        "（中文没有空格）；符号占比高时未知率才是关键。"
    )
    out.append("")
    out.append("### 基础词表占用：预算里先被扣掉的那一块")
    out.append("")
    out.append(
        "每个算法的初始词表都必须装下语料里出现过的**每个字符**，否则它连一个词都编不出来。"
        "这块占用**因算法而异**，而且差距很大 —— 它是「词表预算」这个设计变量上"
        "一个容易被忽略的隐形约束。"
    )
    out.append("")
    out.append(markdown_table(("算法", "基础词表", "其中字符", "说明"), base_vocab_rows(result)))
    out.append("")
    out.append(
        "WordPiece 的占用是别人的两倍，因为它的非词首子词一律带 `##` 前缀，"
        "所以每个字符需要两种形式。在 2048 的预算下，它只剩约一半的额度用来学合并 —— "
        "这是它在小预算下明显落后的直接原因，而不是它的挑选准则更差。"
        "**词表预算必须大于这个占用**，否则训练出来的不是分词器而是按字符切分；"
        "配置加载期会直接报错并算出下限，不会等到跑完才失败。"
    )
    out.append("")

    out.append("## 三、对比矩阵")
    out.append("")
    out.append(
        f"{counters.get('n_specs', 0)} 个 spec（"
        f"{counters.get('n_algorithms', 0)} 算法 × "
        f"{counters.get('n_vocab_sizes', 0)} 词表规模 × "
        f"{counters.get('n_pre_tokenizers', 0)} 预分词规则）。"
        "「前沿」列：`★` 表示在主前沿上，后面的数字是它一共进了几条前沿。"
    )
    out.append("")
    out.append(markdown_table(MATRIX_HEADER, matrix_rows(result)))
    out.append("")
    out.append(
        "**两条读表时的口径限制**（不写明的话这两列会被读成它们不是的东西）："
    )
    out.append("")
    out.append(
        "1. **「覆盖率」跨预分词规则不可比。** 它是词级覆盖率，而「词」的定义"
        "依赖预分词规则：`unicode` 下一个汉字串是一个词，`char` 下每个汉字都是一个词。"
        "后者只要单字认识就算覆盖，前者要整串字符都认识才算 —— 于是 `char` 的覆盖率"
        "天然偏高（本表里 0.977 vs 0.916），这不是它的词表更好，而是分母更细。"
        "**只在同一规则内部比覆盖率。**"
    )
    out.append(
        "2. **「未知率」的分母是 token 数，所以它会随压缩率一起动。** `char` 规则的"
        "未知率最低（0.0233），但它的 token 数是别人的两倍多 —— 同样的未知字符数"
        "摊到更大的分母上自然更小。要看未知的绝对量，请对照「实际词表」与 Diff 一节。"
    )
    out.append("")
    out.append("### 预分词规则的影响")
    out.append("")
    out.append(
        "这是本项目与「只比较算法和词表规模」的报告最大的不同："
        "**切分规则被当成一个需要比较的变量，而不是预处理细节。**"
    )
    out.append("")
    out.append(markdown_table(("预分词规则", "最好压缩", "对应 spec", "生育率(tok/词)", "说明"), pretokenizer_rows(result)))
    out.append("")
    out.append(
        "看这张表时注意**压缩率与生育率会给出不同的结论**。"
        "`whitespace` 在中文上把整句变成一个词，词表被整句占满 —— "
        "但它的压缩率只比 `unicode` 差一点点，因为编码总能退回字符级。"
        "真正被暴露出来的是**生育率**：它按词算，立刻显示每个「词」要切出好几个 token。"
        "只看压缩率会得出「预分词规则无关紧要」的错误结论，这正是本项目要量五个维度的原因。"
    )
    out.append("")

    out.append("## 四、Pareto 前沿")
    out.append("")
    for pair in pairs:
        key = frontier_key(pair)
        out.append(f"### {key}")
        out.append("")
        if has_timing_axis(pair):
            out.append(
                "> **含实测轴**：这条前沿里有一个轴是计时测出来的（编码/解码吞吐）。"
                "速度只能靠测量得到，而测量有噪声 —— 两个速度本来就很接近的 spec "
                "会在两次运行之间交换前沿成员（实测过：同一份配置跑两次，"
                "这条前沿的成员从 `wordpiece-v128` 变成了 `wordpiece-v128+whitespace`）。"
                "所以**这条前沿与它的超体积不可逐位复现**，只能当参考；"
                "要下结论请用不含实测轴的那条。"
            )
            out.append("")
        ref = reference_point(ids, points, pair)
        out.append(
            f"- 前沿点数：{len(result.frontiers.get(key, []))}"
            f"　超体积：{counters.get('hypervolume', {}).get(key, 0.0):,.1f}"
        )
        out.append(
            f"- 参考点：({ref[0]:.4f}, {ref[1]:.4f}) —— "
            "取全部候选点在两个方向上的最差值再外扩 5%。取全部候选点而不是前沿点，"
            "是为了让超体积能在「同一候选集的不同前沿子集」之间比较。"
        )
        out.append("")
        out.append(markdown_table(frontier_header(pair), frontier_rows(result, key)))
        out.append("")
    baseline = counters.get("baseline") or {}
    if baseline:
        same = baseline.get("best_at_same_vocab") or {}
        out.append(f"**基线口径**：`{baseline['spec_id']}` —— {baseline.get('why', '')}。")
        out.append("")
        out.append(
            f"- 它被 {baseline.get('dominated_by', 0)} 个 spec 支配"
            + (
                f"（例如 {', '.join(baseline.get('dominated_by_sample', [])[:4])}）"
                if baseline.get("dominated_by_sample")
                else ""
            )
            + "。"
        )
        if same:
            if same["spec_id"] == baseline["spec_id"]:
                out.append(
                    f"- **不额外花预算**：同一个词表档里最好的就是基线自己（"
                    f"{same['tokens_per_char']:.4f}）。也就是说**在这份语料上，"
                    "「不比较就直接上」恰好是对的** —— 默认选择不是错的，"
                    "只是这个结论只有比较过之后才敢说出口。"
                    "这也是本项目不把「比基线好多少」当主结论的原因："
                    "这里它是一个恒为 0 的差，读起来像「搜索没用」，"
                    "而真相是**搜索确认了默认选择**。"
                )
            else:
                out.append(
                    f"- **不额外花预算**：同一个词表档里最好的 `{same['spec_id']}` 是 "
                    f"{same['tokens_per_char']:.4f}，比基线好 {same['gain']:+.4f}"
                    f"（{same['gain_pct']:+.2f}%）。这部分的收益完全来自「选了哪个算法/规则」，"
                    "与预算无关。"
                )
        upgrade = baseline.get("selected_vs_baseline")
        if upgrade:
            out.append(
                f"- **多花预算**：入选的 `{upgrade['spec_id']}` 是 "
                f"{upgrade['vocab_size']} 档，压缩比基线好 {upgrade['gain']:+.4f}。"
                f"{upgrade['note']}。"
            )
        else:
            out.append(
                "- **多花预算**：入选者就是基线本身，所以没有「比基线好多少」这个数字 —— "
                "那会是一个恒为 0 的差。"
            )
        out.append("")

    out.append("### 换个词表预算，赢家会换人吗")
    out.append("")
    out.append(
        "这张表回答的是配置里 ``specs`` 单独列出来要回答的那个问题："
        "**词表预算本身是不是一个能改变结论的变量**。"
        "每一格是该算法在该预算下最好的压缩率（括号里是它的预分词规则）。"
    )
    out.append("")
    # 表与下面的结论用的是同一份行 —— 原来算了两遍
    gap_rows = budget_sensitivity_rows(result)
    out.append(
        markdown_table(
            ("词表预算",)
            + tuple(result.algorithms())
            + ("该档赢家", "赢家压缩", "该档算法差距", "该档规则差距"),
            gap_rows,
        )
    )
    out.append("")
    out.append(
        "每一格是该算法在该预算下最好的压缩率，括号里是它用的预分词规则。"
        "**两条差距列刻意分开，因为它们的含义完全不同**："
        "「该档算法差距」固定赢家的预分词规则、只让算法变；"
        "「该档规则差距」固定赢家的算法、只让规则变。"
        "变量一次只动一个，数字才读得懂。"
    )
    out.append("")
    # 结论里的数字**从表里算**，不写死 —— 写死的话换一份语料就会出现
    # 「正文说 10.5%、表格写 10.2%」这种自相矛盾，而它不会被任何测试发现。
    if gap_rows:
        first, last = gap_rows[0], gap_rows[-1]
        algo = [row[-2] for row in gap_rows if row[-2] != "-"]
        rule = [row[-1] for row in gap_rows if row[-1] != "-"]
        out.append(
            f"拆开之后结论才站得住：**算法差距随预算从 {first[-2]}（{first[0]}）"
            f"降到 {last[-2]}（{last[0]}）** —— 预算越大，WordPiece 被 `##` "
            "吃掉的那一半额度越摊越薄，算法之间的差距被摊平。"
            + (
                f"而**规则差距稳定在 {min(rule)}~{max(rule)}**，"
                "因为它由 `char` 决定（每个码点一个词，`tokens_per_char` 与预算无关）。"
                if rule
                else ""
            )
        )
        out.append("")
        out.append(
            "第一版把两者合成一列「该档最大差距」（取该档最差 spec 比最好 spec）。"
            "那份语料上它恒定在 +86%~+89% —— 因为**最差的 spec 永远是 `char` 那一个**，"
            "于是那一列实际报告的是「规则差距」，却被摆在「算法选型有多重要」的位置上。"
            "结果是同一份报告里出现了「预算越大差距越大」与「预算越小选错的代价越大」"
            "两句互相矛盾的话。合并出来的数字不属于任何一条决策轴。"
        )
        out.append("")

    out.append("## 五、Tokenizer Diff")
    out.append("")
    out.append(
        "同一段文本在不同 spec 下的切分。**这是「为什么」的唯一来源** —— "
        "矩阵只说差多少，Diff 说差在哪。"
    )
    out.append("")
    diff_rows_data = diff_rows(result)
    if diff_rows_data:
        spec_ids = list((result.diffs or {}).get("specs", []))
        out.append(markdown_table(diff_header(spec_ids), diff_rows_data))
        out.append("")
        out.append("### 边界一致度")
        out.append("")
        out.append(
            "每个 spec 与其它 spec 的 token 边界 Jaccard 系数（对 Diff 文本取平均）。"
            "**这个数字决定跨模型迁移的对齐难度**：一致度低的两个分词器，"
            "同一个字串在两套 id 空间里落在完全不同的位置，混用或换分词器继续训练"
            "都要重做对齐。"
        )
        out.append("")
        out.append(markdown_table(("spec", "平均边界一致度"), agreement_rows(result)))
        out.append("")
        out.append("平均每行 token 数（序列长度的直接度量）：")
        out.append("")
        out.append(
            markdown_table(
                ("spec", "平均 token 数"),
                [
                    (sid, f"{value:.2f}")
                    for sid, value in sorted(
                        ((result.diffs or {}).get("mean_tokens_per_line") or {}).items()
                    )
                ],
            )
        )
        out.append("")
    else:
        out.append("（没有可用的 Diff 文本）")
        out.append("")

    out.append("## 六、每个维度上谁最强")
    out.append("")
    best_rows: list[tuple[str, ...]] = []
    for name in (
        "tokens_per_char",
        "tokens_per_word",
        "vocab_coverage",
        "unknown_rate",
        "hard_unknown_rate",
        "encode_tokens_per_sec",
        "decode_tokens_per_sec",
    ):
        run = result.best_by(name, METRIC_DIRECTIONS[name])
        if run is None:
            continue
        best_rows.append(
            (
                METRIC_LABELS.get(name, name),
                METRIC_DIRECTIONS[name],
                run.spec_id,
                f"{run.metrics.get(name):,.6f}",
            )
        )
    out.append(markdown_table(("维度", "方向", "最好的 spec", "数值"), best_rows))
    out.append("")
    out.append(
        "这张表是**单维最优**，与前两节的 Pareto 前沿不是一回事："
        "某一维最优的 spec 很可能在另一维很差，所以它未必在前沿上。"
        "两处都给出来，是为了让「谁在哪方面强」和「谁整体不被支配」都能被检查。"
    )
    out.append("")
    char_wins = [
        METRIC_LABELS.get(row[0], row[0])
        for row in best_rows
        if "+char" in row[2]
    ]
    if char_wins:
        out.append(
            "> **这张表有一半的赢家是 `char` 规则**（"
            + "、".join(char_wins)
            + "）。那不是因为它的词表更好 —— `char` 把「词」定义成单个码点，"
            "于是 `tokens_per_word` 恒为 1.0、覆盖率与未知率的判定粒度也最细。"
            "这正是第三节那条口径限制的实例：**这三个指标跨预分词规则不可比**。"
            "所以这张表要横着读（看某一维谁最强），不要竖着读"
            "（不要以为 `char` 整体更好 —— 它的压缩率是别人的近两倍）。"
        )
        out.append("")

    out.append("## 七、血缘与产物")
    out.append("")
    if result.dag is not None:
        summary = result.dag.summary()
        out.append(
            f"- DAG：{summary['nodes']} 节点 / {summary['edges']} 边 / "
            f"{summary['merge_points']} 个收拢点"
        )
        out.append(
            "- 节点构成：" + "、".join(f"{k} {v}" for k, v in sorted(summary["by_kind"].items()))
        )
        out.append("")
        out.append("```")
        out.append(result.dag.to_ascii(limit=24))
        out.append("```")
        out.append("")
    out.append("| 文件 | 内容 |")
    out.append("| --- | --- |")
    for name, desc in (
        ("result.json", "完整产物（矩阵 / 前沿 / Diff / 版本 / 计数器）"),
        ("matrix.csv", "对比矩阵"),
        ("frontier.csv", "各目标对的前沿点"),
        ("runs.jsonl", "每个 spec 一行的完整结果（含切分样本）"),
        ("diffs.json", "Tokenizer Diff 的完整结果"),
        ("lineage.json / lineage.dot", "血缘图（JSON / Graphviz）"),
        ("lineage_selected.dot", "只画入选 spec 的祖先链"),
        ("artifacts/<spec>.json", "每个 spec 的**可载入**分词器产物"),
        ("version.json", "版本（摘要由入选词表决定）"),
    ):
        out.append(f"| `{name}` | {desc} |")
    out.append("")
    return "\n".join(out)


def _code_version() -> str:
    from .schema import CODE_VERSION

    return CODE_VERSION


def meta_payload(result: TokenizerResult) -> dict[str, Any]:
    """给 CLI 的 ``--json`` 用的轻量摘要（不含 runs 全量）。"""
    return {
        "version": result.version.to_dict() if result.version else {},
        "selected": result.selected,
        "counters": result.counters,
        "frontiers": result.frontiers,
        "summary": summary_line(result),
    }
