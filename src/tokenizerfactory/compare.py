"""对比矩阵与 Pareto 前沿 —— 回答「同一语料下哪种分词策略更合适」。

这一层刻意做得很薄：它只做三件事，而且每件都有明确的口径。

1. **摊平**：把每个 spec 的结果变成「指标名 → 数值」，含 spec 派生量
   （``vocab_size`` 目标 / ``vocab_size_actual`` 实际）。设计变量与结果变量
   必须同时出现 —— BPE 在 pair 用尽时会提前停下，``actual < target``
   本身就是一个结果。

2. **求前沿**：在**离散**设计空间上做非支配筛选。
   与 DataMixtureFactory 不同，这里的决策变量是离散的（算法 × 词表规模 × 规则），
   没有连续搜索，所以没有「采样密度」这类问题；代价是前沿上的点数很少
   （通常 3~6 个），于是**前沿上每一个点都必须能被解释**，
   不能只报一个前沿大小就完事。

3. **选一个**：``select`` 规则。这里必须诚实：Pareto 前沿只说明
   「哪些配置不被别人支配」，它**不告诉你选哪个**。真实部署里词表大小是
   硬约束（embedding 表的显存/参数量），所以默认规则是
   「预算内压缩最好的那个」。这是**一个额外的价值判断**，不是从数据里推出来的，
   报告里会写明这一点。

坐标约定（踩过坑，写清楚）
--------------------------
- 支配判定与排序用 ``min`` 形式：两个方向都转成「越小越好」
  （``max`` 方向的指标取负）。
- 超体积用**同一个 min 形式**加一个参考点，不再是 DataMixtureFactory 那套
  「最大形式 + 参考点」。同一份代码里并存两套坐标约定是上一个项目
  真实踩过的坑（``hypervolume_of_front`` 吃最大形式的点却内部调
  ``(quality, cost)`` 约定的 ``dominates``，静默削掉半个前沿）。
  本项目只保留一套，并且超体积函数自己重算非支配集，不信任调用方传来的点。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .metrics import METRIC_DIRECTIONS, point_of
from .schema import TokenizerResult, TokenizerRun


def frontier_key(pair: Sequence[str]) -> str:
    """目标对的稳定键名（也是 ``result.frontiers`` 的键）。"""
    return "+".join(pair)


def frontier_pairs(cfg: dict[str, Any]) -> list[list[str]]:
    """主目标对 + 额外目标对。主目标对排在第一个。"""
    main = list(cfg["pareto"]["objectives"])
    extras = [list(p) for p in cfg["pareto"].get("extra_objectives", [])]
    out = [main]
    for pair in extras:
        if pair not in out:
            out.append(pair)
    return out


# ----------------------------------------------------------------------
# 支配
# ----------------------------------------------------------------------
def to_min_form(point: dict[str, float], objectives: Sequence[str]) -> tuple[float, ...]:
    """按指标方向把点转成「越小越好」的形式。"""
    return tuple(
        point[o] if METRIC_DIRECTIONS[o] == "min" else -point[o] for o in objectives
    )


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """``a`` 是否支配 ``b``（全不差，且至少一维更好）。"""
    not_worse = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return not_worse and strictly_better


def non_dominated(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> list[str]:
    """非支配集。返回顺序与输入一致，便于稳定输出。"""
    values = {i: to_min_form(points[i], objectives) for i in ids}
    out: list[str] = []
    for i in ids:
        vi = values[i]
        if any(
            j != i and dominates(values[j], vi)
            for j in ids
        ):
            continue
        out.append(i)
    return out


def dominators(
    target: str, ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> list[str]:
    """谁支配了 ``target``。报告里用来解释「基线为什么没进前沿」。"""
    vt = to_min_form(points[target], objectives)
    return [i for i in ids if i != target and dominates(to_min_form(points[i], objectives), vt)]


def dominated_count(
    target: str, ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> int:
    """``target`` 支配了多少个 spec。

    前沿上每个点都值得给出这个数：它回答「这个取舍轴上的位置，
    在真实的设计空间里排掉了多少个别的配置」——
    比只报一个「前沿有 N 个点」有信息量得多。
    """
    vt = to_min_form(points[target], objectives)
    return sum(
        1
        for i in ids
        if i != target and dominates(vt, to_min_form(points[i], objectives))
    )


def hv_contributions(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> dict[str, float]:
    """每个前沿点对超体积的**增量**贡献（阶梯面积）。

    前沿点的增量之和恒等于总超体积 —— 这条恒等式有测试守着。
    它让「某个点值不值得留在前沿上」变成一个可读的数字：
    贡献接近 0 的点说明它只是勉强没被支配，对取舍空间的贡献可以忽略。
    """
    ref = reference_point(points.keys(), points, objectives)
    front = non_dominated(ids, points, objectives)
    ordered = sorted(
        (to_min_form(points[i], objectives), i) for i in front
    )
    out: dict[str, float] = {}
    prev_y = ref[1]
    for value, i in ordered:
        x, y = value
        if y >= prev_y:
            out[i] = 0.0
            continue
        out[i] = (ref[0] - x) * (prev_y - y)
        prev_y = y
    return out


# ----------------------------------------------------------------------
# 超体积（二维，min 形式）
# ----------------------------------------------------------------------
#: 参考点外扩比例。参考点取全部候选点的最差值再外扩 —— 不外扩的话，
#: 「恰好在预算上限」那个点的箱体宽度为 0、贡献 0 面积，于是超体积对
#: 「要不要保留最大预算那一档」异常敏感。这是标准的边界约定而非 bug，
#: 但它不适合回答本项目的任何问题，所以外扩掉。
REFERENCE_MARGIN = 0.05


def reference_point(
    ids: Iterable[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> tuple[float, float]:
    """参考点 = 全部候选点在两个方向上的最差值，再外扩 ``REFERENCE_MARGIN``。

    取「全部候选点」而不是「前沿点」：这样超体积可以在
    「同一候选集的不同前沿子集」之间比较 —— 而那正是它的用途
    （例如「多加两条预分词规则，前沿整体覆盖了多少更多空间」）。
    """
    mins = [to_min_form(points[i], objectives) for i in ids]
    if not mins:
        return (1.0, 1.0)
    return tuple(  # type: ignore[return-value]
        max(v[k] for v in mins) * (1.0 + REFERENCE_MARGIN) for k in range(2)
    )


def hypervolume_2d(
    frontier: Iterable[tuple[float, ...]], reference: tuple[float, float]
) -> float:
    """二维 min 形式的超体积：``∪ [x_i, R0] × [y_i, R1]`` 的面积。

    自己先重算一遍非支配集再走阶梯求和 —— 调用方传进来的点若含有被支配点，
    阶梯会算错（多算面积）而**不会报错**。多算一遍很便宜，静默错一次很贵。
    """
    pts = list(frontier)
    keep = [
        p
        for p in pts
        if not any(dominates(q, p) for q in pts if q != p)
    ]
    if not keep:
        return 0.0
    ordered = sorted(keep, key=lambda p: (p[0], p[1]))
    hv = 0.0
    prev_y = reference[1]
    for x, y in ordered:
        if y >= prev_y:
            continue  # 退化：不在阶梯上，跳过（正常输入不会出现）
        hv += (reference[0] - x) * (prev_y - y)
        prev_y = y
    return hv


def hypervolume_of(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> float:
    """一组点的超体积（先筛非支配，再用全部候选点定参考点）。"""
    ref = reference_point(points.keys(), points, objectives)
    front = non_dominated(ids, points, objectives)
    return hypervolume_2d([to_min_form(points[i], objectives) for i in front], ref)


# ----------------------------------------------------------------------
# 拐点
# ----------------------------------------------------------------------
def knee_point(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> str | None:
    """前沿的「拐点」：离首尾连线最远的那个点。

    每个轴先按方向归一化到 ``[0, 1]``（越大越好）再算距离。

    归一化的作用要说准（第一版 docstring 把它写错了，被测试纠正）：
    逐轴仿射缩放**不会**改变「点到直线距离」的排序 —— 归一化前后两个距离
    只差一个常数因子，所以它改变的不是选谁。真正依赖归一化的是
    ``start`` / ``end`` 的选取：它们按 ``(x, y)`` 做**字典序**比较，
    用原始量时量级大的那个轴会单独决定选谁（词表规模会彻底压掉压缩率）。
    顺带的好处是中间量都落在 ``[0, 1]``，数值条件好一些。
    """
    ordered = sorted(ids)
    if not ordered:
        return None
    if len(ordered) <= 2:
        return ordered[0]

    def axis(name: str) -> dict[str, float]:
        values = [points[i][name] for i in ordered]
        lo, hi = min(values), max(values)
        span = hi - lo
        out: dict[str, float] = {}
        for i in ordered:
            if span == 0:
                out[i] = 1.0
            elif METRIC_DIRECTIONS[name] == "min":
                out[i] = 1.0 - (points[i][name] - lo) / span
            else:
                out[i] = (points[i][name] - lo) / span
        return out

    o0, o1 = objectives
    x, y = axis(o0), axis(o1)
    start = max(ordered, key=lambda i: (x[i], y[i]))
    end = max(ordered, key=lambda i: (y[i], x[i]))
    dx, dy = x[end] - x[start], y[end] - y[start]
    length = math.hypot(dx, dy)
    if length == 0:
        return ordered[0]

    best: str | None = None
    best_distance = -1.0
    for i in ordered:
        distance = abs(dy * (x[i] - x[start]) - dx * (y[i] - y[start])) / length
        if distance > best_distance:
            best_distance, best = distance, i
    return best


# ----------------------------------------------------------------------
# 选型
# ----------------------------------------------------------------------
def select(result: TokenizerResult, cfg: dict[str, Any]) -> tuple[str, str]:
    """返回 ``(spec_id, 理由)``。

    理由必须一并返回：选型规则是**一个价值判断**，不是一个从数据里推出来的
    结论。报告里要把「按什么规则选的」和「按别的规则会选谁」都写出来。
    """
    if not result.runs:
        return "", "没有可选的 run"

    rule = cfg["select"]["rule"]
    if rule == "knee":
        key = frontier_key(cfg["pareto"]["objectives"])
        ids = result.frontiers.get(key, [])
        chosen = knee_point(ids, {r.spec_id: point_of(r) for r in result.runs}, cfg["pareto"]["objectives"])
        if chosen:
            return chosen, f"knee：主前沿 {key} 上离首尾连线最远的点"
        return "", "主前沿为空，knee 规则无法选型"

    budget = int(cfg["select"]["vocab_budget"])
    within = [r for r in result.runs if r.spec.vocab_size <= budget]
    if not within:
        within = list(result.runs)
        reason = (
            f"max_compression_within_budget：没有任何 spec 的 vocab_size ≤ {budget}，"
            f"退化为在全部 {len(within)} 个 spec 里选压缩最好的"
        )
    else:
        reason = (
            f"max_compression_within_budget：在 vocab_size ≤ {budget} 的 "
            f"{len(within)} 个 spec 里选 tokens_per_char 最小的"
        )
    chosen = min(
        within,
        key=lambda r: (r.metrics.tokens_per_char, r.spec.vocab_size, r.spec_id),
    )
    return chosen.spec_id, reason


# ----------------------------------------------------------------------
# 组装
# ----------------------------------------------------------------------
def build_matrix(result: TokenizerResult, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """对比矩阵：每个 spec 一行，含设计变量 + 五维度量。"""
    rows: list[dict[str, Any]] = []
    for run in sorted(
        result.runs, key=lambda r: (r.spec.algorithm, r.spec.vocab_size, r.spec_id)
    ):
        rows.append(
            {
                "spec_id": run.spec_id,
                "algorithm": run.spec.algorithm,
                "pre_tokenizer": run.spec.pre_tokenizer,
                "lowercase": run.spec.lowercase,
                "vocab_size": run.spec.vocab_size,
                "vocab_size_actual": len(run.vocab),
                "vocab_digest": run.vocab_digest[:12],
                "train_seconds": round(run.train_seconds, 4),
                "frontier": [],
                **run.metrics.to_dict(),
            }
        )
    return rows


def assemble(result: TokenizerResult, cfg: dict[str, Any]) -> None:
    """就地填好 ``matrix`` / ``frontiers`` / ``selected`` / ``counters``。"""
    points = {r.spec_id: point_of(r) for r in result.runs}
    ids = sorted(points)

    frontiers: dict[str, list[str]] = {}
    for pair in frontier_pairs(cfg):
        key = frontier_key(pair)
        frontiers[key] = non_dominated(ids, points, pair)
    result.frontiers = frontiers

    matrix = build_matrix(result, cfg)
    for row in matrix:
        row["frontier"] = sorted(
            key for key, members in frontiers.items() if row["spec_id"] in members
        )
    result.matrix = matrix

    selected, reason = select(result, cfg)
    result.selected = selected

    main_pair = list(cfg["pareto"]["objectives"])
    result.counters = {
        "n_specs": len(result.runs),
        "n_algorithms": len(result.algorithms()),
        "n_vocab_sizes": len(result.vocab_sizes()),
        "n_pre_tokenizers": len({r.spec.pre_tokenizer for r in result.runs}),
        "selected_reason": reason,
        "frontier_sizes": {k: len(v) for k, v in sorted(frontiers.items())},
        "hypervolume": {
            key: round(hypervolume_of(frontiers[key], points, pair), 6)
            for key, pair in ((frontier_key(p), p) for p in frontier_pairs(cfg))
        },
        "reference_point": {
            frontier_key(p): [round(v, 6) for v in reference_point(ids, points, p)]
            for p in frontier_pairs(cfg)
        },
        "best_by": {
            name: {
                "spec_id": run.spec_id,
                "value": round(run.metrics.get(name), 6),
            }
            for name in (
                "tokens_per_char",
                "tokens_per_word",
                "vocab_coverage",
                "unknown_rate",
                "encode_tokens_per_sec",
            )
            if (run := result.best_by(name, METRIC_DIRECTIONS[name])) is not None
        },
        # 基线口径：同预算下最「朴素」的那个 spec（最小词表、按字符切分）。
        # 「搜索/多配置优于基线」是 README 的验收标准之一，必须有明确的基线。
        "baseline": _baseline(result, main_pair),
    }


def _baseline(result: TokenizerResult, objectives: Sequence[str]) -> dict[str, Any]:
    """基线 = **不比较就直接上的那个配置**。

    具体是「最常见的默认选择」：``bpe`` + 最小词表预算 + ``unicode`` 预分词 +
    不小写。这不是从数据里推出来的，而是一个关于「人默认会怎么做」的假设 ——
    必须写出来，否则「比基线好多少」这个说法没有意义。

    为什么不用「前沿上词表最小的那个点」当基线？因为那已经是比较的结果了，
    拿它当基线等于「用最好的低预算配置去比最好的高预算配置」，
    改进量会被压到接近 0（实测只有 +0.0079），读起来像是「搜索没什么用」。
    真正该回答的是：**如果你不比较，你会损失多少。**
    """
    runs = result.runs
    if not runs:
        return {}
    smallest = min(r.spec.vocab_size for r in runs)
    naive = [
        r
        for r in runs
        if r.spec.algorithm == "bpe"
        and r.spec.vocab_size == smallest
        and r.spec.pre_tokenizer == "unicode"
        and not r.spec.lowercase
    ]
    baseline_run = (
        min(naive, key=lambda r: (r.metrics.tokens_per_char, r.spec_id))
        if naive
        else min(runs, key=lambda r: (r.spec.vocab_size, r.spec_id))
    )

    ids = sorted(r.spec_id for r in runs)
    points = {r.spec_id: point_of(r) for r in runs}
    dom = dominators(baseline_run.spec_id, ids, points, objectives)

    # 同一预算下最好的那个 —— 「不额外花预算，光靠选型能拿回多少」
    same_budget = [r for r in runs if r.spec.vocab_size == baseline_run.spec.vocab_size]
    best_same = min(
        same_budget, key=lambda r: (r.metrics.tokens_per_char, r.spec_id)
    )
    selected_run = result.find(result.selected)

    payload: dict[str, Any] = {
        "spec_id": baseline_run.spec_id,
        "why": (
            f"不比较就直接上的默认选择：bpe + 最小词表预算 {smallest} + unicode 预分词"
        ),
        "values": {
            o: round(points[baseline_run.spec_id][o], 6) for o in objectives
        },
        "vocab_size": baseline_run.spec.vocab_size,
        "dominated_by": len(dom),
        "dominated_by_sample": dom[:8],
        "best_at_same_vocab": {
            "spec_id": best_same.spec_id,
            "tokens_per_char": round(best_same.metrics.tokens_per_char, 6),
            "gain": round(
                baseline_run.metrics.tokens_per_char - best_same.metrics.tokens_per_char, 6
            ),
            "gain_pct": round(
                100.0
                * (
                    baseline_run.metrics.tokens_per_char
                    - best_same.metrics.tokens_per_char
                )
                / baseline_run.metrics.tokens_per_char,
                2,
            )
            if baseline_run.metrics.tokens_per_char
            else 0.0,
        },
    }
    if selected_run is not None and selected_run.spec_id != baseline_run.spec_id:
        payload["selected_vs_baseline"] = {
            "spec_id": selected_run.spec_id,
            "vocab_size": selected_run.spec.vocab_size,
            "same_budget": selected_run.spec.vocab_size == baseline_run.spec.vocab_size,
            "gain": round(
                baseline_run.metrics.tokens_per_char
                - selected_run.metrics.tokens_per_char,
                6,
            ),
            "note": (
                "同一预算下比较"
                if selected_run.spec.vocab_size == baseline_run.spec.vocab_size
                else f"预算不同（{baseline_run.spec.vocab_size} → "
                f"{selected_run.spec.vocab_size}），这个差值里混着「多花的预算」"
            ),
        }
    return payload
