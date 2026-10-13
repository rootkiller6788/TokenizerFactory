"""血缘图。

与前几个项目共用同一套图抽象（见 `factory-everything-项目总表.md` 的
「跨栈同构族：图引擎」），但**这张图记录的东西不一样**：

| 项目 | 图记录什么 | 边的含义 |
| --- | --- | --- |
| DatasetFactory | 数据流过哪些 stage | 上游产物 → 下游产物 |
| SyntheticFactory | 进化链（基因组如何被迭代） | 上一轮 evolve → 下一轮 generate |
| DataMixtureFactory | 搜索谱系（配方如何被提出） | 父试验 → 子试验 |
| **TokenizerFactory** | **设计空间的展开与收敛** | **语料 → 规则 → 每个 spec 的训练 → 评测 → 入选** |

本项目的图是**「扇形展开再收拢」**的形状：一个语料分叉出多个预分词规则，
每个规则下再分叉出 `算法 × 词表规模` 个训练节点，最后只有前沿上的那几个
收拢到 Select 与 Artifact。收拢点（多输入节点）因此**恰好是「被选中的东西」** ——
看一张图就知道哪些 spec 进了前沿，不必翻报告。

节点粒度
--------
一个 spec 拆成两个节点：

    train:{spec_id}   ← 输入是它用的 norm 节点（该 spec 的预分词规则）
    bench:{spec_id}   ← 输入是 train，记录五维度量

为什么 encode 与 measure 不拆开？因为「编码 benchmark → 量五个维度」是
**一次不可分割的评测**：中间没有任何产物被复用，拆开只会把图撑大而信息量不变。
这与 DataMixtureFactory 里 train → benchmark → score 不拆开是同一条理由。

不变量
------
**每个 spec 的血缘链必须是本图里的一条真实路径**，且终点是 ``corpus`` 根节点：

    corpus → norm:{rule} → train:{spec_id} → bench:{spec_id}

这条不变量在前几个项目里都救过场（见 python-pipeline-scaffold skill 的坑 12、13）：
扁平列表和图一旦分家，``explain`` 就会悄悄缺一块，而且不报错。
"""

from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import utcnow

#: 语料根节点 —— 所有链路的起点
CORPUS_NODE_ID = "corpus"


def norm_node_id(pre_tokenizer: str, lowercase: bool) -> str:
    """归一化节点 id。同一套（预分词规则, 是否小写）只建一个节点。"""
    return f"norm:{pre_tokenizer}{'+lower' if lowercase else ''}"


def train_node_id(spec_id: str) -> str:
    return f"train:{spec_id}"


def bench_node_id(spec_id: str) -> str:
    return f"bench:{spec_id}"


def select_node_id(tag: str) -> str:
    return f"sel:{tag}"


def artifact_node_id(tag: str) -> str:
    return f"art:{tag}"


@dataclass
class LineageNode:
    """DAG 节点。

    kind 取值：``corpus`` | ``normalize`` | ``train`` | ``bench`` | ``select`` | ``artifact``
    """

    node_id: str
    kind: str
    label: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # ``outputs`` 排序后导出。它的追加顺序取决于「子节点是何时被加进来的」，
        # 而 ``from_dict`` 重建时子节点是按拓扑序加进来的 —— 不排序的话，
        # 同一个图的两条构建路径会导出两份不同的 lineage.json，
        # 于是「产物是否可复现」这个检查永远过不了。
        payload["outputs"] = sorted(self.outputs)
        return payload


class LineageDAG:
    """不可变追加式血缘图。"""

    def __init__(self) -> None:
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def add_node(self, node: LineageNode) -> str:
        if node.node_id in self.nodes:
            raise ValueError(f"node already exists: {node.node_id}")
        for parent in node.inputs:
            if parent not in self.nodes:
                raise ValueError(
                    f"unknown parent '{parent}' for node '{node.node_id}'"
                )
        self.nodes[node.node_id] = node
        for parent in node.inputs:
            self.nodes[parent].outputs.append(node.node_id)
            self.edges.append((parent, node.node_id))
        return node.node_id

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def parents(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].inputs)

    def children(self, node_id: str) -> list[str]:
        return list(self.nodes[node_id].outputs)

    def ancestors(self, node_id: str) -> list[str]:
        """所有上游节点（不含自身）。"""
        seen: set[str] = set()
        stack = list(self.parents(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parents(cur))
        return sorted(seen)

    def descendants(self, node_id: str) -> list[str]:
        """所有下游节点（不含自身）。"""
        seen: set[str] = set()
        stack = list(self.children(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.children(cur))
        return sorted(seen)

    def path_to_root(self, node_id: str) -> list[str]:
        """沿父节点走到根的一条路径（多父时取名字最小的，保证确定）。"""
        path = [node_id]
        seen = {node_id}
        cur = node_id
        while True:
            nxt = next(
                (p for p in sorted(self.parents(cur)) if p not in seen), None
            )
            if nxt is None:
                return path
            path.append(nxt)
            seen.add(nxt)
            cur = nxt

    def topological(self) -> list[str]:
        """拓扑序；存在环则抛错。

        用最小堆而不是 FIFO 队列：同一批「已就绪」的节点里总是先取名字最小的，
        得到**字典序最小的拓扑序** —— 既确定，又符合阅读直觉。
        换成 FIFO 时顺序取决于入队时机，会出现「收拢节点排在它的某个上游之前」
        这种合法但看着像链路断了的结果。
        """
        indeg = {n: 0 for n in self.nodes}
        for _, dst in self.edges:
            indeg[dst] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        heapq.heapify(queue)
        order: list[str] = []
        while queue:
            cur = heapq.heappop(queue)
            order.append(cur)
            for child in sorted(self.children(cur)):
                indeg[child] -= 1
                if indeg[child] == 0:
                    heapq.heappush(queue, child)
        if len(order) != len(self.nodes):
            raise ValueError("cycle detected in lineage DAG")
        return order

    def by_kind(self, kind: str) -> list[str]:
        return sorted(n for n, node in self.nodes.items() if node.kind == kind)

    def merge_points(self) -> list[str]:
        """多输入节点 —— 本项目里就是 Select 与 Artifact（被选中的东西）。"""
        return sorted(n for n in self.topological() if len(self.nodes[n].inputs) > 1)

    def leaves(self) -> list[str]:
        return sorted(n for n, node in self.nodes.items() if not node.outputs)

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "by_kind": kinds,
            "merge_points": len(self.merge_points()),
        }

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[n].to_dict() for n in self.topological()],
            # 边**排序后**导出。``self.edges`` 的插入顺序取决于「节点是何时被加进来的」，
            # 而 ``from_dict`` 重建时节点是按拓扑序加进来的 —— 不排序的话，
            # 同一个图的两条构建路径会导出两份不同的 lineage.json，
            # 于是「产物是否可复现」这个检查永远过不了。
            "edges": [{"from": a, "to": b} for a, b in sorted(self.edges)],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineageDAG":
        """从 ``to_dict()`` 的产物重建。

        存在的理由很实际：``dag`` 是只读子命令，它不能靠「同一次 run 的内存对象」，
        但 ``--dot`` / ``--frontier-only`` / ``--limit`` 又都需要图对象。
        于是就地重建一份 —— 不重建的话 ``--dot`` 在 CLI 里会静默退化成
        「打印一段摘要加一句提示」，用户以为拿到了 DOT 却拿到别的东西。

        两个细节：
        - ``nodes`` 按拓扑序排列，所以顺序迭代时父节点一定先于子节点；
        - 节点的 ``outputs`` 一律**清空**后交给 ``add_node`` 重新推出来。
          直接照抄会让父节点的 ``outputs`` 被追加两次（``add_node`` 会再追加一遍）。
        """
        dag = cls()
        for raw in payload.get("nodes", []):
            dag.add_node(
                LineageNode(
                    node_id=str(raw["node_id"]),
                    kind=str(raw["kind"]),
                    label=str(raw["label"]),
                    inputs=[str(p) for p in raw.get("inputs", [])],
                    params=dict(raw.get("params", {})),
                    created_at=str(raw.get("created_at", utcnow())),
                    stats=dict(raw.get("stats", {})),
                )
            )
        return dag

    def to_dot(self, only_selected_ancestry: bool = False) -> str:
        """Graphviz DOT。

        ``only_selected_ancestry=True`` 时只画 Select / Artifact 的祖先闭包 ——
        本项目有 3 算法 × N 词表规模 × M 规则 个训练节点，全画出来太密，
        而真正有信息量的是「入选的分词器是从哪些配置里长出来的」。
        """
        if only_selected_ancestry:
            keep: set[str] = set()
            for nid in self.by_kind("select") + self.by_kind("artifact"):
                keep.add(nid)
                keep |= set(self.ancestors(nid))
        else:
            keep = set(self.nodes)

        colors = {
            "corpus": "lightgrey",
            "normalize": "lightyellow",
            "train": "white",
            "bench": "white",
            "select": "lightgreen",
            "artifact": "lightblue",
        }
        lines = [
            "digraph lineage {",
            "  rankdir=LR;",
            '  node [shape=box, fontname="Helvetica"];',
        ]
        # 按**拓扑序**遍历节点，不按插入顺序：插入顺序取决于「节点是何时被
        # 加进来的」，重建（``from_dict``）时会不一样，于是同一张图会导出
        # 两份不同的 DOT。拓扑序是图自身的性质，与构建路径无关。
        for node in (self.nodes[n] for n in self.topological()):
            if node.node_id not in keep:
                continue
            color = colors.get(node.kind, "white")
            lines.append(
                f'  "{node.node_id}" [label="{node.label}\\n({node.kind})", '
                f"style=filled, fillcolor={color}];"
            )
        for a, b in sorted(self.edges):
            if a in keep and b in keep:
                lines.append(f'  "{a}" -> "{b}";')
        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self, limit: int | None = None) -> str:
        """按拓扑序渲染。收拢点（多输入节点）永远单独列出且不受 ``limit`` 影响。"""
        order = self.topological()
        if not order:
            return "(empty DAG)"

        shown = order
        elided = 0
        if limit is not None and len(order) > limit:
            head = limit // 2
            tail = limit - head
            shown = order[:head] + order[len(order) - tail :]
            elided = len(order) - len(shown)

        id_w = max(len(n) for n in order)
        label_w = min(48, max(len(self.nodes[n].label) for n in order))
        kind_w = max(len(self.nodes[n].kind) for n in order)

        lines: list[str] = ["Lineage DAG（拓扑序）", ""]
        for idx, nid in enumerate(shown):
            if elided and idx == limit // 2:
                lines.append(f"  … 省略 {elided} 个节点（用 --dot 看全图）…")
            node = self.nodes[nid]
            stats = "  ".join(f"{k}={v}" for k, v in (node.stats or {}).items())
            lines.append(
                f"  {nid:<{id_w}}  {node.label:<{label_w}}  "
                f"[{node.kind:<{kind_w}}]  {stats}"
            )

        merges = self.merge_points()
        if merges:
            lines.append("")
            lines.append(f"收拢点（多输入节点，{len(merges)} 个）：")
            for nid in merges[:12]:
                node = self.nodes[nid]
                lines.append(f"  {nid}  ← {', '.join(node.inputs)}")
            if len(merges) > 12:
                lines.append(f"  … 另有 {len(merges) - 12} 个")

        return "\n".join(lines)
