"""命令行入口。

设计约定（与同工作区其它项目共用一套，见 python-pipeline-scaffold skill）：

- **``--out`` 只属于 ``run``**。其余子命令是**只读观察者**，
  它们只读 ``out/`` 里的产物，不写任何东西 —— 观察者会改状态是最容易
  把人绕晕的一类设计。
- **只读子命令不做训练**。它们从 ``result.json`` + ``runs.jsonl`` 重建结果对象；
  两处例外都是**载入**而不是重算：``encode`` 从 ``artifacts/<spec>.json``
  载入一个分词器（那正好是「导出的产物真的能用」这条承诺的可执行检查），
  ``dag`` 从 ``lineage.json`` 重建血缘图（否则 ``--dot`` 只能打印一句提示）。
- **未知 ``spec_id`` 返回 1 并给相近建议**，不抛栈。
- 刻意**没有** ``rounds`` 子命令：本项目不是一轮一轮迭代的，
  用「第几轮」描述会误导。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import report
from .compare import frontier_key, frontier_pairs
from .config import load_config
from .lineage import LineageDAG
from .metrics import METRIC_DIRECTIONS, METRIC_LABELS
from .pipeline import run_pipeline, write_outputs
from .schema import CorpusStats, TokenizerMetrics, TokenizerResult, TokenizerRun, TokenizerSpec
from .tokenizers import load_tokenizer


# ----------------------------------------------------------------------
# 读取产物
# ----------------------------------------------------------------------
def _load_result(out_dir: Path) -> TokenizerResult:
    result_path = out_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"找不到 {result_path}；先跑一次 run（例如 python main.py run --out {out_dir}）"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = TokenizerResult(corpus=CorpusStats(**payload["corpus"]))
    result.matrix = payload.get("matrix", [])
    result.frontiers = payload.get("frontiers", {})
    result.diffs = payload.get("diffs", {})
    result.selected = payload.get("selected", "")
    result.counters = payload.get("counters", {})
    result.cfg = payload.get("config", {})

    runs_path = out_dir / "runs.jsonl"
    if runs_path.exists():
        runs: list[TokenizerRun] = []
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            runs.append(_run_from_dict(json.loads(line)))
        result.runs = runs

    # 有 artifact 就把真实词表挂回 run —— explain 要展示它，
    # 而 result.json 里刻意没存词表（12 个 spec 的完整词表会把文件撑大好几倍）
    artifacts = out_dir / "artifacts"
    if artifacts.is_dir():
        for run in result.runs:
            path = artifacts / f"{run.spec_id}.json"
            if path.exists():
                run.vocab = list(
                    json.loads(path.read_text(encoding="utf-8")).get("vocab", [])
                )
    return result


def _run_from_dict(payload: dict[str, Any]) -> TokenizerRun:
    spec_dict = dict(payload["spec"])
    spec_dict.pop("spec_id", None)
    return TokenizerRun(
        spec=TokenizerSpec(**spec_dict),
        vocab=[],
        vocab_digest=str(payload.get("vocab_digest", "")),
        metrics=TokenizerMetrics(**payload["metrics"]),
        train_seconds=float(payload.get("train_seconds", 0.0)),
        sample=[(w, pieces) for w, pieces in payload.get("sample", [])],
        meta=dict(payload.get("meta", {})),
    )


def _print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    print(report.markdown_table(header, rows))


def _suggest(result: TokenizerResult, spec_id: str) -> str:
    known = result.known_spec_ids()
    prefix = [k for k in known if k.startswith(spec_id.split("-")[0])]
    pool = prefix or known
    return ", ".join(pool[:8]) + ("…" if len(pool) > 8 else "")


# ----------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result, tokenizers = run_pipeline(cfg, verbose=not args.quiet)
    written = write_outputs(result, tokenizers, args.out)
    report_path = Path(args.out) / "report.md"
    report_path.write_text(report.render_markdown(result), encoding="utf-8")
    written["report.md"] = report_path

    print()
    print(report.summary_line(result))
    print()
    print(f"产物目录：{Path(args.out).resolve()}")
    for name in sorted(written):
        print(f"  {name}")
    print()
    print(f"下一步：python main.py compare --out {args.out}")
    print(f"        python main.py diff    --out {args.out}")
    print(f"        python main.py explain {result.selected} --out {args.out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    if not result.runs:
        print("没有可对比的 run。", file=sys.stderr)
        return 1
    _print_table(report.MATRIX_HEADER, report.matrix_rows(result))
    print()
    print(report.summary_line(result))
    return 0


def _cmd_frontier(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    pairs = frontier_pairs(result.cfg) if result.cfg else []
    if args.pair:
        wanted = args.pair.replace("+", " ")
        pairs = [p for p in pairs if " ".join(p) == wanted]
        if not pairs:
            print(
                f"未知目标对 {args.pair!r}；可用："
                + ", ".join(frontier_key(p) for p in frontier_pairs(result.cfg)),
                file=sys.stderr,
            )
            return 1
    if not pairs:
        print("结果里没有记录目标对。", file=sys.stderr)
        return 1

    for pair in pairs:
        key = frontier_key(pair)
        rows = report.frontier_rows(result, key)
        print(f"## {key}")
        print()
        if not rows:
            print("（空）")
        else:
            _print_table(report.frontier_header(pair), rows)
        hv = (result.counters.get("hypervolume") or {}).get(key)
        if hv is not None:
            print()
            print(f"超体积：{hv:,.1f}　（参考点口径见 report.md 第四节）")
        print()
    print("★ = 最终入选")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    rows = report.diff_rows(result)
    if not rows:
        print("结果里没有 Diff 数据。", file=sys.stderr)
        return 1
    spec_ids = list((result.diffs or {}).get("specs", []))
    _print_table(report.diff_header(spec_ids), rows)
    print()
    print("边界一致度（与其它 spec 的 token 边界 Jaccard 系数，越高越容易互相迁移）：")
    print()
    _print_table(("spec", "平均边界一致度"), report.agreement_rows(result))
    print()
    _print_table(
        ("spec", "平均 token 数"),
        [
            (sid, f"{value:.2f}")
            for sid, value in sorted(
                ((result.diffs or {}).get("mean_tokens_per_line") or {}).items()
            )
        ],
    )
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    if result.find(args.spec_id) is None:
        print(f"未知 spec_id：{args.spec_id}", file=sys.stderr)
        print(f"相近的可选项：{_suggest(result, args.spec_id)}", file=sys.stderr)
        return 1
    print(report.explain(result, args.spec_id))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    main_key = (
        frontier_key(result.cfg["pareto"]["objectives"])
        if result.cfg.get("pareto")
        else ""
    )
    print(f"共 {len(result.runs)} 个 spec（★ 入选，▲ 在主前沿上，· 其它）")
    print()
    for run in sorted(
        result.runs, key=lambda r: (r.spec.algorithm, r.spec.vocab_size, r.spec_id)
    ):
        mark = "★" if run.spec_id == result.selected else (
            "▲" if run.spec_id in result.frontiers.get(main_key, []) else "·"
        )
        print(
            f"  {mark} {run.spec_id:<24} vocab={len(run.vocab) or run.spec.vocab_size:<5} "
            f"tok/char={run.metrics.tokens_per_char:.4f} "
            f"unk={run.metrics.unknown_rate:.4f} "
            f"hard_unk={run.metrics.hard_unknown_rate:.4f}"
        )
    return 0


def _load_dag(out_dir: Path) -> Any:
    """从 ``lineage.json`` 重建血缘图。

    ``dag`` 是只读子命令，不能依赖「同一次 run 的内存对象」；但 ``--dot`` /
    ``--frontier-only`` / ``--limit`` 都需要图对象。不重建的话 ``--dot``
    会静默退化成「打印摘要 + 一句提示」—— 用户以为拿到了 DOT，却拿到别的东西。
    """
    path = out_dir / "lineage.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("nodes"):
        return None
    return LineageDAG.from_dict(payload)


def _cmd_dag(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    dag = result.dag if result.dag is not None else _load_dag(Path(args.out))
    if dag is None:
        print(f"找不到 {Path(args.out) / 'lineage.json'}", file=sys.stderr)
        return 1
    if args.dot:
        print(dag.to_dot(only_selected_ancestry=args.frontier_only))
    else:
        print(dag.to_ascii(limit=args.limit))
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    """从 artifact 载入分词器并编码任意文本 —— 「产物真的能用」的可执行检查。"""
    out_dir = Path(args.out)
    path = out_dir / "artifacts" / f"{args.spec_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in (out_dir / "artifacts").glob("*.json"))
        print(f"找不到 {path}", file=sys.stderr)
        if available:
            print(f"可用的 spec：{', '.join(available[:10])}", file=sys.stderr)
        return 1

    tokenizer = load_tokenizer(json.loads(path.read_text(encoding="utf-8")))
    from .corpus import pre_tokenize
    from .normalize import normalize, normalize_words

    text = " ".join(args.text)
    words = normalize_words(
        pre_tokenize(normalize(text), tokenizer.spec.pre_tokenizer),
        tokenizer.spec.lowercase,
    )
    tokens: list[str] = []
    for word in words:
        tokens.extend(tokenizer.pieces_of(word))
    ids = [tokenizer.token_to_id[t] for t in tokens]

    print(f"spec: {args.spec_id}　算法={tokenizer.algorithm}　词表={len(tokenizer.vocab)}")
    print(f"预分词（{tokenizer.spec.pre_tokenizer}）：{' '.join(words)}")
    print()
    print(f"token 数：{len(tokens)}")
    print("tokens : " + " | ".join(tokens))
    print("ids    : " + " ".join(str(i) for i in ids))
    print()
    print(f"decode : {tokenizer.decode(ids)!r}")
    return 0


# ----------------------------------------------------------------------
#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenizerfactory",
        description="同一语料下对比 BPE / WordPiece / Unigram 的分词策略，产出对比矩阵、Pareto 前沿与 Tokenizer Diff",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="跑完整流水线并落盘产物")
    p_run.add_argument("--config", default="configs/default.json", help="配置文件路径；相对路径按项目根解析")
    p_run.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_run.add_argument("--quiet", action="store_true", help="不打印逐个 spec 的进度")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="打印对比矩阵（只读）")
    p_cmp.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_cmp.set_defaults(func=_cmd_compare)

    p_front = sub.add_parser("frontier", help="打印 Pareto 前沿（只读）")
    p_front.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_front.add_argument("--pair", default="", help="只看某个目标对，例如 vocab_size+tokens_per_char")
    p_front.set_defaults(func=_cmd_frontier)

    p_diff = sub.add_parser("diff", help="打印 Tokenizer Diff 与边界一致度（只读）")
    p_diff.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_diff.set_defaults(func=_cmd_diff)

    p_exp = sub.add_parser("explain", help="解释某个 spec 的结果（只读）")
    p_exp.add_argument("spec_id")
    p_exp.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_exp.set_defaults(func=_cmd_explain)

    p_list = sub.add_parser("list", help="列出全部 spec（只读）")
    p_list.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_list.set_defaults(func=_cmd_list)

    p_dag = sub.add_parser("dag", help="打印血缘图（只读）")
    p_dag.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_dag.add_argument("--dot", action="store_true", help="输出 Graphviz DOT")
    p_dag.add_argument(
        "--frontier-only",
        action="store_true",
        help="只画入选 spec 的祖先链（配合 --dot）",
    )
    p_dag.add_argument("--limit", type=int, default=30, help="ASCII 输出最多显示多少节点")
    p_dag.set_defaults(func=_cmd_dag)

    p_enc = sub.add_parser("encode", help="用导出的 artifact 编码任意文本（只读）")
    p_enc.add_argument("spec_id")
    p_enc.add_argument("text", nargs="+", help="要编码的文本（多个词会拼成一句）")
    p_enc.add_argument("--out", default="out", help="产物目录；相对路径按项目根解析")
    p_enc.set_defaults(func=_cmd_encode)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _anchor_args(args)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"配置或参数错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
