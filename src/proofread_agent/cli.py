"""命令行入口。

用法示例::

    python -m proofread_agent.cli info
    python -m proofread_agent.cli audit data/samples/demo_report.md --doc-type 调研专报
    python -m proofread_agent.cli audit xxx.md --preset 学术期刊 --format html
    python -m proofread_agent.cli terms update --approve-pending
    python -m proofread_agent.cli graph --mermaid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import load_settings
from .pipeline import ProofreadAgent
from .report import Severity

_SEV_ORDER = ["FATAL", "MAJOR", "MINOR", "INFO"]


def _c(text: str, color: str) -> str:
    codes = {"red": "31", "green": "32", "yellow": "33", "blue": "34",
             "magenta": "35", "cyan": "36", "gray": "90", "bold": "1"}
    if not sys.stdout.isatty():
        return text
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


def cmd_info(args: argparse.Namespace) -> int:
    agent = ProofreadAgent(args.config)
    info = agent.info
    print(_c("■ 智能审校 Agent 运行环境", "bold"))
    for k, v in info.items():
        if k == "graph_nodes":
            print(f"  {k:22s}: {len(v)} 个节点")
            for n in v:
                print(f"      - {n}")
        else:
            print(f"  {k:22s}: {v}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    agent = ProofreadAgent(args.config, preset=args.preset)
    path = Path(args.path)
    if not path.exists():
        print(_c(f"✗ 文件不存在：{path}", "red"))
        return 2

    report = agent.audit_file(path, doc_type=args.doc_type,
                              style_preset=args.preset,
                              target_journal=args.journal or "")
    text = path.read_text(encoding="utf-8")

    # ---- 终端摘要 ----
    counts = report.counts
    gate_color = {"BLOCK": "red", "REVIEW": "yellow", "PASS": "green"}[report.release_gate]
    print()
    print(_c(f"■ 审校报告｜{report.doc_title}", "bold"))
    print(f"  签发门禁 : {_c(report.release_gate, gate_color)}"
          f"  综合评分 : {_c(str(report.overall_score), gate_color)}")
    print(f"  文体/预设: {report.doc_type} / {report.style_preset}")
    print(f"  术语库   : {report.terminology_version}"
          f"  推理后端: {report.llm_backend}"
          f"{'  (已降级为规则引擎)' if report.degraded else ''}")
    print(f"  问题统计 : 致命 {counts['FATAL']}｜严重 {counts['MAJOR']}"
          f"｜一般 {counts['MINOR']}｜提示 {counts['INFO']}")
    print()
    for ds in report.dimension_scores:
        bar_len = int(ds.score / 100 * 30)
        color = "green" if ds.score >= 85 else "yellow" if ds.score >= 60 else "red"
        print(f"  {ds.dimension.value:14s} {_c('█' * bar_len + '·' * (30 - bar_len), color)}"
              f" {ds.score:5.1f}")
    print()

    if not args.quiet:
        limit = args.limit
        shown = 0
        for cat, items in report.by_category().items():
            print(_c(f"── {cat}（{len(items)} 条）", "bold"))
            for i in items:
                if shown >= limit:
                    break
                color = {"FATAL": "red", "MAJOR": "yellow", "MINOR": "green",
                         "INFO": "blue"}[i.severity.value]
                print(f"  {_c('[' + i.severity.label + ']', color)} {i.message}")
                print(f"        {_c('行', 'gray')}{i.line}  原文 "
                      f"{_c(i.original[:40], 'red')} → 建议 {_c(i.suggestion[:40], 'green')}")
                if args.verbose and i.explain:
                    print(f"        依据: {i.explain.replace(chr(10), ' ')[:160]}")
                shown += 1
            print()
        if shown >= limit:
            print(_c(f"  …… 仅显示前 {limit} 条，使用 --limit 调整", "gray"))

    # ---- 导出 ----
    out_dir = Path(args.out or "outputs")
    formats = args.format or ["markdown", "html", "json"]
    written = agent.export(report, out_dir, text=text,
                           stem=path.stem + "_audit", formats=formats)
    print(_c("■ 报告已导出：", "bold"))
    for fmt, p in written.items():
        print(f"  {fmt:9s}: {p}")
    if args.json_stdout:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.release_gate == "PASS" else 1


def cmd_terms(args: argparse.Namespace) -> int:
    agent = ProofreadAgent(args.config)
    if args.action == "update":
        result = agent.update_terminology(auto_promote=args.auto_promote)
        print(_c("■ 术语库更新", "bold"))
        print(f"  扫描来源       : {result['sources_scanned']}")
        print(f"  候选条目(原始) : {result['candidates_raw']}")
        print(f"  新增候选       : {result['candidates_new']}")
        print(f"  待审文件       : {result['pending_file']}")
        print(f"  影子校验       : {result['shadow_diff'].get('summary')}")
        print(f"  是否已转正     : {result['promoted']}（模式：{result['mode']}）")
        if result.get("pending_file"):
            print(_c(f"\n  提示：请人工核对 {result['pending_file']} 后执行 "
                     f"`terms approve --file <path>`", "yellow"))
        return 0
    if args.action == "approve":
        if not args.file:
            print(_c("✗ 需要 --file 指定待审文件", "red"))
            return 2
        result = agent.approve_pending(args.file, args.ids)
        print(_c("■ 术语审核通过并写入本地台账", "bold"))
        for k, v in result.items():
            print(f"  {k}: {v}")
        return 0
    if args.action == "show":
        store = agent.store
        print(_c(f"■ 术语库 v{store.version}：{len(store.entries)} 条术语，"
                 f"{len(store.banned)} 条禁用模式", "bold"))
        for e in store.entries[: args.limit]:
            print(f"  {_c(e.id, 'cyan')} [{e.severity.value:5s}] {e.term}"
                  f"  变体 {len(e.variants)} 个｜来源 {e.source[:28]}")
        if len(store.entries) > args.limit:
            print(_c(f"  …… 共 {len(store.entries)} 条", "gray"))
        return 0
    return 2


def cmd_graph(args: argparse.Namespace) -> int:
    agent = ProofreadAgent(args.config)
    if args.mermaid:
        print(agent.graph_mermaid())
        return 0
    print(_c("■ 多 Agent 工作流节点", "bold"))
    for i, a in enumerate(agent.describe_agents(), 1):
        print(f"  {i:2d}. {_c(a['label'], 'cyan')}  [{a['node']}]")
        print(f"      {a['role']}")
    print()
    print(f"  执行引擎: {agent.info['engine']}"
          f"（{'LangGraph' if agent.info['engine'] == 'langgraph' else '内置顺序执行器'}）")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    args.path = args.path or "data/samples/demo_report.md"
    args.doc_type = args.doc_type or "调研专报"
    args.preset = args.preset or "公文专报"
    return cmd_audit(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="proofread-agent",
        description="基于本地部署大语言模型的智能审校 Agent（政治术语/语文规范/逻辑/"
                    "专业深度/风格迁移）",
    )
    p.add_argument("--config", default=None, help="配置文件路径，默认 config/settings.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="查看运行环境与 Agent 配置")
    s.set_defaults(func=cmd_info)

    def _audit_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("path", nargs="?", help="待审校文件路径（.md/.txt）")
        sp.add_argument("--doc-type", default="调研专报",
                        choices=["调研专报", "学术期刊", "研究报告", "新闻通稿", "公文", "论文"])
        sp.add_argument("--preset", default=None,
                        choices=["公文专报", "学术期刊", "新闻通稿"])
        sp.add_argument("--journal", default="", help="目标刊物名称（影响格式规则）")
        sp.add_argument("--out", default="outputs", help="报告输出目录")
        sp.add_argument("--format", nargs="*", choices=["markdown", "html", "json"])
        sp.add_argument("--limit", type=int, default=25, help="终端展示的问题条数上限")
        sp.add_argument("--quiet", action="store_true", help="只输出汇总，不列问题")
        sp.add_argument("--verbose", action="store_true", help="输出规则依据")
        sp.add_argument("--json-stdout", action="store_true", help="额外把 JSON 报告打印到标准输出")

    s = sub.add_parser("audit", help="执行审校")
    _audit_args(s)
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("demo", help="用内置示例稿件跑一次端到端演示")
    _audit_args(s)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("terms", help="术语库维护")
    s.add_argument("action", choices=["show", "update", "approve"])
    s.add_argument("--file", default=None, help="待审文件（approve 时使用）")
    s.add_argument("--ids", nargs="*", default=None, help="仅审核指定条目 ID")
    s.add_argument("--auto-promote", action="store_true", help="跳过人工审批直接转正")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_terms)

    s = sub.add_parser("graph", help="查看多 Agent 工作流结构")
    s.add_argument("--mermaid", action="store_true", help="输出 Mermaid 流程图")
    s.set_defaults(func=cmd_graph)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # 把项目根加入搜索路径，允许直接以脚本方式运行
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
