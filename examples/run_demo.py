"""端到端演示脚本：一次运行展示系统的全部能力。

运行::

    python examples/run_demo.py            # 用内置示例稿件
    python examples/run_demo.py 我的稿件.md --doc-type 学术期刊
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofread_agent import ProofreadAgent  # noqa: E402
from proofread_agent.terminology import TerminologyUpdater  # noqa: E402


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="智能审校 Agent 端到端演示")
    ap.add_argument("path", nargs="?", default=str(ROOT / "data/samples/demo_report.md"))
    ap.add_argument("--doc-type", default="调研专报")
    ap.add_argument("--preset", default="公文专报")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    banner("① 运行环境")
    agent = ProofreadAgent()
    for k, v in agent.info.items():
        if k != "graph_nodes":
            print(f"    {k:22s}: {v}")
    print(f"    图节点数              : {len(agent.info['graph_nodes'])}")

    banner("② 多 Agent 工作流结构")
    for i, a in enumerate(agent.describe_agents(), 1):
        print(f"    {i:2d}. {a['label']:22s} [{a['node']}]")
        print(f"        {a['role']}")

    banner("③ 术语库动态更新（影子模式）")
    updater = TerminologyUpdater(agent.store, agent.settings)
    sample_policy = (
        "要坚持不懈用习近平新时代中国特色社会主义思想凝心铸魂，"
        "坚持稳中求进工作总基调，牢牢把握高质量发展这个首要任务，"
        "铸牢中华民族共同体意识。"
    )
    cands = updater.extract_candidates(sample_policy, "demo-policy-file", "示例政策文件")
    print(f"    从示例政策文本中抽取候选术语 {len(cands)} 条（进入待审区，不直接生效）：")
    for c in cands[:6]:
        print(f"      · {c.term}   （置信度 {c.confidence:.2f}，句式「{c.cue}」）")

    banner("④ 执行审校")
    path = Path(args.path)
    report = agent.audit_file(path, doc_type=args.doc_type, style_preset=args.preset)
    c = report.counts
    print(f"    稿件      : {report.doc_title}")
    print(f"    签发门禁  : {report.release_gate}    综合评分: {report.overall_score}")
    print(f"    问题统计  : 致命 {c['FATAL']}｜严重 {c['MAJOR']}｜"
          f"一般 {c['MINOR']}｜提示 {c['INFO']}")
    print()
    for ds in report.dimension_scores:
        bar = "█" * int(ds.score / 100 * 34)
        print(f"    {ds.dimension.value:14s} {bar:<34s} {ds.score:5.1f}")

    banner("⑤ 高优先级问题（前 12 条）")
    for i in report.sorted_issues()[:12]:
        print(f"    [{i.severity.label}] 行 {i.line}｜{i.category.value}")
        print(f"        {i.message}")
        if i.original and i.suggestion:
            print(f"        原文：{i.original[:46]}")
            print(f"        建议：{i.suggestion[:60]}")
        print(f"        规则 {i.rule_id}｜置信度 {i.confidence}｜出处 {i.source}")

    banner("⑥ 确定性复算记录")
    for r in (report.meta.get("fact_check") or [])[:8]:
        print(f"    {r['issue_rule']:26s} {r['verdict']:16s} {r.get('formula','')[:52]}")

    banner("⑦ 批评修正（误报压制）")
    notes = report.meta.get("critic_notes") or []
    if not notes:
        print("    本次未产生需要驳回/降级的意见")
    for n in notes[:8]:
        print(f"    {n['action']:10s} by {n['by']:9s} {n['reason'][:70]}")

    banner("⑧ 自动修订留痕")
    rev = (report.meta.get("llm_notes") or {}).get("revision") or {}
    print(f"    自动应用 {rev.get('changed', 0)} 处安全修改；"
          f"保留 {len(rev.get('skipped', []))} 处高风险问题交由人工处理")
    for e in (rev.get("edits") or [])[:10]:
        print(f"      · {e['rule_id']:26s} {e['before'][:24]:26s} → {e['after'][:30]}")

    banner("⑨ 导出报告")
    written = agent.export(report, args.out, stem=path.stem + "_audit")
    for fmt, p in written.items():
        print(f"    {fmt:9s}: {p}")

    banner("⑩ Agent 执行轨迹")
    for t in report.to_dict().get("trace", []):
        flag = "（本轮未激活）" if t.get("skipped") else ""
        print(f"    {t.get('node'):20s} {str(t.get('elapsed_ms', '-')):>8s} ms  "
              f"{t.get('summary', '')[:66]}{flag}")

    print()
    print("完成。请打开 HTML 报告查看划改稿与完整问题清单。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
