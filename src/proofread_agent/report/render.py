"""报告渲染：Markdown / HTML / JSON。

HTML 报告包含五个视图，对应编辑的实际工作流：
1. **总览**：评分、签发门禁、维度雷达（用条形代替，避免引入图表库依赖）；
2. **划改稿**：原稿按字符偏移着色，点击色块直达问题详情；
3. **问题清单**：按类别分组，含"原文 → 建议"对照、规则依据、置信度；
4. **修改留痕**：自动修订记录与 diff；
5. **审计轨迹**：每个 Agent 的耗时与产出，满足可追溯要求。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AuditReport, Category, Issue, Severity

_SEV_COLOR = {
    "FATAL": ("#fff1f0", "#cf1322", "#a8071a"),
    "MAJOR": ("#fff7e6", "#d46b08", "#ad4e00"),
    "MINOR": ("#f6ffed", "#389e0d", "#237804"),
    "INFO": ("#f0f5ff", "#2f54eb", "#1d39c4"),
}
_GATE = {
    "BLOCK": ("#cf1322", "不予签发", "存在致命级问题，必须修改后重新审校"),
    "REVIEW": ("#d46b08", "退改", "存在需要修改的问题，建议修改后再审"),
    "PASS": ("#389e0d", "通过", "未发现阻断性问题"),
}


# ======================================================================
# JSON
# ======================================================================
def render_json(report: AuditReport, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=indent)


# ======================================================================
# Markdown
# ======================================================================
def render_markdown(report: AuditReport, text: str = "") -> str:
    d = report.to_dict()
    counts = report.counts
    gate_color, gate_label, gate_desc = _GATE[report.release_gate]
    out: List[str] = []
    out.append(f"# 智能审校报告｜{report.doc_title}")
    out.append("")
    out.append(f"> **签发门禁：{gate_label}（{report.release_gate}）** — {gate_desc}")
    out.append(f"> 综合评分 **{report.overall_score}** / 100｜文体：{report.doc_type}"
               f"｜风格预设：{report.style_preset or '未指定'}")
    out.append(f"> 字数 {report.char_count}｜段落 {report.paragraph_count}｜"
               f"术语库版本 `{report.terminology_version}`｜推理后端 `{report.llm_backend}`"
               f"{'（已降级为规则引擎）' if report.degraded else ''}")
    out.append("")

    out.append("## 一、维度评分")
    out.append("")
    out.append("| 维度 | 得分 | 致命 | 严重 | 一般 | 提示 | 合计 |")
    out.append("|------|-----:|-----:|-----:|-----:|-----:|-----:|")
    for ds in report.dimension_scores:
        out.append(f"| {ds.dimension.value} | {ds.score} | {ds.fatal} | {ds.major} | "
                   f"{ds.minor} | {ds.info} | {ds.total} |")
    out.append(f"| **综合** | **{report.overall_score}** | {counts['FATAL']} | "
               f"{counts['MAJOR']} | {counts['MINOR']} | {counts['INFO']} | "
               f"{len(report.issues)} |")
    out.append("")

    out.append("## 二、审校结论")
    out.append("")
    out.append("```")
    out.append(report.summary)
    out.append("```")
    out.append("")

    out.append("## 三、问题清单")
    out.append("")
    for cat, items in report.by_category().items():
        out.append(f"### {cat}（{len(items)} 条）")
        out.append("")
        for i, issue in enumerate(items, 1):
            out.append(f"**{i}. [{issue.severity.label}] {issue.message}**")
            out.append("")
            out.append(f"- 位置：第 {issue.line} 行，第 {issue.paragraph + 1} 段"
                       f"（字符 {issue.span.start}–{issue.span.end}）")
            if issue.original:
                out.append(f"- 原文：`{issue.original}`")
            if issue.suggestion:
                out.append(f"- 建议：`{issue.suggestion}`")
            if issue.explain:
                out.append(f"- 依据：{issue.explain.replace(chr(10), ' ')}")
            if issue.source:
                out.append(f"- 出处：{issue.source}")
            out.append(f"- 规则：`{issue.rule_id}`｜置信度 {issue.confidence}｜"
                       f"检出 Agent：`{issue.agent}`")
            out.append("")

    fact = d.get("fact_check") or []
    if fact:
        out.append("## 四、数据与事实复算记录")
        out.append("")
        out.append("| 规则 | 复算公式 | 结论 | 置信度 | 信源 |")
        out.append("|------|----------|------|-------:|------|")
        for r in fact:
            out.append(f"| {r.get('issue_rule')} | {r.get('formula', '')} | "
                       f"{r.get('verdict')} | {r.get('confidence')} | {r.get('source', '')} |")
        out.append("")

    notes = d.get("critic_notes") or []
    if notes:
        out.append("## 五、批评修正记录（误报压制）")
        out.append("")
        for n in notes:
            out.append(f"- `{n.get('issue_id')}` → {n.get('action')}"
                       f"（{n.get('by')}）：{n.get('reason')}")
        out.append("")

    rev = (d.get("llm_notes") or {}).get("revision") or {}
    if rev.get("edits"):
        out.append("## 六、自动修订留痕")
        out.append("")
        out.append(f"共自动应用 {rev.get('changed', 0)} 处修改。")
        out.append("")
        out.append("| 位置 | 原文 | 改为 | 规则 |")
        out.append("|-----:|------|------|------|")
        for e in rev["edits"]:
            out.append(f"| {e['span'][0]} | `{e['before']}` | `{e['after']}` | {e['rule_id']} |")
        out.append("")

    out.append("## 七、Agent 执行轨迹")
    out.append("")
    out.append("| 时间 | 节点 | 智能体 | 摘要 | 耗时(ms) |")
    out.append("|------|------|--------|------|--------:|")
    for t in d.get("trace", []):
        out.append(f"| {t.get('ts')} | `{t.get('node')}` | {t.get('agent')} | "
                   f"{t.get('summary')} | {t.get('elapsed_ms', '-')} |")
    out.append("")
    return "\n".join(out)


# ======================================================================
# HTML
# ======================================================================
def render_html(report: AuditReport, text: str = "") -> str:
    d = report.to_dict()
    counts = report.counts
    gate_color, gate_label, gate_desc = _GATE[report.release_gate]
    body_text = text or ""

    parts: List[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>智能审校报告｜{html.escape(report.doc_title)}</title>
<style>
:root{{
  --bg:#f7f8fa; --card:#ffffff; --line:#e5e7eb; --tx:#1f2328; --tx2:#57606a;
  --fatal:#cf1322; --major:#d46b08; --minor:#389e0d; --info:#2f54eb; --accent:#0958d9;
  --mono:'Cascadia Mono',Consolas,'Courier New',monospace;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);
  font-family:'Segoe UI','Microsoft YaHei',system-ui,-apple-system,sans-serif;
  line-height:1.7;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 24px 80px}}
header{{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:26px 30px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
h1{{margin:0 0 6px;font-size:23px;font-weight:650;letter-spacing:.2px}}
.sub{{color:var(--tx2);font-size:13px}}
.gate{{display:inline-flex;align-items:center;gap:8px;margin-top:14px;padding:7px 16px;
  border-radius:999px;font-weight:650;font-size:14px;color:#fff;background:{gate_color}}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;margin:18px 0 0}}
.kpi{{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.kpi .n{{font-size:22px;font-weight:680;font-variant-numeric:tabular-nums}}
.kpi .l{{font-size:12px;color:var(--tx2);margin-top:2px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 26px;margin-top:18px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
h2{{font-size:16px;margin:0 0 14px;padding-bottom:9px;border-bottom:1px solid var(--line);
  font-weight:650}}
h3{{font-size:14px;margin:20px 0 10px;color:var(--tx2);font-weight:650}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--tx2);font-weight:600;background:#fafbfc}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.bar{{height:7px;border-radius:4px;background:#eef0f3;overflow:hidden;min-width:80px}}
.bar>i{{display:block;height:100%;border-radius:4px}}
.issue{{border:1px solid var(--line);border-left-width:4px;border-radius:9px;
  padding:13px 16px;margin-bottom:10px;background:#fff}}
.issue .top{{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap}}
.chip{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;color:#fff;
  letter-spacing:.3px}}
.chip.FATAL{{background:var(--fatal)}} .chip.MAJOR{{background:var(--major)}}
.chip.MINOR{{background:var(--minor)}} .chip.INFO{{background:var(--info)}}
.msg{{font-weight:620;font-size:14px}}
.meta{{font-size:12px;color:var(--tx2);margin-top:6px}}
.meta code,.mono{{font-family:var(--mono);background:#f2f4f7;padding:1px 5px;border-radius:4px;
  font-size:12px;color:#374151}}
.diff{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:9px;font-size:13px}}
.diff>div{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 11px}}
.diff .h{{font-size:11px;color:var(--tx2);margin-bottom:3px}}
.old{{text-decoration:line-through;text-decoration-color:var(--fatal);color:#8b1a1a}}
.new{{color:#0a6b2f;font-weight:600}}
.tl{{position:relative;padding-left:20px}}
.tl:before{{content:'';position:absolute;left:5px;top:4px;bottom:4px;width:2px;background:var(--line)}}
.tl .item{{position:relative;margin-bottom:12px;font-size:13px}}
.tl .item:before{{content:'';position:absolute;left:-19px;top:6px;width:9px;height:9px;
  border-radius:50%;background:var(--accent);border:2px solid #fff;box-shadow:0 0 0 1px var(--line)}}
.tl .t{{color:var(--tx2);font-size:12px}}
pre.manuscript{{white-space:pre-wrap;word-break:break-word;font-family:'Microsoft YaHei',sans-serif;
  font-size:14.5px;line-height:2.05;background:#fff;border:1px dashed var(--line);
  border-radius:10px;padding:20px 22px;margin:0}}
mark{{border-radius:3px;padding:1px 0;cursor:help;border-bottom:1.5px solid}}
mark.FATAL{{background:#ffdcd6;border-color:var(--fatal)}}
mark.MAJOR{{background:#ffeccc;border-color:var(--major)}}
mark.MINOR{{background:#ddf5d6;border-color:var(--minor)}}
mark.INFO{{background:#dfe8ff;border-color:var(--info)}}
pre.raw{{white-space:pre-wrap;font-family:var(--mono);font-size:12px;background:#f8f9fb;
  border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;max-height:420px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--tx2);margin-bottom:12px}}
.legend span{{display:inline-flex;align-items:center;gap:5px}}
.dot{{width:10px;height:10px;border-radius:3px;display:inline-block}}
.summary{{white-space:pre-wrap;font-family:var(--mono);font-size:12.5px;background:#f8f9fb;
  border:1px solid var(--line);border-radius:9px;padding:14px 16px}}
</style></head><body><div class="wrap">""")

    # ---- header ----
    parts.append(f"""<header>
<h1>{html.escape(report.doc_title)}</h1>
<div class="sub">智能审校报告 · 基于本地部署大语言模型的多 Agent 协作系统</div>
<div class="sub">稿件类型 {html.escape(report.doc_type)}｜目标文体 {html.escape(report.style_preset or '未指定')}
｜术语库版本 <span class="mono">{html.escape(report.terminology_version)}</span>
｜推理后端 <span class="mono">{html.escape(report.llm_backend)}</span>
{'（本地模型不可达，本次已降级为纯规则引擎）' if report.degraded else ''}</div>
<div class="gate">● 签发门禁 {gate_label}（{report.release_gate}）· {gate_desc}</div>
<div class="kpis">
  <div class="kpi"><div class="n">{report.overall_score}</div><div class="l">综合评分</div></div>
  <div class="kpi"><div class="n" style="color:var(--fatal)">{counts['FATAL']}</div><div class="l">致命问题</div></div>
  <div class="kpi"><div class="n" style="color:var(--major)">{counts['MAJOR']}</div><div class="l">严重问题</div></div>
  <div class="kpi"><div class="n" style="color:var(--minor)">{counts['MINOR']}</div><div class="l">一般问题</div></div>
  <div class="kpi"><div class="n" style="color:var(--info)">{counts['INFO']}</div><div class="l">提示</div></div>
  <div class="kpi"><div class="n">{report.char_count}</div><div class="l">正文字数</div></div>
</div></header>""")

    # ---- dimensions ----
    parts.append('<section><h2>一、维度评分</h2><table>'
                 '<thead><tr><th>维度</th><th>得分</th><th style="width:32%">分布</th>'
                 '<th class="num">致命</th><th class="num">严重</th><th class="num">一般</th>'
                 '<th class="num">提示</th></tr></thead><tbody>')
    for ds in report.dimension_scores:
        color = ("#389e0d" if ds.score >= 85 else "#d46b08" if ds.score >= 60 else "#cf1322")
        parts.append(f"<tr><td><b>{ds.dimension.value}</b></td>"
                     f"<td class='num'><b>{ds.score}</b></td>"
                     f"<td><div class='bar'><i style='width:{ds.score}%;background:{color}'></i></div></td>"
                     f"<td class='num'>{ds.fatal}</td><td class='num'>{ds.major}</td>"
                     f"<td class='num'>{ds.minor}</td><td class='num'>{ds.info}</td></tr>")
    parts.append("</tbody></table>")

    prof = report.style_metrics or {}
    if prof:
        parts.append("<h3>文体量化画像</h3><table><tbody>")
        for k, v in prof.items():
            parts.append(f"<tr><td style='width:180px'>{html.escape(str(k))}</td>"
                         f"<td><span class='mono'>{html.escape(str(v))}</span></td></tr>")
        parts.append("</tbody></table>")
    plan = d.get("transfer_plan") or []
    if plan:
        parts.append("<h3>风格迁移动作清单</h3><ol style='font-size:13.5px;margin:0;padding-left:22px'>")
        for p in plan:
            parts.append(f"<li><b>{html.escape(str(p.get('动作','')))}</b> — "
                         f"{html.escape(str(p.get('说明','')))}</li>")
        parts.append("</ol>")
    parts.append("</section>")

    # ---- summary ----
    parts.append(f"<section><h2>二、审校结论</h2>"
                 f"<div class='summary'>{html.escape(report.summary)}</div>")
    reason = d.get("route_reason")
    if reason:
        parts.append(f"<div class='meta' style='margin-top:12px'>调度依据："
                     f"{html.escape(str(reason))}</div>")
    parts.append("</section>")

    # ---- manuscript ----
    if body_text:
        parts.append("<section><h2>三、划改稿（按问题定位着色）</h2>"
                     "<div class='legend'>"
                     "<span><i class='dot' style='background:#ffdcd6;"
                     "border:1px solid var(--fatal)'></i>致命</span>"
                     "<span><i class='dot' style='background:#ffeccc;"
                     "border:1px solid var(--major)'></i>严重</span>"
                     "<span><i class='dot' style='background:#ddf5d6;"
                     "border:1px solid var(--minor)'></i>一般</span>"
                     "<span><i class='dot' style='background:#dfe8ff;"
                     "border:1px solid var(--info)'></i>提示</span></div>")
        parts.append("<pre class='manuscript'>" + _highlight(body_text, report.issues) + "</pre></section>")

    # ---- issues ----
    parts.append("<section><h2>四、问题清单</h2>")
    for cat, items in report.by_category().items():
        parts.append(f"<h3>{html.escape(cat)}（{len(items)} 条）</h3>")
        for issue in items:
            parts.append(_issue_html(issue))
    parts.append("</section>")

    # ---- fact check ----
    fact = d.get("fact_check") or []
    if fact:
        parts.append("<section><h2>五、数据与事实复算记录</h2><table><thead><tr>"
                     "<th>规则</th><th>复算过程</th><th>结论</th><th class='num'>置信度</th>"
                     "<th>信源</th></tr></thead><tbody>")
        for r in fact:
            v = r.get("verdict")
            color = {"confirmed_error": "var(--fatal)", "needs_human": "var(--major)"}.get(v, "var(--tx2)")
            parts.append(f"<tr><td><span class='mono'>{html.escape(str(r.get('issue_rule')))}</span></td>"
                         f"<td>{html.escape(str(r.get('formula','')))}</td>"
                         f"<td style='color:{color};font-weight:600'>{html.escape(str(v))}</td>"
                         f"<td class='num'>{r.get('confidence')}</td>"
                         f"<td>{html.escape(str(r.get('source','')))}</td></tr>")
        parts.append("</tbody></table></section>")

    # ---- critic ----
    notes = d.get("critic_notes") or []
    if notes:
        parts.append("<section><h2>六、批评修正记录（误报压制）</h2><table><thead><tr>"
                     "<th>问题 ID</th><th>动作</th><th>执行者</th><th>理由</th>"
                     "</tr></thead><tbody>")
        for n in notes:
            parts.append(f"<tr><td><span class='mono'>{html.escape(str(n.get('issue_id')))}</span></td>"
                         f"<td>{html.escape(str(n.get('action')))}</td>"
                         f"<td>{html.escape(str(n.get('by')))}</td>"
                         f"<td>{html.escape(str(n.get('reason')))}</td></tr>")
        parts.append("</tbody></table></section>")

    # ---- revision ----
    revision = (d.get("llm_notes") or {}).get("revision") or {}
    if revision:
        parts.append("<section><h2>七、自动修订留痕</h2>")
        parts.append(f"<div class='meta'>自动应用 <b>{revision.get('changed', 0)}</b> 处修改，"
                     f"保留 <b>{len(revision.get('skipped', []))}</b> 处高风险问题交由人工处理。</div>")
        if revision.get("edits"):
            parts.append("<table style='margin-top:12px'><thead><tr><th class='num'>位置</th>"
                         "<th>原文</th><th>改为</th><th>规则</th></tr></thead><tbody>")
            for e in revision["edits"]:
                parts.append(f"<tr><td class='num'>{e['span'][0]}</td>"
                             f"<td class='old'>{html.escape(str(e['before']))}</td>"
                             f"<td class='new'>{html.escape(str(e['after']))}</td>"
                             f"<td><span class='mono'>{html.escape(str(e['rule_id']))}</span></td></tr>")
            parts.append("</tbody></table>")
        if revision.get("revised_preview"):
            parts.append("<h3>修订稿预览</h3><pre class='raw'>"
                         + html.escape(str(revision["revised_preview"])) + "</pre>")
        parts.append("</section>")

    # ---- trace ----
    parts.append("<section><h2>八、Agent 执行轨迹</h2><div class='tl'>")
    for t in d.get("trace", []):
        ms = t.get("elapsed_ms")
        parts.append(f"<div class='item'><div><b>{html.escape(str(t.get('agent')))}</b> "
                     f"<span class='mono'>{html.escape(str(t.get('node')))}</span></div>"
                     f"<div class='t'>{html.escape(str(t.get('ts')))}"
                     f"{f'｜{ms} ms' if ms is not None else ''}"
                     f"{'｜本轮未激活' if t.get('skipped') else ''}</div>"
                     f"<div>{html.escape(str(t.get('summary','')))}</div></div>")
    parts.append("</div></section>")

    parts.append("</div></body></html>")
    return "".join(parts)


def _issue_html(issue: Issue) -> str:
    bg, fg, _ = _SEV_COLOR[issue.severity.value]
    parts = [f"<div class='issue' style='border-left-color:{fg};background:{bg}'>",
             "<div class='top'>",
             f"<span class='chip {issue.severity.value}'>{issue.severity.label}</span>",
             f"<span class='msg'>{html.escape(issue.message)}</span>", "</div>",
             f"<div class='meta'>第 {issue.line} 行｜第 {issue.paragraph + 1} 段｜"
             f"字符 {issue.span.start}–{issue.span.end}｜规则 "
             f"<code>{html.escape(issue.rule_id)}</code>｜置信度 {issue.confidence}｜"
             f"Agent <code>{html.escape(issue.agent)}</code></div>"]
    if issue.original:
        parts.append("<div class='diff'>")
        parts.append(f"<div><div class='h'>原文</div><span class='old'>"
                     f"{html.escape(issue.original)}</span></div>")
        parts.append(f"<div><div class='h'>建议</div><span class='new'>"
                     f"{html.escape(issue.suggestion)}</span></div>")
        parts.append("</div>")
    elif issue.suggestion:
        parts.append(f"<div class='meta'>建议：<span class='new'>"
                     f"{html.escape(issue.suggestion)}</span></div>")
    if issue.explain:
        parts.append(f"<div class='meta'>依据：{html.escape(issue.explain)}</div>")
    if issue.source:
        parts.append(f"<div class='meta'>出处：{html.escape(issue.source)}</div>")
    parts.append("</div>")
    return "".join(parts)


def highlight_manuscript(text: str, issues: List[Issue], with_tooltip: bool = True) -> str:
    """把原稿按问题区间着色，返回 HTML 片段（供 Web 前端复用划改稿视图）。"""
    return _highlight(text, issues, with_tooltip)


def issue_to_html(issue: Issue) -> str:
    """单条问题的卡片 HTML（供 Web 前端复用）。"""
    return _issue_html(issue)


def severity_color(sev: "Severity") -> str:
    """严重度 -> 主色（供图表与徽标统一取色）。"""
    return _SEV_COLOR[sev.value][1]


def _highlight(text: str, issues: List[Issue], with_tooltip: bool = True) -> str:
    """把原稿按问题区间着色。重叠区间按严重度优先，只渲染一次。"""
    spans: List[tuple] = []
    for i in issues:
        if i.span.length <= 0 or i.span.end > len(text):
            continue
        spans.append((i.span.start, i.span.end, i.severity.value, i.message,
                      i.suggestion, i.rule_id))
    spans.sort(key=lambda x: (x[0], -x[1]))
    picked: List[tuple] = []
    last_end = -1
    for s in spans:
        if s[0] < last_end:
            continue
        picked.append(s)
        last_end = s[1]

    out: List[str] = []
    cursor = 0
    for start, end, sev, msg, sug, rule in picked:
        out.append(html.escape(text[cursor:start]))
        tip = f"[{sev}] {msg}" + (f" → 建议：{sug}" if sug else "") + f"（{rule}）"
        attr = f" title='{html.escape(tip)}'" if with_tooltip else ""
        out.append(f"<mark class='{sev}'{attr}>"
                   f"{html.escape(text[start:end])}</mark>")
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


# ======================================================================
# 输出
# ======================================================================
def write_reports(report: AuditReport, text: str, out_dir: str | Path,
                  formats: Optional[List[str]] = None,
                  stem: str = "audit_report") -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = formats or ["markdown", "html", "json"]
    written: Dict[str, Path] = {}
    if "markdown" in formats:
        p = out / f"{stem}.md"
        p.write_text(render_markdown(report, text), encoding="utf-8")
        written["markdown"] = p
    if "html" in formats:
        p = out / f"{stem}.html"
        p.write_text(render_html(report, text), encoding="utf-8")
        written["html"] = p
    if "json" in formats:
        p = out / f"{stem}.json"
        p.write_text(render_json(report), encoding="utf-8")
        written["json"] = p
    return written
