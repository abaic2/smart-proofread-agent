"""报告聚合 Agent。

把并行 Agent 的产出汇聚成一份可签发（或明确阻断）的审校报告：

* **维度评分**：每个维度按下式折算，避免"问题多就一律零分"的退化：
  ``score = 100 · exp(−penalty / scale)``，penalty 为该维度所有问题的
  严重度权重之和。指数衰减保证：零问题=100 分，单个 FATAL 显著拉低，
  但不会因为一处提示性建议就归零。
* **综合评分**：各维度加权（政治权重最高，因为它是"一票否决"维度）。
* **签发门禁**：出现任一 FATAL → BLOCK（不予签发）；综合分 < 75 或
  MAJOR 超阈值 → REVIEW（退改）；否则 PASS。
* **摘要生成**：优先由本地模型撰写面向编辑的自然语言总结，
  模型不可用时使用模板化摘要（结构化、不丢信息）。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from ..engines.base import annotate_positions
from ..report import AuditReport, Category, Dimension, DimensionScore, Issue, Severity
from .base import BaseAgent
from .state import AuditState, trace_entry

# 维度 -> 严重度权重衰减尺度
_SCALE = {
    Dimension.POLITICAL: 150.0,
    Dimension.LANGUAGE: 130.0,
    Dimension.LOGIC: 200.0,
    Dimension.ACADEMIC: 200.0,
    Dimension.STYLE: 120.0,
}
# 维度 -> 综合权重
_WEIGHT = {
    Dimension.POLITICAL: 0.35,
    Dimension.LANGUAGE: 0.18,
    Dimension.LOGIC: 0.17,
    Dimension.ACADEMIC: 0.25,
    Dimension.STYLE: 0.05,
}
_CATEGORY_DIM = {
    Category.POLITICAL_TERM: Dimension.POLITICAL,
    Category.LANGUAGE_NORM: Dimension.LANGUAGE,
    Category.FORMAT: Dimension.LANGUAGE,
    Category.LOGIC: Dimension.LOGIC,
    Category.DATA_COMPLIANCE: Dimension.ACADEMIC,
    Category.DUPLICATE: Dimension.ACADEMIC,
    Category.REFERENCE: Dimension.ACADEMIC,
    Category.STYLE: Dimension.STYLE,
}


class ReportAgent(BaseAgent):
    name = "report_agent"
    label = "报告聚合 Agent"
    role = "去重、维度评分、签发门禁、报告生成"

    def execute(self, state: AuditState) -> Dict[str, Any]:
        import math

        doc = self.doc(state)
        issues = self._dedup(list(state.get("reviewed_issues")
                                  or state.get("issues") or []))
        suppressed = set(state.get("suppressed_ids") or [])
        issues = [i for i in issues
                  if not (i.severity == Severity.INFO and i.issue_id in suppressed)]
        annotate_positions(doc, issues)

        # ---- 维度聚合 ----
        by_dim: Dict[Dimension, List[Issue]] = {}
        for i in issues:
            by_dim.setdefault(_CATEGORY_DIM.get(i.category, Dimension.LANGUAGE), []).append(i)

        dim_scores: List[DimensionScore] = []
        for dim in [Dimension.POLITICAL, Dimension.LANGUAGE, Dimension.LOGIC,
                    Dimension.ACADEMIC, Dimension.STYLE]:
            items = by_dim.get(dim, [])
            penalty = sum(i.severity.weight for i in items)
            score = 100.0 * math.exp(-penalty / _SCALE[dim]) if penalty else 100.0
            c = Counter(i.severity.value for i in items)
            dim_scores.append(DimensionScore(
                dimension=dim, score=round(score, 1),
                fatal=c.get("FATAL", 0), major=c.get("MAJOR", 0),
                minor=c.get("MINOR", 0), info=c.get("INFO", 0), total=len(items),
            ))

        overall = sum(d.score * _WEIGHT[d.dimension] for d in dim_scores)

        # ---- 门禁 ----
        counts = Counter(i.severity.value for i in issues)
        if counts.get("FATAL", 0) > 0:
            gate = "BLOCK"
        elif overall < 75.0 or counts.get("MAJOR", 0) > 5:
            gate = "REVIEW"
        else:
            gate = "PASS"

        # ---- 风格画像 ----
        style_stats = ((state.get("engine_results") or {}).get("style") or {})
        profile = getattr(style_stats, "stats", {}).get("profile", {}) if style_stats else {}
        if not profile:
            profile = (state.get("llm_review") or {}).get("style_profile", {}) or {}
        style_score = (getattr(style_stats, "stats", {}) or {}).get("style_score", 100.0) \
            if style_stats else 100.0

        summary = self._summarize(state, issues, dim_scores, overall, gate, counts)

        # 报告自身的 trace 条目需在此显式追加：BaseAgent 是在本方法返回之后
        # 才写入 trace 的，若只取 state["trace"] 会缺少最后一个节点。
        trace = list(state.get("trace") or [])
        trace.append(trace_entry(
            self.name, self.label,
            f"综合 {overall:.1f} 分，门禁 {gate}，共 {len(issues)} 条问题",
        ))

        report = AuditReport(
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_type=state.get("doc_type", "调研专报"),
            char_count=doc.char_count,
            paragraph_count=len(doc.paragraphs),
            issues=issues,
            dimension_scores=dim_scores,
            overall_score=round(overall, 1),
            release_gate=gate,
            summary=summary,
            style_preset=state.get("style_preset", ""),
            style_metrics=profile,
            degraded=getattr(state.get("llm"), "backend_name", "mock") == "mock",
            llm_backend=getattr(state.get("llm"), "backend_name", "none"),
            terminology_version=self.store.version if self.store else "",
            trace=trace,
        )
        # 补充元信息（事实校验、批评意见、调度信息、风格迁移清单）
        report.meta = {
            "fact_check": state.get("fact_check") or [],
            "critic_notes": state.get("critic_notes") or [],
            "route": state.get("route") or [],
            "route_reason": state.get("route_reason", ""),
            "defects_profile": state.get("defects_profile") or {},
            "style_score": style_score,
            "transfer_plan": (state.get("llm_review") or {}).get("style_transfer_plan", []),
            "llm_notes": {
                k: v for k, v in (state.get("llm_review") or {}).items()
                if k in ("logic_rewrite_demo", "style_rewrite_demo", "terminology_verdicts",
                         "intake", "critic", "academic", "supervisor", "fact_check_summary",
                         "revision", "revision_diff", "join", "style_profile",
                         "style_transfer_plan", "style_rewrite_note")
            },
            "llm_stats": getattr(state.get("llm"), "stats", {}),
        }
        return {"report": report, "issues": [],
                "llm_review": {"report_preview": f"{len(issues)} 条问题 / 综合 {round(overall,1)} 分"}}

    # ---- 去重 -------------------------------------------------------
    @staticmethod
    def _dedup(issues: List[Issue]) -> List[Issue]:
        seen: Dict[str, Issue] = {}
        for i in issues:
            fp = i.fingerprint
            if fp in seen:
                if i.severity.weight > seen[fp].severity.weight:
                    seen[fp] = i
                continue
            seen[fp] = i
        return sorted(seen.values(), key=lambda x: (-x.severity.weight, x.span.start))

    # ---- 摘要 -------------------------------------------------------
    def _summarize(self, state: AuditState, issues: List[Issue],
                   dims: List[DimensionScore], overall: float, gate: str,
                   counts: Counter) -> str:
        head = {
            "BLOCK": "存在政治性/原则性问题，不予签发。",
            "REVIEW": "存在需要修改的问题，建议退改后再审。",
            "PASS": "未发现阻断性问题，可通过审校。",
        }[gate]
        top = [i for i in issues if i.severity in (Severity.FATAL, Severity.MAJOR)][:8]
        lines = [
            f"【结论】{head}",
            f"【评分】综合 {overall:.1f} 分｜"
            + "｜".join(f"{d.dimension.value} {d.score}" for d in dims),
            f"【统计】致命 {counts.get('FATAL', 0)}、严重 {counts.get('MAJOR', 0)}、"
            f"一般 {counts.get('MINOR', 0)}、提示 {counts.get('INFO', 0)}",
        ]
        if top:
            lines.append("【必须修改】")
            for i in top:
                lines.append(f"  · [{i.severity.label}][{i.category.value}] {i.message}")

        if self.llm_available(state):
            narrative = self.ask_llm_text(
                state,
                system=("你是审校报告撰写人。基于结构化审校结果，写一段面向编辑的"
                        "中文修改建议（不超过 200 字），按优先级排序，不要复述数据。"),
                user="\n".join(lines),
            )
            if narrative:
                lines.append(f"【修改建议】{narrative.strip()}")
        return "\n".join(lines)

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        rep = patch.get("report")
        if not rep:
            return "未生成报告"
        return (f"综合 {rep.overall_score} 分，门禁 {rep.release_gate}，"
                f"共 {len(rep.issues)} 条问题")
