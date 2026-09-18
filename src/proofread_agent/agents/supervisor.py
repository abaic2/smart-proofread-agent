"""受理预检 Agent 与调度 Supervisor。

IntakeAgent 负责"看清稿件"：解析结构、统计特征、探测依赖可用性，
并生成一份**病灶画像**（defects profile）——用低成本规则快速估计稿件
可能存在的问题分布，作为后续调度的依据。这一步是整条流水线的"分诊台"。

SupervisorAgent 负责"派活"：依据病灶画像与稿件类型，决定激活哪些审校
Agent。短稿或纯学术稿不必跑全部 Agent，既省算力也减少噪声。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import BaseAgent
from .state import AuditState, trace_entry

# 用于病灶画像的廉价特征词
_POLITICAL_HINTS = ["党中央", "习近平总书记", "中国特色社会主义", "党的领导", "两个维护",
                    "四个意识", "战略布局", "总体布局", "新发展理念", "一带一路",
                    "党中央", "国家主席", "中共中央"]
_DATA_HINTS = re.compile(r"\d+(?:\.\d+)?\s*(?:%|万|亿|人|户|家|个|项|次|元|条)")
_LOGIC_HINTS = ["因此", "由此可见", "说明", "表明", "必然", "众所周知", "显而易见",
                "全部", "普遍", "全国"]
_FORMAL_HINTS = ["本文", "研究表明", "实证", "样本", "文献", "显著性", "假设"]


class IntakeAgent(BaseAgent):
    name = "intake_agent"
    label = "受理预检 Agent"
    role = "文档解析、特征统计、依赖探测、病灶画像"

    def execute(self, state: AuditState) -> Dict[str, Any]:
        doc = self.doc(state)
        body = doc.body_text
        profile = {
            "char_count": doc.char_count,
            "paragraph_count": len(doc.paragraphs),
            "political_density": round(
                sum(body.count(w) for w in _POLITICAL_HINTS) / max(1, doc.char_count / 100), 3),
            "data_points": len(_DATA_HINTS.findall(body)),
            "logic_markers": sum(body.count(w) for w in _LOGIC_HINTS),
            "formal_markers": sum(body.count(w) for w in _FORMAL_HINTS),
            "has_references": bool(doc.ref_items),
            "reference_count": len(doc.ref_items),
            "citation_count": len(doc.in_text_citations),
            "quote_spans": len(doc.quote_spans),
            "is_academic": sum(body.count(w) for w in _FORMAL_HINTS) >= 2 or bool(doc.ref_items),
            "in_quotes_ratio": round(
                sum(e - s for s, e in doc.quote_spans) / max(1, len(doc.text)), 3),
        }
        profile["doc_type_guess"] = _guess_doc_type(profile)

        llm = state.get("llm")
        backend = getattr(llm, "backend_name", "none")
        corpus_note = self._corpus_note(state)

        return {
            "defects_profile": profile,
            "llm_review": {
                "intake": {
                    "doc": doc.to_meta(),
                    "profile": profile,
                    "llm_backend": backend,
                    "terminology_version": self.store.version if self.store else "n/a",
                    "corpus": corpus_note,
                }
            },
        }

    def _corpus_note(self, state: AuditState) -> Dict[str, Any]:
        try:
            d = self.settings.path(
                (self.settings.engine.get("academic", {}) or {})
                .get("duplicate", {}).get("internal_corpus", "data/corpus"))
            files = sorted(p.name for p in d.glob("*.md")) if d.exists() else []
            return {"dir": str(d), "files": files}
        except Exception:
            return {}

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        p = patch.get("defects_profile", {})
        return (f"字数 {p.get('char_count')}，段落 {p.get('paragraph_count')}，"
                f"数据点 {p.get('data_points')}，参考文献 {p.get('reference_count')} 条，"
                f"判定文体倾向：{p.get('doc_type_guess')}")


class SupervisorAgent(BaseAgent):
    name = "supervisor_agent"
    label = "调度 Supervisor"
    role = "根据病灶画像与稿件类型编排审校 Agent"

    #: Agent 名称 -> 图节点名
    AGENT_NODES = {
        "terminology": "terminology_agent",
        "language": "language_agent",
        "logic": "logic_agent",
        "academic": "academic_agent",
        "style": "style_agent",
    }

    def execute(self, state: AuditState) -> Dict[str, Any]:
        profile = state.get("defects_profile") or {}
        doc_type = state.get("doc_type", "调研专报")
        route: List[str] = []
        reasons: List[str] = []

        # 政治术语：只要涉及政治语汇或为公文类稿件，一律必跑
        if profile.get("political_density", 0) > 0 or doc_type in ("调研专报", "公文", "新闻通稿"):
            route.append("terminology")
            reasons.append("涉及政治语汇或属公文类稿件 → 激活政治术语 Agent")

        # 语文规范：始终执行
        route.append("language")
        reasons.append("语文规范为底线要求 → 始终执行")

        # 逻辑：有论证标记或篇幅足够
        if profile.get("logic_markers", 0) >= 2 or profile.get("char_count", 0) > 300:
            route.append("logic")
            reasons.append(f"检出 {profile.get('logic_markers', 0)} 处论证标记 → 激活逻辑 Agent")

        # 专业深度：有数据点或参考文献，或为学术稿
        if profile.get("data_points", 0) >= 3 or profile.get("has_references") \
                or doc_type in ("学术期刊", "论文", "研究报告"):
            route.append("academic")
            reasons.append(f"数据点 {profile.get('data_points', 0)} 个、"
                           f"参考文献 {profile.get('reference_count', 0)} 条 → 激活专业深度 Agent")

        # 风格：有明确目标文体时执行
        if state.get("style_preset"):
            route.append("style")
            reasons.append(f"指定目标文体「{state['style_preset']}」→ 激活风格迁移 Agent")

        return {
            "route": route,
            "route_reason": "；".join(reasons),
            "llm_review": {"supervisor": {"route": route, "reasons": reasons,
                                          "doc_type": doc_type}},
        }

    def route_fn(self, state: AuditState) -> List[str]:
        """LangGraph 条件边函数：返回一组节点名即触发并行 fan-out。"""
        route = state.get("route") or self.execute(state)["route"]
        nodes = [self.AGENT_NODES[a] for a in route if a in self.AGENT_NODES]
        return nodes or [self.AGENT_NODES["language"]]

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        return f"激活 Agent：{'、'.join(patch.get('route', []))}"


def _guess_doc_type(profile: Dict[str, Any]) -> str:
    if profile.get("is_academic") and profile.get("reference_count", 0) >= 2:
        return "学术期刊"
    if profile.get("political_density", 0) > 0.15:
        return "公文/专报"
    return "通用文稿"
