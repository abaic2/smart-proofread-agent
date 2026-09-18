"""五个专业审校 Agent。

分工与人类校审团队一一对应：

| Agent              | 对应角色       | 职责                                     |
|--------------------|----------------|------------------------------------------|
| TerminologyAgent   | 政治把关编辑   | 术语精准性、固定表述、组合提法顺序        |
| LanguageAgent      | 文字编辑       | 错别字、标点、数字、冗余、可读性          |
| LogicAgent         | 逻辑审稿人     | 推论越界、因果跳步、绝对化、指代不明      |
| AcademicAgent      | 专业审稿人     | 数据自洽、格式、重复发表、参考文献        |
| StyleAgent         | 体裁编辑       | 文体量化画像与风格迁移动作清单            |

每个 Agent 都由"确定性规则引擎 + 可选的本地小模型复核"两层构成：
规则层负责召回与可解释性，模型层负责规则覆盖不到的语义边缘案例。
模型不可用时自动降级，不影响主流程。
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..llm import EmbeddingClient
from ..report import Category, Dimension, Issue, Severity, Span, make_issue
from .base import BaseAgent, EngineAgent


class TerminologyAgent(EngineAgent):
    name = "terminology_agent"
    label = "政治术语审校 Agent"
    role = "政治把关：中央政策术语、固定表述、组合提法顺序"
    engine_key = "terminology"
    requires: List[str] = []

    def __init__(self, settings: Any, store: Any, llm: Any, embedding: Any = None) -> None:
        from ..engines import TermEngine
        engine = TermEngine(settings, store, embedding or EmbeddingClient(
            (settings.llm.get("embedding") or {}), settings.llm))
        super().__init__(settings, store, llm, engine)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        patch = super().execute(state)
        res = patch["engine_results"]["terminology"]

        # ⸺ LLM 复核：对 L3 语义命中等"需要判断"的条目做二次确认
        uncertain = [i for i in res.issues if i.meta.get("level") in ("L3", "L2")
                     and i.confidence < 0.92]
        if uncertain and self.llm_available(state):
            verdicts = self.ask_llm_json(
                state,
                system=("你是中央政策文本审校专家。判断每个候选是否确实构成政治术语使用错误。"
                        "只输出 JSON：{\"verdicts\":[{\"idx\":0,\"is_error\":true,"
                        "\"severity\":\"FATAL|MAJOR|MINOR|INFO\",\"reason\":\"...\","
                        "\"suggestion\":\"...\"}]}"),
                user=self._batch_prompt(uncertain),
                fallback={"verdicts": []},
            )
            patch["llm_review"] = {"terminology_verdicts": verdicts}
            patch["issues"] = self._apply_verdicts(uncertain, verdicts, res.issues)
            patch["engine_results"] = {
                "terminology": res,
                "terminology_llm": {"verdicts": verdicts, "reviewed": len(uncertain)},
            }
        return patch

    @staticmethod
    def _batch_prompt(issues: List[Issue]) -> str:
        lines = []
        for k, i in enumerate(issues):
            lines.append(f"{k}. 原文片段：「{i.original}」｜系统建议：「{i.suggestion}」"
                         f"｜规则说明：{i.explain[:100]}")
        return "请逐条判断：\n" + "\n".join(lines)

    def _apply_verdicts(self, uncertain: List[Issue], verdicts: Dict[str, Any],
                        all_issues: List[Issue]) -> List[Issue]:
        vlist = verdicts.get("verdicts") or []
        by_idx = {int(v.get("idx", -1)): v for v in vlist if isinstance(v, dict)}
        out: List[Issue] = [i for i in all_issues if i not in uncertain]
        for k, issue in enumerate(uncertain):
            v = by_idx.get(k)
            if v is None:
                issue.severity = Severity.INFO        # 模型未能判断 → 降级为提示
                issue.message = "[待人工确认] " + issue.message
                out.append(issue)
                continue
            if v.get("is_error") is False:
                continue                              # 模型判定为误报 → 丢弃
            sev = str(v.get("severity", "MINOR")).upper()
            if sev in Severity.__members__:
                issue.severity = Severity(sev)
            if v.get("suggestion"):
                issue.suggestion = str(v["suggestion"])
            issue.explain = f"{issue.explain}\n模型复核意见：{v.get('reason', '')}"
            issue.confidence = min(0.99, issue.confidence + 0.05)
            issue.agent = "terminology_agent+llm"
            out.append(issue)
        return out

    def summarize(self, patch: Dict[str, Any], state: Dict[str, Any]) -> str:
        base = super().summarize(patch, state)
        if patch.get("llm_review", {}).get("terminology_verdicts"):
            return base + "｜已对低置信度条目做模型复核"
        return base


class LanguageAgent(EngineAgent):
    name = "language_agent"
    label = "语文规范审校 Agent"
    role = "语言文字：字词、标点、数字、冗余、可读性"
    engine_key = "language"

    def __init__(self, settings: Any, store: Any, llm: Any) -> None:
        from ..engines import NormEngine
        super().__init__(settings, store, llm, NormEngine(settings, store))


class LogicAgent(EngineAgent):
    name = "logic_agent"
    label = "逻辑严密性评估 Agent"
    role = "论证逻辑：推论越界、因果跳步、绝对化、指代不明"
    engine_key = "logic"

    def __init__(self, settings: Any, store: Any, llm: Any) -> None:
        from ..engines import LogicEngine
        super().__init__(settings, store, llm, LogicEngine(settings, store))

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        patch = super().execute(state)
        # 对 MAJOR 级逻辑缺陷，用本地模型生成"改写示范"，降低编辑的修改成本
        majors = [i for i in patch["issues"] if i.severity in (Severity.MAJOR, Severity.FATAL)][:3]
        if majors and self.llm_available(state):
            demo = self.ask_llm_text(
                state,
                system=("你是学术与公文写作专家。针对给出的逻辑缺陷，重写该句使其严谨，"
                        "只输出改写后的句子，不要解释。"),
                user="\n\n".join(f"原句：{i.original}\n缺陷：{i.message}" for i in majors),
            )
            patch["llm_review"] = {"logic_rewrite_demo": demo}
        return patch


class AcademicAgent(EngineAgent):
    name = "academic_agent"
    label = "专业文本深度审校 Agent"
    role = "数据合规、格式规范、重复发表、参考文献"
    engine_key = "academic"

    def __init__(self, settings: Any, store: Any, llm: Any, embedding: Any = None) -> None:
        from ..engines import AcademicEngine
        engine = AcademicEngine(settings, store, embedding or EmbeddingClient(
            (settings.llm.get("embedding") or {}), settings.llm))
        super().__init__(settings, store, llm, engine)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        patch = super().execute(state)
        # 观点级重复发表：与语料库做观点句比对
        claims = self.engine.extract_claims(self.doc(state))
        patch["llm_review"] = {"claims_extracted": len(claims),
                               "claims_sample": [c["text"][:50] for c in claims[:3]]}
        patch["engine_results"]["academic_claims"] = {"count": len(claims)}
        return patch


class StyleAgent(EngineAgent):
    name = "style_agent"
    label = "风格迁移 Agent"
    role = "文体量化画像、风格迁移动作清单"
    engine_key = "style"

    def __init__(self, settings: Any, store: Any, llm: Any, preset: str = "") -> None:
        from ..engines import StyleEngine
        super().__init__(settings, store, llm, StyleEngine(settings, store, preset or None))

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        patch = super().execute(state)
        # 生成风格迁移对照稿（suggest 模式：只给建议，不直接改稿）
        profile = patch.get("llm_review", {}).get("style_profile", {})
        plan = patch.get("llm_review", {}).get("style_transfer_plan", [])
        if plan and self.llm_available(state):
            demo = self.ask_llm_text(
                state,
                system=(f"你是{self.engine.preset_name}文体写作专家。"
                        "根据风格画像与迁移要求，给出 2 条最关键的改写示范，"
                        "格式为「原句 → 改句」，不要其他内容。"),
                user=f"风格画像：{profile}\n迁移要求：{plan}",
            )
            patch["llm_review"]["style_rewrite_demo"] = demo
        return patch
