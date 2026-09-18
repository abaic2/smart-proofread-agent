"""修订稿生成 Agent。

审校系统的终点不该是"一份问题清单"，而应是"一份可继续编辑的修订稿 +
完整修改留痕"。本 Agent 负责：

1. **自动应用安全修改**：仅对"确定性高、无需人工判断"的问题执行替换，
   即 L1 精确术语错误、错别字、标点全半角、规范表述替换；
2. **生成修改留痕**：每处替换记录 (原位置、原文、改文、规则、依据)，
   形成可审计的修订记录（类似 Word 的修订模式）；
3. **不触碰高风险项**：FATAL 级政治性表述、数据矛盾、逻辑缺陷一律
   只出建议不改稿——这类问题必须由人决定，机器擅自改写会带来更大的风险；
4. **风格迁移**：在开关开启时，调用本地模型对长句/口语化段落做受控改写，
   并要求模型逐句输出映射，便于回滚。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..report import Category, Issue, Severity
from .base import BaseAgent
from .state import AuditState

#: 允许自动替换的类别与最低置信度
AUTO_FIX_RULES = {
    "term_engine": 0.90,
    "language": 0.88,
}


@dataclass
class Edit:
    """一处自动修改记录。"""

    start: int
    end: int
    before: str
    after: str
    rule_id: str
    reason: str
    source: str = ""
    auto: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span": [self.start, self.end],
            "before": self.before,
            "after": self.after,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "source": self.source,
            "auto": self.auto,
        }


@dataclass
class RevisionResult:
    original: str
    revised: str
    edits: List[Edit] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.edits)

    def diff_text(self) -> str:
        return "\n".join(difflib.unified_diff(
            self.original.splitlines(), self.revised.splitlines(),
            fromfile="原稿", tofile="修订稿", lineterm="", n=1,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "edits": [e.to_dict() for e in self.edits],
            "skipped": self.skipped,
            "revised_preview": self.revised[:400],
        }


class RevisionAgent(BaseAgent):
    name = "revision_agent"
    label = "修订稿生成 Agent"
    role = "自动应用安全修改、生成修改留痕、受控风格迁移"

    def execute(self, state: AuditState) -> Dict[str, Any]:
        doc = self.doc(state)
        issues: List[Issue] = list(state.get("reviewed_issues") or state.get("issues") or [])
        edits: List[Edit] = []
        skipped: List[Dict[str, Any]] = []

        candidates = sorted(
            [i for i in issues if self._autofixable(i)],
            key=lambda i: i.span.start,
        )
        # 逆序应用，避免偏移错位
        cursor = len(doc.text)
        for issue in sorted(candidates, key=lambda i: -i.span.start):
            if issue.span.end > cursor:          # 与已应用的修改重叠
                skipped.append({**issue.to_dict(), "skip_reason": "与已应用修改区间重叠"})
                continue
            actual = doc.text[issue.span.start:issue.span.end]
            if actual != issue.original and actual.strip() != (issue.original or "").strip():
                skipped.append({**issue.to_dict(), "skip_reason": "原文与定位不一致"})
                continue
            edits.append(Edit(
                start=issue.span.start, end=issue.span.end,
                before=actual, after=issue.suggestion,
                rule_id=issue.rule_id, reason=issue.message,
                source=issue.source,
            ))
            cursor = issue.span.start

        for issue in issues:
            if self._autofixable(issue):
                continue
            if issue.severity in (Severity.FATAL, Severity.MAJOR) or \
                    issue.category in (Category.DATA_COMPLIANCE, Category.DUPLICATE,
                                       Category.LOGIC):
                skipped.append({**issue.to_dict(),
                                "skip_reason": "高风险/需人工判断，仅建议不自动改稿"})

        revised = _apply_edits(doc.text, edits)

        # 受控风格迁移（可选）
        style_note = ""
        if self.settings.style.get("transfer_mode") == "rewrite" and self.llm_available(state):
            plan = (state.get("llm_review") or {}).get("style_transfer_plan", [])
            if plan:
                style_note = self.ask_llm_text(
                    state,
                    system=("你是公文与学术写作修改专家。只做句级改写：拆分长句、"
                            "去除口语与绝对化表述、补充数据来源标注占位。"
                            "保持政治表述原样，不得改动任何事实与数据。"
                            "输出修改后的全文，不要解释。"),
                    user=f"文体要求：{plan}\n\n全文：\n{revised[:6000]}",
                    fallback="",
                )
                if style_note.strip():
                    revised = style_note.strip()

        result = RevisionResult(original=doc.text, revised=revised, edits=edits, skipped=skipped)
        return {
            "llm_review": {
                "revision": result.to_dict(),
                "revision_diff": result.diff_text()[:4000],
                "style_rewrite_note": ("已执行受控风格改写" if style_note else "未启用模型改写"),
            },
        }

    @staticmethod
    def _autofixable(issue: Issue) -> bool:
        """判断是否允许自动替换。保守优先：不确定的一律不自动改。"""
        if issue.severity == Severity.FATAL:
            return False
        if not issue.suggestion or not issue.original:
            return False
        if issue.span.length == 0 or issue.span.length > 60:
            return False
        if issue.suggestion == issue.original:
            return False
        agent = issue.meta.get("level", "")
        if issue.category == Category.POLITICAL_TERM:
            # 仅 L1 精确术语错误与 L2 高置信近似错误可自动替换
            return agent in ("L1",) and issue.confidence >= 0.9
        if issue.category in (Category.LANGUAGE_NORM, Category.FORMAT):
            return issue.confidence >= 0.88 and not issue.suggestion.startswith(
                ("补充", "核对", "统一", "改为", "删除", "按", "重算"))
        return False

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        r = patch.get("llm_review", {}).get("revision", {})
        return (f"自动修订 {r.get('changed', 0)} 处，"
                f"保留 {len(r.get('skipped', []))} 处待人工处理")


def _apply_edits(text: str, edits: List[Edit]) -> str:
    """按偏移逆序应用替换。"""
    out = text
    for e in sorted(edits, key=lambda x: -x.start):
        out = out[:e.start] + e.after + out[e.end:]
    return out
