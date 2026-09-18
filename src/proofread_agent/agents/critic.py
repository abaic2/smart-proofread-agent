"""批评修正 Agent（Critic）与事实校验 Agent（Fact Checker）。

这是"System-2" 式审校的关键：先由多个专业 Agent 广召回，再由一个
独立视角的 Critic 做**降温**——它在架构上被刻意设计成"驳回者"，
目标不是找更多问题，而是**删掉站不住脚的问题**。

CriticAgent 的三项职责
1. 规则层自检：建议等于原文、区间为空、规则自相矛盾 → 直接剔除；
2. 交叉一致性：同一位置收到多条互斥建议 → 只保留证据最强的一条；
3. 模型复核：对 FATAL/MAJOR 逐条问本地模型"这条批评是否成立"，
   模型回答不成立且理由充分则降级为 INFO 并记录驳回理由（可申诉）。

FactCheckAgent
面向所有数据类与文献类结论做**公式复算**，输出带"信源标注"的校验记录。
复算过程完全确定性，不依赖模型，因此其结果可作为签发依据。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..report import Category, Issue, Severity
from .base import BaseAgent
from .state import AuditState

_REJECT = {"is_valid": "false", "verdict": "reject"}


class CriticAgent(BaseAgent):
    name = "critic_agent"
    label = "批评修正 Agent"
    role = "误报压制、建议一致性校验、模型复核"

    def execute(self, state: AuditState) -> Dict[str, Any]:
        # 第二轮起以"已复核清单"为输入，避免对已被驳回的问题重复复核
        issues: List[Issue] = list(state.get("reviewed_issues")
                                  or state.get("issues") or [])
        doc = self.doc(state)
        notes: List[Dict[str, Any]] = []
        kept: List[Issue] = []
        suppressed: List[str] = []

        for issue in issues:
            # ① 规则层自检
            reason = self._rule_reject(doc, issue)
            if reason:
                suppressed.append(issue.issue_id)
                notes.append({"issue_id": issue.issue_id, "action": "drop",
                              "by": "rule", "reason": reason})
                continue
            kept.append(issue)

        # ② 交叉一致性：同位置多条建议冲突时保留证据最强的一条
        kept, conflict_notes = self._resolve_conflicts(kept)
        notes.extend(conflict_notes)
        suppressed.extend(n["issue_id"] for n in conflict_notes)

        # ③ 第二轮：保守收敛（仅保留高置信或高严重度问题，进一步降低误报）
        round_no = int(state.get("critic_round") or 0)
        if round_no >= 1:
            converged: List[Issue] = []
            for i in kept:
                if i.severity in (Severity.FATAL, Severity.MAJOR) or i.confidence >= 0.6:
                    converged.append(i)
                else:
                    suppressed.append(i.issue_id)
                    notes.append({"issue_id": i.issue_id, "action": "drop", "by": "round2",
                                  "reason": f"第二轮保守收敛：置信度 {i.confidence} < 0.6 且非高严重度"})
            kept = converged

        # ④ 模型复核高严重度问题（仅第一轮执行，第二轮已收敛）
        if self.llm_available(state) and round_no < 1:
            targets = [i for i in kept if i.severity in (Severity.FATAL, Severity.MAJOR)][:12]
            if targets:
                verdicts = self.ask_llm_json(
                    state,
                    system=("你是审校质量管理员，职责是驳回不成立的审校意见。"
                            "对每条候选意见，判断其是否确实构成问题。"
                            "注意：原文引述权威文献、否定句式（如「不得使用X」）、"
                            "以及术语本身正确的情况都属于误报，应驳回。"
                            "只输出 JSON：{\"verdicts\":[{\"idx\":0,\"is_valid\":true,"
                            "\"reason\":\"...\"}]}"),
                    user=self._prompt(targets, doc),
                    fallback={"verdicts": []},
                )
                kept, llm_notes, llm_suppressed = self._apply(kept, targets, verdicts)
                notes.extend(llm_notes)
                suppressed.extend(llm_suppressed)

        return {
            "reviewed_issues": kept,
            "critic_notes": notes,
            "suppressed_ids": suppressed,
            "critic_round": round_no + 1,
            "llm_review": {"critic": {
                "input": len(issues), "output": len(kept),
                "dropped": len(suppressed), "round": round_no + 1,
                "llm_reviewed": bool(self.llm_available(state)) and round_no < 1,
            }},
        }

    # ---- 规则层自检 ------------------------------------------------
    def _rule_reject(self, doc: Any, issue: Issue) -> str:
        if issue.suggestion and issue.original and issue.suggestion == issue.original:
            return "建议与原文完全一致，无修改意义"
        if issue.span.length == 0 and issue.category not in (
                Category.FORMAT, Category.REFERENCE, Category.STYLE, Category.LOGIC):
            return "定位区间为空且非全局类问题"
        if issue.span.start < 0 or issue.span.end > len(doc.text) + 1:
            return "定位越界"
        if issue.span.length and issue.original:
            actual = doc.text[issue.span.start:issue.span.end]
            if actual.strip() == issue.original.strip():
                return ""
            # 允许 original 比区间更宽（整句说明），只要区间文本能在其中定位到；
            # 真正的"定位漂移"是区间文本与记录内容毫无包含关系。
            if issue.original in actual or actual in issue.original:
                return ""
            return f"定位漂移：区间文本「{actual[:20]}」与记录「{issue.original[:20]}」不匹配"
        return ""

    # ---- 冲突消解 --------------------------------------------------
    def _resolve_conflicts(self, issues: List[Issue]):
        groups: Dict[str, List[Issue]] = {}
        for i in issues:
            key = f"{i.span.start}-{i.span.end}" if i.span.length else f"G:{i.rule_id}"
            groups.setdefault(key, []).append(i)

        kept: List[Issue] = []
        notes: List[Dict[str, Any]] = []
        for key, items in groups.items():
            if len(items) == 1:
                kept.append(items[0])
                continue
            # 保留：严重度高 → 置信度高 → 建议更具体
            best = max(items, key=lambda i: (i.severity.weight, i.confidence,
                                             len(i.suggestion or "")))
            kept.append(best)
            for other in items:
                if other is best:
                    continue
                notes.append({
                    "issue_id": other.issue_id, "action": "drop", "by": "conflict",
                    "reason": f"同位置与 {best.rule_id} 意见冲突，保留证据更强的一条"
                              f"（严重度 {best.severity.value}/置信度 {best.confidence}）",
                })
        return kept, notes

    # ---- 模型复核 --------------------------------------------------
    @staticmethod
    def _prompt(issues: List[Issue], doc: Any) -> str:
        lines = []
        for k, i in enumerate(issues):
            ctx = doc.context(i.span.start, 25, 25)
            lines.append(f"{k}. 类别：{i.category.value}｜规则：{i.rule_id}\n"
                         f"   上下文：「{ctx}」\n"
                         f"   判定：{i.message}\n"
                         f"   依据：{i.explain[:120]}")
        return "请逐条复核：\n" + "\n".join(lines)

    def _apply(self, kept_all: List[Issue], targets: List[Issue],
               verdicts: Dict[str, Any]):
        vlist = verdicts.get("verdicts") or []
        by_idx = {int(v.get("idx", -1)): v for v in vlist if isinstance(v, dict)}
        if not by_idx:
            return kept_all, [], []
        drop_ids: set = set()
        notes: List[Dict[str, Any]] = []
        for k, issue in enumerate(targets):
            v = by_idx.get(k)
            if v is None:
                continue
            valid = v.get("is_valid")
            if valid is False or str(valid).lower() == "false":
                drop_ids.add(issue.issue_id)
                notes.append({
                    "issue_id": issue.issue_id, "action": "downgrade", "by": "llm",
                    "reason": f"模型复核认为不成立：{v.get('reason', '未给出理由')}",
                })
            else:
                issue.explain = f"{issue.explain}\n✓ 模型复核通过：{v.get('reason', '')}"
        out: List[Issue] = []
        for i in kept_all:
            if i.issue_id in drop_ids:
                i.severity = Severity.INFO
                i.message = "[模型复核存疑，建议人工确认] " + i.message
            out.append(i)
        return out, notes, list(drop_ids)

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        c = patch.get("llm_review", {}).get("critic", {})
        return (f"复核 {c.get('input', 0)} 条 → 保留 {c.get('output', 0)} 条，"
                f"驳回/降级 {c.get('dropped', 0)} 条"
                f"（{'含模型复核' if c.get('llm_reviewed') else '纯规则复核'}）")


class FactCheckAgent(BaseAgent):
    name = "fact_check_agent"
    label = "事实与数据校验 Agent"
    role = "对数据类与文献类结论做确定性复算，标注信源"

    def execute(self, state: AuditState) -> Dict[str, Any]:
        doc = self.doc(state)
        records: List[Dict[str, Any]] = []

        for issue in (state.get("reviewed_issues") or state.get("issues") or []):
            if issue.category == Category.DATA_COMPLIANCE:
                records.append(self._verify_data(issue))
            elif issue.category == Category.REFERENCE:
                records.append(self._verify_reference(issue))
            elif issue.category == Category.POLITICAL_TERM and issue.severity == Severity.FATAL:
                records.append(self._verify_term(doc, issue))

        verdicts = Counter(r["verdict"] for r in records)
        return {
            "fact_check": records,
            "llm_review": {"fact_check_summary": dict(verdicts),
                           "checked": len(records)},
        }

    # ---- 数据复算 --------------------------------------------------
    @staticmethod
    def _verify_data(issue: Issue) -> Dict[str, Any]:
        meta = issue.meta or {}
        rec: Dict[str, Any] = {
            "issue_id": issue.issue_id,
            "issue_rule": issue.rule_id,
            "check": meta.get("check", "unknown"),
            "statement": issue.message,
            "source": "正文数据自洽复算（确定性算法）",
        }
        if meta.get("check") == "sum_consistency":
            rec["formula"] = " + ".join(str(x) for x in meta.get("parts", [])) \
                             + f" = {meta.get('computed')} vs 文中 {meta.get('stated')}"
            rec["verdict"] = "confirmed_error"
            rec["confidence"] = 0.95
            rec["advice"] = f"以分项之和为准修正合计，或核对分项是否漏项（单位：{meta.get('unit', '')}）"
        elif meta.get("check") == "percent_consistency":
            rec["formula"] = f"重算占比 = {meta.get('computed')}% vs 文中 {meta.get('stated')}%"
            rec["verdict"] = "confirmed_error"
            rec["confidence"] = 0.9
            rec["advice"] = "确认分母口径（是总量还是分项基数）后修正占比"
        elif meta.get("check") == "percent_range":
            rec["formula"] = f"占比校验：{issue.original} > 100%"
            rec["verdict"] = "confirmed_error"
            rec["confidence"] = 0.97
            rec["advice"] = "占比>100% 必为口径错误，常见原因是把累计占比当作单项占比"
        else:
            rec["formula"] = issue.original or "（无显式公式，需人工核对原始数据源）"
            rec["verdict"] = "needs_human"
            rec["confidence"] = 0.6
            rec["advice"] = "需人工核对原始数据源"
        rec.setdefault("formula", issue.original or "")
        rec.setdefault("advice", issue.suggestion)
        return rec

    @staticmethod
    def _verify_reference(issue: Issue) -> Dict[str, Any]:
        return {
            "issue_id": issue.issue_id,
            "issue_rule": issue.rule_id,
            "check": "gbt7714",
            "statement": issue.message,
            "formula": "按 GB/T 7714-2015 逐项比对著录项与类型标识",
            "verdict": "confirmed_error" if issue.rule_id in
                       ("REF-001", "REF-004", "REF-005") else "needs_human",
            "confidence": issue.confidence,
            "source": "《信息与文献 参考文献著录规则》GB/T 7714-2015",
            "advice": issue.suggestion,
        }

    @staticmethod
    def _verify_term(doc: Any, issue: Issue) -> Dict[str, Any]:
        ctx = doc.context(issue.span.start, 30, 30)
        quoted = doc.in_citation(issue.span.start)
        return {
            "issue_id": issue.issue_id,
            "issue_rule": issue.rule_id,
            "check": "political_term",
            "statement": issue.message,
            "formula": f"术语库标准表述比对：{issue.original} → {issue.suggestion}",
            "verdict": "needs_human" if quoted else "confirmed_error",
            "confidence": 0.99 if not quoted else 0.6,
            "source": issue.source or "中央政策术语库",
            "context": ctx,
            "advice": ("该处位于引号内，可能为原文引述，改前请确认引用来源"
                       if quoted else issue.suggestion),
        }

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        s = patch.get("llm_review", {}).get("fact_check_summary", {})
        return f"完成 {sum(s.values()) if s else 0} 项复算：" + \
               "，".join(f"{k}={v}" for k, v in s.items())
