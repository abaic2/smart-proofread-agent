"""审校引擎基类与结果契约。

引擎是**纯函数式**的：输入 Document，输出 Issue 列表，不做任何 IO、不调用 LLM。
这样设计的原因：审校规则需要可测试、可复现、可离线运行；LLM 只用于
"引擎无法机判"的语义任务，由 Agent 层负责编排。

五个引擎：
    term_engine      政治术语精准审校
    norm_engine      语文规范审核
    logic_engine     逻辑严密性评估
    academic_engine  专业文本深度审校（数据合规/格式/重复发表/参考文献）
    style_engine     风格迁移与一致性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..document import Document
from ..report import Category, Dimension, Issue, Severity, Span


@dataclass
class EngineResult:
    """引擎输出。"""

    engine: str
    dimension: Dimension
    issues: List[Issue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def extend(self, issues: List[Issue]) -> None:
        self.issues.extend(issues)


class BaseEngine:
    """所有引擎的基类。"""

    name: str = "base"
    dimension: Dimension = Dimension.LANGUAGE
    description: str = ""

    def __init__(self, settings: Any, store: Any = None) -> None:
        self.settings = settings
        self.store = store
        self.cfg: Dict[str, Any] = {}

    # 子类实现
    def run(self, doc: Document) -> EngineResult:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    # ---- 公共工具 --------------------------------------------------
    def result(self, **stats: Any) -> EngineResult:
        return EngineResult(engine=self.name, dimension=self.dimension, stats=stats)

    @staticmethod
    def resolve_overlaps(issues: List[Issue]) -> List[Issue]:
        """区间重叠消解。

        同一条文本可能被多条规则命中（如"五个必由之路"既触发术语规则又触发
        逻辑规则）。策略：优先级 = 严重度权重 → 匹配长度 → 规则特异性，
        保留优先级高者，其余丢弃，避免同一处刷出多条相似提示。
        """
        ordered = sorted(
            issues,
            key=lambda i: (-i.severity.weight, -i.span.length, i.rule_id),
        )
        kept: List[Issue] = []
        for issue in ordered:
            if any(_overlaps(issue.span, k.span) for k in kept):
                continue
            kept.append(issue)
        return sorted(kept, key=lambda i: i.span.start)


def _overlaps(a: Span, b: Span) -> bool:
    if a.length == 0 or b.length == 0:
        return a.start == b.start
    return not (a.end <= b.start or b.end <= a.start)


def severity_from_str(value: str, default: Severity = Severity.MINOR) -> Severity:
    try:
        return Severity(value)
    except Exception:
        return default


def annotate_positions(doc: Document, issues: List[Issue]) -> List[Issue]:
    """回填行号、段号，并**强制统一 original 与 span 的数据契约**。

    约定：``issue.original`` 必须严格等于 ``doc.text[span.start:span.end]``。
    引擎出于可读性考虑常把整句塞进 original，这会导致下游（批评修正、
    自动修订）判定"定位漂移"而误杀正确问题。这里统一收敛：把原文片段
    归一为精确区间文本，原先的整句说明移入 ``meta['sentence']`` 供报告展示。
    """
    for issue in issues:
        pos = max(0, min(issue.span.start, len(doc.text)))
        issue.line = doc.line_of(pos)
        para = doc.paragraph_of(pos)
        issue.paragraph = max(0, para.get("index", -1))

        if issue.span.length > 0 and issue.span.end <= len(doc.text):
            exact = doc.text[issue.span.start:issue.span.end]
            if issue.original and issue.original != exact:
                issue.meta.setdefault("sentence", issue.original)
                issue.original = exact
            elif not issue.original:
                issue.original = exact
    return issues


__all__ = [
    "BaseEngine",
    "EngineResult",
    "Category",
    "Dimension",
    "Issue",
    "Severity",
    "Span",
    "annotate_positions",
    "severity_from_str",
]
