"""审校问题模型：全流程统一的数据契约。

设计要点
--------
* 使用 dataclass 而非 pydantic：零外部依赖，LangGraph 状态可直接序列化。
* ``Span`` 精确到字符偏移，保证前端"原稿划改"可以精确定位与一键替换。
* ``Issue`` 携带 ``evidence``/``rule_id``/``source``，满足审校结果可追溯、
  可申诉（编辑可复核规则依据）的业务要求。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """严重级别。FATAL 用于政治性错误，必须阻断签发。"""

    FATAL = "FATAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"

    @property
    def label(self) -> str:
        return {"FATAL": "致命", "MAJOR": "严重", "MINOR": "一般", "INFO": "提示"}[self.value]

    @property
    def weight(self) -> int:
        return {"FATAL": 100, "MAJOR": 20, "MINOR": 5, "INFO": 1}[self.value]


class Category(str, Enum):
    POLITICAL_TERM = "政治术语"
    LANGUAGE_NORM = "语文规范"
    LOGIC = "逻辑严密性"
    DATA_COMPLIANCE = "数据合规"
    FORMAT = "格式规范"
    DUPLICATE = "重复发表"
    REFERENCE = "参考文献"
    STYLE = "风格适配"


class Dimension(str, Enum):
    """维度中文名，用于报告分组。"""

    POLITICAL = "政治术语精准性"
    LANGUAGE = "语文规范性"
    LOGIC = "逻辑严密性"
    ACADEMIC = "专业文本严谨性"
    STYLE = "风格一致性"


@dataclass
class Span:
    """原稿定位：字符级偏移，start 含、end 不含。"""

    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


@dataclass
class Issue:
    """一条审校问题。"""

    issue_id: str
    category: Category
    severity: Severity
    rule_id: str
    message: str
    span: Span
    original: str = ""
    suggestion: str = ""
    explain: str = ""
    evidence: str = ""
    source: str = ""
    confidence: float = 1.0
    agent: str = ""
    paragraph: int = 0
    line: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- 序列化 --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        d["severity_label"] = self.severity.label
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @property
    def fingerprint(self) -> str:
        """同位置同规则视为同一问题，用于去重与跨版本对比。"""
        key = f"{self.rule_id}|{self.span.start}|{self.span.end}|{self.original}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def make_issue(category: Category, severity: Severity, rule_id: str, message: str,
               span: Span, **kwargs: Any) -> Issue:
    """工厂函数，自动生成稳定 ID。"""
    iid = hashlib.md5(
        f"{rule_id}|{span.start}|{span.end}".encode("utf-8")
    ).hexdigest()[:10]
    issue = Issue(
        issue_id=f"{rule_id}#{iid}",
        category=category,
        severity=severity,
        rule_id=rule_id,
        message=message,
        span=span,
        **kwargs,
    )
    if not issue.original and span.length:
        issue.original = kwargs.get("_text", "")[span.start:span.end] if "_text" in kwargs else ""
    issue.meta.pop("_text", None)
    return issue


@dataclass
class DimensionScore:
    """单维度得分。"""

    dimension: Dimension
    score: float                     # 0-100
    fatal: int = 0
    major: int = 0
    minor: int = 0
    info: int = 0
    total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dimension"] = self.dimension.value
        return d


@dataclass
class AuditReport:
    """终态报告。"""

    doc_id: str
    doc_title: str
    doc_type: str
    char_count: int
    paragraph_count: int
    issues: List[Issue] = field(default_factory=list)
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    overall_score: float = 100.0
    release_gate: str = "PASS"          # PASS | REVIEW | BLOCK
    summary: str = ""
    style_preset: str = ""
    style_metrics: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False               # 是否因本地模型不可用而降级
    llm_backend: str = ""
    terminology_version: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)   # 扩展元信息（事实校验、调度信息等）

    # ---- 统计 ----------------------------------------------------
    @property
    def counts(self) -> Dict[str, int]:
        out = {"FATAL": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
        for i in self.issues:
            out[i.severity.value] += 1
        return out

    def by_category(self) -> Dict[str, List[Issue]]:
        grouped: Dict[str, List[Issue]] = {}
        for i in self.issues:
            grouped.setdefault(i.category.value, []).append(i)
        for lst in grouped.values():
            lst.sort(key=lambda x: (x.severity.weight * -1, x.span.start))
        return grouped

    def sorted_issues(self) -> List[Issue]:
        return sorted(self.issues, key=lambda x: (-x.severity.weight, x.span.start))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "char_count": self.char_count,
            "paragraph_count": self.paragraph_count,
            "overall_score": self.overall_score,
            "release_gate": self.release_gate,
            "summary": self.summary,
            "style_preset": self.style_preset,
            "style_metrics": self.style_metrics,
            "degraded": self.degraded,
            "llm_backend": self.llm_backend,
            "terminology_version": self.terminology_version,
            "counts": self.counts,
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "issues": [i.to_dict() for i in self.sorted_issues()],
            "trace": self.trace,
            **self.meta,
        }
