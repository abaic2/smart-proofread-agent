"""智能审校 Agent —— 基于本地部署大语言模型的多 Agent 协作校审系统。

对外主入口::

    from proofread_agent import ProofreadAgent

    agent = ProofreadAgent()
    report = agent.audit_file("data/samples/demo_report.md")
    print(report.overall_score, report.release_gate)
"""

from .agents import AuditPipeline
from .config import Settings, load_settings
from .document import Document
from .llm import EmbeddingClient, LocalLLM, MockLLM, build_llm
from .pipeline import ProofreadAgent
from .report import (
    AuditReport,
    Category,
    Dimension,
    DimensionScore,
    Issue,
    Severity,
    Span,
    render_html,
    render_json,
    render_markdown,
    write_reports,
)
from .terminology import TerminologyStore, TerminologyUpdater

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ProofreadAgent",
    "AuditPipeline",
    "Settings",
    "load_settings",
    "Document",
    "EmbeddingClient",
    "LocalLLM",
    "MockLLM",
    "build_llm",
    "AuditReport",
    "Category",
    "Dimension",
    "DimensionScore",
    "Issue",
    "Severity",
    "Span",
    "render_html",
    "render_json",
    "render_markdown",
    "write_reports",
    "TerminologyStore",
    "TerminologyUpdater",
]
