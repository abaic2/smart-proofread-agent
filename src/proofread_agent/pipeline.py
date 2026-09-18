"""对外统一门面：``ProofreadAgent``。

把"配置 → 术语库 → 本地模型 → 引擎 → LangGraph → 报告"整条链路收敛成
三行代码即可调用的 API，供 CLI、Web 服务、批处理脚本共用：

.. code-block:: python

    from proofread_agent import ProofreadAgent

    agent = ProofreadAgent()                      # 读 config/settings.yaml
    report = agent.audit_file("data/samples/demo_report.md", doc_type="调研专报")
    agent.export(report, "outputs")

也支持在无 GPU、无模型服务的内网环境直接运行——此时自动降级为
纯规则引擎，报告上标注 ``degraded=True``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import AuditPipeline
from .config import Settings, load_settings
from .document import Document
from .llm import EmbeddingClient, build_llm
from .report import AuditReport, render_html, render_markdown, write_reports
from .terminology import TerminologyStore, TerminologyUpdater


class ProofreadAgent:
    """智能审校 Agent 门面。"""

    def __init__(self, config_path: str | os.PathLike | None = None,
                 settings: Optional[Settings] = None,
                 preset: Optional[str] = None) -> None:
        self.settings = settings or load_settings(config_path)
        self.store = TerminologyStore(self.settings)
        self.store.load()
        self.llm = build_llm(self.settings)
        self.embedding = EmbeddingClient(
            (self.settings.llm.get("embedding") or {}), self.settings.llm)
        self.embedding.probe()
        self.preset = preset or self.settings.style.get("default_preset", "公文专报")
        self.pipeline = AuditPipeline(self.settings, self.store, self.llm, self.embedding)

    # ---- 信息 -------------------------------------------------------
    @property
    def info(self) -> Dict[str, Any]:
        return {
            "engine": self.pipeline.engine,
            "llm_backend": getattr(self.llm, "backend_name", "none"),
            "llm_model": getattr(self.llm, "model", "none"),
            "degraded": getattr(self.llm, "backend_name", "") == "mock",
            "embedding_mode": getattr(self.embedding, "mode", "unknown"),
            "terminology_version": self.store.version,
            "terminology_entries": len(self.store.entries),
            "banned_rules": len(self.store.banned),
            "style_preset": self.preset,
            "graph_nodes": [a["node"] for a in self.pipeline.describe()],
        }

    # ---- 审校 -------------------------------------------------------
    def audit_file(self, path: str | os.PathLike, doc_type: str = "调研专报",
                   style_preset: Optional[str] = None,
                   target_journal: str = "") -> AuditReport:
        doc = Document.from_file(path, doc_type=doc_type)
        return self.audit_document(doc, doc_type, style_preset, target_journal)

    def audit_text(self, text: str, doc_type: str = "调研专报",
                   style_preset: Optional[str] = None, title: str = "") -> AuditReport:
        doc = Document.from_text(text, doc_type=doc_type, title=title)
        return self.audit_document(doc, doc_type, style_preset)

    def audit_document(self, doc: Document, doc_type: str = "调研专报",
                       style_preset: Optional[str] = None,
                       target_journal: str = "") -> AuditReport:
        self._last_text = doc.text
        return self.pipeline.run(
            doc, doc_type=doc_type,
            style_preset=style_preset or self.preset,
            target_journal=target_journal,
        )

    # ---- 术语库维护 -------------------------------------------------
    def update_terminology(self, auto_promote: bool = False) -> Dict[str, Any]:
        updater = TerminologyUpdater(self.store, self.settings)
        return updater.run_update(auto_promote=auto_promote)

    def approve_pending(self, pending_file: str | os.PathLike,
                        ids: Optional[List[str]] = None) -> Dict[str, Any]:
        updater = TerminologyUpdater(self.store, self.settings)
        return updater.approve_pending(pending_file, ids)

    # ---- 报告导出 ---------------------------------------------------
    def export(self, report: AuditReport, out_dir: str | os.PathLike = "outputs",
               text: str = "", stem: str = "audit_report",
               formats: Optional[List[str]] = None) -> Dict[str, Path]:
        formats = formats or self.settings.report.get("formats", ["markdown", "html", "json"])
        if not text:
            text = getattr(self, "_last_text", "") or ""
        return write_reports(report, text, out_dir, formats, stem)

    def render(self, report: AuditReport, text: str = "",
               fmt: str = "markdown") -> str:
        if fmt == "html":
            return render_html(report, text)
        return render_markdown(report, text)

    def graph_mermaid(self) -> str:
        return self.pipeline.mermaid()

    def describe_agents(self) -> List[Dict[str, str]]:
        return self.pipeline.describe()
