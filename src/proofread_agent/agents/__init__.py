"""多 Agent 协作体系。"""

from .base import BaseAgent, EngineAgent
from .critic import CriticAgent, FactCheckAgent
from .graph import SPECIALIST_NODES, AuditPipeline, JoinAgent
from .report_agent import ReportAgent
from .revision import RevisionAgent
from .specialists import (
    AcademicAgent,
    LanguageAgent,
    LogicAgent,
    StyleAgent,
    TerminologyAgent,
)
from .state import AuditState, initial_state, trace_entry
from .supervisor import IntakeAgent, SupervisorAgent

__all__ = [
    "BaseAgent",
    "EngineAgent",
    "CriticAgent",
    "FactCheckAgent",
    "AuditPipeline",
    "JoinAgent",
    "SPECIALIST_NODES",
    "ReportAgent",
    "RevisionAgent",
    "AcademicAgent",
    "LanguageAgent",
    "LogicAgent",
    "StyleAgent",
    "TerminologyAgent",
    "AuditState",
    "initial_state",
    "trace_entry",
    "IntakeAgent",
    "SupervisorAgent",
]
