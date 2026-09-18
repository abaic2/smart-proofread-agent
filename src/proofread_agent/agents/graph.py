"""LangGraph 多 Agent 工作流编排。

图结构（默认并行模式）::

                    ┌────────────────────────────────────────────┐
                    │                                            │
    START ──► intake ──► supervisor ──┬──► terminology_agent ──┐  │
                                      ├──► language_agent    ──┤  │
                                      ├──► logic_agent       ──┼──┴──► join
                                      ├──► academic_agent    ──┤        │
                                      └──► style_agent       ──┘        │
                                                                       ▼
                              ┌──────────────◄────────── critic_agent ◄─┘
                              │                              │
                    （批评修正环，最多 max_critic_rounds 轮）    │
                              └──────────────────────────────┘
                                                             ▼
                                   fact_check_agent ──► revision_agent ──► report_agent ──► END

编排要点
--------
1. **并行 fan-out / fan-in**：五个专业 Agent 在同一超步并发执行，
   共享 State 通过 reducer（``merge_dict`` / ``extend_list``）安全归并。
   为保证同步屏障（barrier）语义，未激活的 Agent 也会被调度，但立即返回，
   不产生实际计算。
2. **动态路由**：Supervisor 依据病灶画像决定激活哪些 Agent，
   通过条件边返回节点名列表实现运行时 fan-out。
3. **批评修正环**：Critic 对并行召回结果做误报压制与一致性消解；
   若仍存在高严重度问题且未达最大轮次，则回环执行第二轮保守收敛。
4. **确定性复算**：FactCheck 用确定性算法复算数据与文献问题，结果可作签发依据。
5. **降级可用**：未安装 langgraph 时自动切换到内置的顺序执行器
   （同样的节点函数、同样的状态语义），保证在受限内网环境中仍可运行。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..document import Document
from ..llm import EmbeddingClient
from ..report import AuditReport
from ..terminology import TerminologyStore
from .base import BaseAgent
from .critic import CriticAgent, FactCheckAgent
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

SPECIALIST_NODES = ["terminology_agent", "language_agent", "logic_agent",
                    "academic_agent", "style_agent"]


class JoinAgent(BaseAgent):
    """同步屏障节点：汇聚并行分支结果并统计召回总量。"""

    name = "join_agent"
    label = "结果汇聚节点"
    role = "等待全部并行 Agent 完成并归并召回结果"

    def execute(self, state: AuditState) -> Dict[str, Any]:
        issues = list(state.get("issues") or [])
        results = state.get("engine_results") or {}
        active = [k for k, v in results.items() if hasattr(v, "issues")]
        return {"llm_review": {"join": {
            "recalled": len(issues),
            "engines": {k: len(getattr(v, "issues", [])) for k, v in results.items()
                        if hasattr(v, "issues")},
            "active_engines": active,
        }}}

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        j = patch.get("llm_review", {}).get("join", {})
        detail = "，".join(f"{k}={v}" for k, v in (j.get("engines") or {}).items())
        return f"汇聚召回 {j.get('recalled', 0)} 条问题（{detail}）"


class AuditPipeline:
    """审校流水线：装配 Agent、编译图、执行。"""

    def __init__(self, settings: Any, store: TerminologyStore, llm: Any,
                 embedding: Optional[EmbeddingClient] = None) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.embedding = embedding or EmbeddingClient(
            (settings.llm.get("embedding") or {}), settings.llm)
        self.graph_cfg = settings.graph or {}
        self.agents: Dict[str, BaseAgent] = {}
        self._app = None
        self._compile()

    # ---- 装配 -------------------------------------------------------
    def _build_agents(self) -> Dict[str, BaseAgent]:
        s, store, llm = self.settings, self.store, self.llm
        agents: Dict[str, BaseAgent] = {
            "intake_agent": IntakeAgent(s, store, llm),
            "supervisor_agent": SupervisorAgent(s, store, llm),
            "terminology_agent": TerminologyAgent(s, store, llm, self.embedding),
            "language_agent": LanguageAgent(s, store, llm),
            "logic_agent": LogicAgent(s, store, llm),
            "academic_agent": AcademicAgent(s, store, llm, self.embedding),
            "style_agent": StyleAgent(s, store, llm),
            "join_agent": JoinAgent(s, store, llm),
            "critic_agent": CriticAgent(s, store, llm),
            "fact_check_agent": FactCheckAgent(s, store, llm),
            "revision_agent": RevisionAgent(s, store, llm),
            "report_agent": ReportAgent(s, store, llm),
        }
        return agents

    def _compile(self) -> None:
        self.agents = self._build_agents()
        try:
            self._app = self._build_langgraph()
            self.engine = "langgraph"
        except ImportError:
            self._app = None
            self.engine = "builtin-sequential"

    def _build_langgraph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AuditState)
        for name, agent in self.agents.items():
            builder.add_node(name, agent)

        builder.add_edge(START, "intake_agent")
        builder.add_edge("intake_agent", "supervisor_agent")

        supervisor: SupervisorAgent = self.agents["supervisor_agent"]  # type: ignore[assignment]

        if self.graph_cfg.get("parallel_agents", True):
            # 条件边返回节点名列表 → 触发并行 fan-out
            builder.add_conditional_edges(
                "supervisor_agent", supervisor.route_fn,
                {n: n for n in SPECIALIST_NODES},
            )
            for node in SPECIALIST_NODES:
                builder.add_edge(node, "join_agent")
        else:
            # 顺序模式：串行执行，便于在低配环境或调试时观察中间态
            chain = SPECIALIST_NODES
            builder.add_edge("supervisor_agent", chain[0])
            for a, b in zip(chain, chain[1:]):
                builder.add_edge(a, b)
            builder.add_edge(chain[-1], "join_agent")

        builder.add_edge("join_agent", "critic_agent")
        builder.add_conditional_edges(
            "critic_agent", self._critic_router,
            {"critic_agent": "critic_agent", "fact_check_agent": "fact_check_agent"},
        )
        builder.add_edge("fact_check_agent", "revision_agent")
        builder.add_edge("revision_agent", "report_agent")
        builder.add_edge("report_agent", END)

        return builder.compile()

    def _critic_router(self, state: AuditState) -> str:
        """批评修正环的循环条件。"""
        if not self.graph_cfg.get("enable_critic", True):
            return "fact_check_agent"
        max_rounds = int(self.graph_cfg.get("max_critic_rounds", 2))
        round_no = int(state.get("critic_round") or 0)
        issues = state.get("reviewed_issues") or state.get("issues") or []
        serious = any(i.severity.value in ("FATAL", "MAJOR") for i in issues)
        if round_no < max_rounds and serious:
            return "critic_agent"
        return "fact_check_agent"

    # ---- 执行 -------------------------------------------------------
    def run(self, doc: Document, doc_type: str = "调研专报",
            style_preset: str = "公文专报", target_journal: str = "") -> AuditReport:
        state = initial_state(doc, self.settings, self.store, self.llm,
                              doc_type=doc_type, style_preset=style_preset,
                              target_journal=target_journal)
        final = self._invoke(state)
        report = final.get("report")
        if report is None:
            raise RuntimeError(f"流水线未产出报告：{final.get('error', '未知原因')}")
        return report

    def _invoke(self, state: AuditState) -> AuditState:
        if self._app is not None:
            return self._app.invoke(state)
        return self._run_sequential(state)

    def _run_sequential(self, state: AuditState) -> AuditState:
        """内置顺序执行器（langgraph 不可用时的等价实现）。

        与图执行保持相同语义：节点返回 partial state，reducer 负责归并。
        """
        from .state import extend_list, last_value, merge_dict

        def merge(acc: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(acc)
            for k, v in (patch or {}).items():
                if k in ("issues", "llm_issues", "critic_notes", "suppressed_ids",
                         "fact_check", "trace"):
                    out[k] = extend_list(out.get(k), v)
                elif k in ("engine_results", "llm_review"):
                    out[k] = merge_dict(out.get(k), v)
                elif k == "reviewed_issues":
                    out[k] = last_value(out.get(k), v)
                else:
                    out[k] = v
            return out

        order = ["intake_agent", "supervisor_agent"] + SPECIALIST_NODES + [
            "join_agent", "critic_agent", "fact_check_agent",
            "revision_agent", "report_agent",
        ]
        cur = dict(state)
        for name in order:
            patch = self.agents[name](cur)
            cur = merge(cur, patch)
            if name == "critic_agent":
                nxt = self._critic_router(cur)
                while nxt == "critic_agent":
                    patch = self.agents["critic_agent"](cur)
                    cur = merge(cur, patch)
                    nxt = self._critic_router(cur)
        return cur

    # ---- 观测 -------------------------------------------------------
    def mermaid(self) -> str:
        """导出 Mermaid 流程图，便于写入文档。"""
        return _MERMAID

    def describe(self) -> List[Dict[str, str]]:
        return [
            {"node": name, "label": a.label, "role": a.role,
             "type": a.__class__.__name__}
            for name, a in self.agents.items()
        ]


_MERMAID = """flowchart TD
    A[受理预检 Agent<br/>文档解析·病灶画像] --> B[调度 Supervisor<br/>动态路由]
    B -->|fan-out| C1[政治术语 Agent<br/>三级匹配]
    B -->|fan-out| C2[语文规范 Agent]
    B -->|fan-out| C3[逻辑严密性 Agent]
    B -->|fan-out| C4[专业深度 Agent<br/>数据·格式·查重·文献]
    B -->|fan-out| C5[风格迁移 Agent]
    C1 --> J[结果汇聚 join]
    C2 --> J
    C3 --> J
    C4 --> J
    C5 --> J
    J --> K[批评修正 Agent<br/>误报压制·一致性消解]
    K -->|达到最大轮次| F[事实与数据校验 Agent<br/>确定性复算]
    K -->|仍有高危问题| K
    F --> R[修订稿生成 Agent<br/>自动修改+留痕]
    R --> P[报告聚合 Agent<br/>评分·门禁·报告]
    P --> E([END])
"""
