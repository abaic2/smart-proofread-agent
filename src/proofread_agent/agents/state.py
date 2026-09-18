"""LangGraph 共享状态（State）定义。

状态设计要点
------------
* **可并行**：``issues`` 与 ``engine_results`` 使用 reducer 归并，
  允许五个审校 Agent 在同一超步（superstep）并发写入而不互相覆盖。
* **可观测**：``trace`` 记录每个节点的输入摘要、耗时、产出数量，
  形成完整审计链，满足"审校结论可回溯"的合规要求。
* **可断点续跑**：状态全部为可序列化对象（dataclass/dict/list/str），
  可直接挂 checkpointer 做持久化与人工介入（human-in-the-loop）。
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Dict, List, TypedDict

from ..document import Document
from ..report import AuditReport, Issue

# ---------------------------------------------------------------- reducers
def merge_dict(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> Dict[str, Any]:
    """字典浅归并：并行 Agent 各写各的 key，互不干扰。"""
    out: Dict[str, Any] = {}
    if left:
        out.update(left)
    if right:
        out.update(right)
    return out


def extend_list(left: List[Any] | None, right: List[Any] | None) -> List[Any]:
    """列表追加：问题清单在并行分支中累加。"""
    return (left or []) + (right or [])


def last_value(left: Any, right: Any) -> Any:
    """默认语义：后写覆盖。"""
    return right if right is not None else left


class AuditState(TypedDict, total=False):
    """多 Agent 共享的工作流状态。"""

    # ---- 输入 ----
    doc: Document
    doc_type: str
    style_preset: str
    target_journal: str

    # ---- 运行时依赖（不入图计算，仅传递引用）----
    settings: Any
    store: Any
    llm: Any

    # ---- 调度 ----
    route: List[str]                       # 本稿件需要激活的 Agent
    route_reason: str
    defects_profile: Dict[str, Any]        # 预检得到的稿件"病灶画像"

    # ---- 各 Agent 产出 ----
    engine_results: Annotated[Dict[str, Any], merge_dict]
    issues: Annotated[List[Issue], extend_list]              # 并行召回（累加）
    reviewed_issues: Annotated[List[Issue], last_value]      # 批评修正后的定稿（覆盖）

    # ---- LLM 阶段产出 ----
    llm_review: Annotated[Dict[str, Any], merge_dict]
    llm_issues: Annotated[List[Issue], extend_list]
    critic_notes: Annotated[List[Dict[str, Any]], extend_list]
    critic_round: int
    suppressed_ids: Annotated[List[str], extend_list]
    fact_check: Annotated[List[Dict[str, Any]], extend_list]

    # ---- 终态 ----
    report: AuditReport
    trace: Annotated[List[Dict[str, Any]], extend_list]
    error: str


def trace_entry(node: str, agent: str, summary: str, **extra: Any) -> Dict[str, Any]:
    """构造一条 trace 记录。"""
    rec = {
        "node": node,
        "agent": agent,
        "summary": summary,
        "ts": time.strftime("%H:%M:%S"),
    }
    rec.update(extra)
    return rec


def initial_state(doc: Document, settings: Any, store: Any, llm: Any,
                  doc_type: str = "调研专报", style_preset: str = "公文专报",
                  target_journal: str = "") -> AuditState:
    """构造初始状态。"""
    return AuditState(
        doc=doc,
        doc_type=doc_type,
        style_preset=style_preset,
        target_journal=target_journal,
        settings=settings,
        store=store,
        llm=llm,
        route=[],
        defects_profile={},
        engine_results={},
        issues=[],
        reviewed_issues=[],
        llm_review={},
        llm_issues=[],
        critic_notes=[],
        critic_round=0,
        suppressed_ids=[],
        fact_check=[],
        trace=[],
    )
