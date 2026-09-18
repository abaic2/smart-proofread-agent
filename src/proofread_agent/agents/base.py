"""Agent 基类：统一的节点签名、耗时统计与 trace 记录。

LangGraph 节点约定 ``(state) -> partial_state``。所有 Agent 继承
``BaseAgent``，把"调用引擎 → 汇总 → 写 trace"这套重复逻辑收敛到一处，
子类只需实现 ``execute``。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..document import Document
from ..engines.base import annotate_positions
from ..report import Issue
from .state import AuditState, trace_entry


class BaseAgent:
    """所有审校 Agent 的基类。"""

    name: str = "agent"
    label: str = "智能体"
    role: str = ""
    requires: List[str] = []          # 依赖的其他 Agent（用于拓扑排序与顺序执行兜底）

    def __init__(self, settings: Any, store: Any = None, llm: Any = None) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm

    # ---- 子类实现 --------------------------------------------------
    def execute(self, state: AuditState) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    # ---- LangGraph 节点签名 ----------------------------------------
    def __call__(self, state: AuditState) -> Dict[str, Any]:
        started = time.time()
        try:
            patch = self.execute(state)
            patch = patch or {}
        except Exception as exc:  # 单个 Agent 失败不应中断整图
            elapsed = (time.time() - started) * 1000
            return {
                "trace": [trace_entry(self.name, self.label, f"执行异常：{exc}",
                                      level="ERROR", elapsed_ms=round(elapsed, 1))],
                "llm_review": {f"{self.name}_error": str(exc)},
            }
        elapsed = (time.time() - started) * 1000
        summary = self.summarize(patch, state)
        patch.setdefault("trace", [])
        patch["trace"] = list(patch["trace"]) + [
            trace_entry(self.name, self.label, summary, elapsed_ms=round(elapsed, 1),
                        produced=len(patch.get("issues", []) or []))
        ]
        return patch

    # ---- 供子类覆写的摘要 -------------------------------------------
    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        n = len(patch.get("issues", []) or [])
        return f"产出问题 {n} 条"

    # ---- 公共工具 --------------------------------------------------
    @staticmethod
    def doc(state: AuditState) -> Document:
        return state["doc"]

    def llm_available(self, state: AuditState) -> bool:
        llm = state.get("llm")
        return bool(llm) and getattr(llm, "backend_name", "mock") != "mock"

    def ask_llm_json(self, state: AuditState, system: str, user: str,
                     fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """调用本地模型并要求 JSON 输出；不可用时返回 fallback。"""
        if not self.llm_available(state):
            return fallback or {}
        try:
            return state["llm"].json_chat([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]) or (fallback or {})
        except Exception as exc:
            return {"_error": str(exc), **(fallback or {})}

    def ask_llm_text(self, state: AuditState, system: str, user: str,
                     fallback: str = "") -> str:
        if not self.llm_available(state):
            return fallback
        try:
            return state["llm"].chat([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]).text or fallback
        except Exception:
            return fallback


class EngineAgent(BaseAgent):
    """把某个审校引擎包装成 Agent 的通用适配器。"""

    engine_key: str = ""

    def __init__(self, settings: Any, store: Any, llm: Any, engine: Any) -> None:
        super().__init__(settings, store, llm)
        self.engine = engine

    def execute(self, state: AuditState) -> Dict[str, Any]:
        # 路由剪枝：Supervisor 未激活本 Agent 时快速跳过（保留在图中以保证同步屏障）
        route = state.get("route")
        if route and self.engine_key not in route:
            return {"trace": [trace_entry(self.name, self.label,
                                          "未在本轮路由范围内，跳过执行", skipped=True)]}
        doc = self.doc(state)
        result = self.engine.run(doc)
        annotate_positions(doc, result.issues)
        patch: Dict[str, Any] = {
            "engine_results": {self.engine_key: result},
            "issues": list(result.issues),
        }
        # 风格 Agent 额外回传画像与迁移清单
        if self.engine_key == "style":
            patch["llm_review"] = {"style_profile": result.stats.get("profile", {}),
                                   "style_transfer_plan": self.engine.transfer_plan()}
        return patch

    def summarize(self, patch: Dict[str, Any], state: AuditState) -> str:
        res = (patch.get("engine_results") or {}).get(self.engine_key)
        if res is None:
            return "无产出"
        sevs: Dict[str, int] = {}
        for i in res.issues:
            sevs[i.severity.value] = sevs.get(i.severity.value, 0) + 1
        detail = " / ".join(f"{k}:{v}" for k, v in sevs.items()) or "无问题"
        stat_text = " ".join(
            f"{k}={v}" for k, v in list(res.stats.items())[:5]
            if not isinstance(v, (dict, list))
        )
        return f"{self.engine.description}｜命中 {len(res.issues)} 条（{detail}）｜{stat_text}"
