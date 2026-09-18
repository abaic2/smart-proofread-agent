"""五大审校引擎。"""

from .academic_engine import AcademicEngine
from .base import BaseEngine, EngineResult
from .logic_engine import LogicEngine
from .norm_engine import NormEngine
from .style_engine import StyleEngine
from .term_engine import TermEngine

__all__ = [
    "AcademicEngine",
    "BaseEngine",
    "EngineResult",
    "LogicEngine",
    "NormEngine",
    "StyleEngine",
    "TermEngine",
]
