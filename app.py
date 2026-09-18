"""智能审校 Agent · Streamlit 交互式工作台

本地运行::

    streamlit run app.py

设计说明
--------
* 视觉：深蓝（#0A1F44）+ 金色（#C9A227）的庄重配色，适配公文/学术场景；
  主内容区保持浅色以保证长文报告的可读性。
* 交互：侧边栏 option_menu 导航；审校过程用 ``st.status`` 逐步回放 Agent 轨迹；
  结果区按"结论 → 划改稿 → 问题清单 → 复算 → 批评修正 → 修订留痕 → 轨迹"分层展开。
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from proofread_agent import ProofreadAgent, Document  # noqa: E402
from proofread_agent.report import (  # noqa: E402
    highlight_manuscript,
    render_html,
    render_json,
    render_markdown,
)
from proofread_agent.report.exporters import export_docx, export_pdf  # noqa: E402
from proofread_agent.terminology import TerminologyUpdater  # noqa: E402

# ======================================================================
# 页面配置与设计系统
# ======================================================================
def _set_page_config() -> None:
    """page_icon 的 material 图标语法在旧版 Streamlit 上不可用，做一次降级。"""
    try:
        st.set_page_config(page_title="智能审校 Agent", page_icon=":material/verified:",
                           layout="wide", initial_sidebar_state="expanded")
    except Exception:
        st.set_page_config(page_title="智能审校 Agent", layout="wide",
                           initial_sidebar_state="expanded")


_set_page_config()

NAVY_DEEP, NAVY, NAVY_SOFT = "#081733", "#0A1F44", "#16305E"
GOLD, GOLD_LIGHT, GOLD_DIM = "#C9A227", "#E8C766", "rgba(201,162,39,0.35)"

SEV_META = {
    "FATAL": {"label": "致命", "color": "#E54545", "bg": "#FCEBEB", "border": "#E54545"},
    "MAJOR": {"label": "严重", "color": "#D2691E", "bg": "#FDF3E7", "border": "#E08A3C"},
    "MINOR": {"label": "一般", "color": "#2E8B57", "bg": "#EAF6EE", "border": "#4CAF7D"},
    "INFO": {"label": "提示", "color": "#3A6FB0", "bg": "#EAF1FA", "border": "#6A9BD8"},
}
CATEGORY_ORDER = ["政治术语", "语文规范", "逻辑严密性", "数据合规", "参考文献",
                  "格式规范", "重复发表", "风格适配"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* ---------- 侧边栏：深蓝底，注意新版 DOM 的 section/div 双写 ---------- */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #081733 0%, #0A1F44 55%, #16305E 100%) !important;
  border-right: 1px solid rgba(201,162,39,0.28);
}
section[data-testid="stSidebar"] > div, div[data-testid="stSidebar"] > div,
div[data-testid="stSidebarContent"],
div[data-testid="stSidebar"] [data-testid="stSidebarContent"],
div[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  background: transparent !important;
}
section[data-testid="stSidebar"] *, div[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small,
div[data-testid="stSidebar"] .stCaption,
div[data-testid="stSidebar"] small { color: #AFC2E8 !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #FFFFFF !important; font-weight: 600;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #22304F !important; }
section[data-testid="stSidebar"] .stButton > button,
div[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,255,255,.08); border: 1px solid rgba(201,162,39,.45);
  color: #FFFFFF !important; border-radius: 11px;
  white-space: normal; line-height: 1.5; text-align: center; padding: 10px 14px;
}
section[data-testid="stSidebar"] .stButton > button:hover,
div[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(201,162,39,.22); border-color: #E8C766;
}

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 20px; padding: 40px 44px 36px;
  background: linear-gradient(135deg, #081733 0%, #0A1F44 52%, #1B3C74 100%);
  border: 1px solid rgba(201,162,39,0.34); margin-bottom: 22px;
}
.hero::before, .hero::after {
  content: ""; position: absolute; border-radius: 50%; pointer-events: none;
}
.hero::before {
  width: 320px; height: 320px; right: -90px; top: -130px;
  background: radial-gradient(circle, rgba(201,162,39,0.26) 0%, rgba(201,162,39,0) 70%);
}
.hero::after {
  width: 240px; height: 240px; left: -70px; bottom: -120px;
  background: radial-gradient(circle, rgba(120,170,255,0.22) 0%, rgba(120,170,255,0) 70%);
}
.hero .eyebrow {
  font-size: 12.5px; letter-spacing: 2.4px; color: #E8C766; font-weight: 600;
  text-transform: uppercase; margin-bottom: 12px;
}
.hero h1 {
  font-family: "Noto Serif SC", "Songti SC", "SimSun", serif;
  font-size: 40px; line-height: 1.3; margin: 0 0 12px; font-weight: 700;
  background: linear-gradient(92deg, #FFFFFF 8%, #FFE9A8 52%, #E8C766 96%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #C7D4EE; font-size: 15px; line-height: 1.85; margin: 0; max-width: 900px; }
.hero .tags { margin-top: 20px; display: flex; flex-wrap: wrap; gap: 9px; }
.hero .tags span {
  font-size: 12.5px; color: #FFE9A8; padding: 5px 13px; border-radius: 999px;
  background: rgba(201,162,39,0.13); border: 1px solid rgba(201,162,39,0.42);
}

/* ---------- 卡片 ---------- */
.kpi {
  background: #FFFFFF; border: 1px solid #E3E8F0; border-left: 4px solid #C9A227;
  border-radius: 13px; padding: 16px 18px; height: 100%;
}
.kpi .v { font-size: 27px; font-weight: 700; line-height: 1.1; letter-spacing: -0.5px; }
.kpi .l { font-size: 12.5px; color: #6B7688; margin-top: 6px; }
.card {
  background: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 14px;
  padding: 20px 24px; margin-bottom: 14px;
}
.card.glass {
  background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFE 100%);
  border: 1px solid #E3E8F0; transition: transform .18s ease, box-shadow .18s ease;
}
.card.glass:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(10,31,68,0.09); }
.card h4 { margin: 0 0 8px; font-size: 15px; font-weight: 700; color: #0A1F44; }
.card .desc { font-size: 13px; color: #5A6478; line-height: 1.75; }

.gate {
  display: inline-flex; align-items: center; gap: 10px; padding: 9px 20px;
  border-radius: 999px; font-size: 15px; font-weight: 700; color: #FFFFFF;
}
.gate .dot { width: 9px; height: 9px; border-radius: 50%; background: rgba(255,255,255,.9); }

.score-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.score-bar .name { width: 130px; font-size: 13px; color: #38445C; }
.score-bar .track { flex: 1; height: 9px; border-radius: 5px; background: #EEF1F7; overflow: hidden; }
.score-bar .fill { height: 100%; border-radius: 5px; }
.score-bar .num { width: 48px; text-align: right; font-size: 13.5px; font-weight: 700; color: #0A1F44; }

.issue {
  border: 1px solid #E3E8F0; border-left-width: 4px; border-radius: 11px;
  padding: 14px 17px; margin-bottom: 11px; background: #FFFFFF;
}
.issue .top { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.issue .msg { font-size: 14px; font-weight: 700; color: #1B2436; }
.chip {
  font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 6px; color: #FFFFFF;
  white-space: nowrap;
}
.meta { font-size: 12px; color: #6B7688; margin-top: 7px; line-height: 1.75; }
.mono, .meta code {
  font-family: "SFMono-Regular", Consolas, monospace; font-size: 11.5px;
  background: #F2F4F8; padding: 1px 6px; border-radius: 4px; color: #374151;
}
.diff2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.diff2 > div { background: #F7F9FC; border: 1px solid #E3E8F0; border-radius: 9px; padding: 9px 12px; }
.diff2 .h { font-size: 11px; color: #6B7688; margin-bottom: 4px; }
.old { text-decoration: line-through; text-decoration-color: #E54545; color: #8B1A1A; font-size: 13px; }
.new { color: #0A6B2F; font-weight: 700; font-size: 13px; }

pre.manuscript {
  white-space: pre-wrap; word-break: break-word; font-family: "Noto Serif SC", "Microsoft YaHei", serif;
  font-size: 15px; line-height: 2.15; background: #FFFFFF; border: 1px dashed #D8DEE9;
  border-radius: 12px; padding: 26px 30px; margin: 0; color: #1B2436;
}
mark { border-radius: 3px; padding: 1px 0; border-bottom: 1.5px solid; cursor: help; }
mark.FATAL { background: #FBD5D5; border-color: #E54545; }
mark.MAJOR { background: #FAE0C6; border-color: #E08A3C; }
mark.MINOR { background: #D8EEDF; border-color: #4CAF7D; }
mark.INFO { background: #DCE8F8; border-color: #6A9BD8; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px; color: #6B7688; margin-bottom: 14px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }

.trace-item {
  border-left: 2px solid #E3E8F0; padding: 0 0 16px 18px; position: relative; margin-left: 5px;
}
.trace-item::before {
  content: ""; position: absolute; left: -7px; top: 5px; width: 11px; height: 11px;
  border-radius: 50%; background: #C9A227; border: 2px solid #FFFFFF; box-shadow: 0 0 0 1px #E3E8F0;
}
.trace-item.skipped::before { background: #C8D0DD; }
.trace-item .t { font-size: 12px; color: #8A94A6; margin-top: 3px; }
.trace-item .s { font-size: 13px; color: #38445C; line-height: 1.7; margin-top: 4px; }

.summary-box {
  white-space: pre-wrap; font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12.5px; line-height: 1.85; background: #F8FAFD;
  border: 1px solid #E3E8F0; border-radius: 11px; padding: 18px 20px; color: #2A3548;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E3E8F0; }
.stTabs [data-baseweb="tab"] {
  height: 42px; font-size: 13.5px; font-weight: 600; color: #5A6478; padding: 0 16px;
}
.stTabs [aria-selected="true"] { color: #0A1F44 !important; border-bottom: 2px solid #C9A227 !important; }
div[data-testid="stExpander"] {
  border: 1px solid #E3E8F0; border-radius: 11px; margin-bottom: 8px; background: #FFFFFF;
}
div[data-testid="stExpander"] summary { font-size: 13.5px; font-weight: 600; color: #1B2436; }
</style>
"""


def inject_css(css: str) -> None:
    """`<style>` 注入：优先 st.html（不做 HTML 净化），旧版本回退 st.markdown。"""
    if hasattr(st, "html"):
        st.html(css)
    else:  # pragma: no cover
        st.markdown(css, unsafe_allow_html=True)


inject_css(CSS)


#: 让组件占满容器宽度。Streamlit 1.49 起 ``use_container_width`` 被 ``width="stretch"``
#: 取代，这里做一次版本自适应，避免云端与本地因版本差异出现弃用告警或参数错误。
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 49) else {"use_container_width": True}


# ======================================================================
# 数据与缓存
# ======================================================================
@st.cache_resource(show_spinner=False)
def get_agent() -> ProofreadAgent:
    return ProofreadAgent()


@st.cache_data(show_spinner=False)
def load_sample() -> str:
    return (ROOT / "data" / "samples" / "demo_report.md").read_text(encoding="utf-8")


# ======================================================================
# 稿件文本抽取（Markdown / 纯文本 / Word / PDF）
# ======================================================================
def extract_docx_text(raw: bytes) -> str:
    """从 .docx 字节流抽取正文与表格文本。"""
    from docx import Document as _Docx

    doc = _Docx(io.BytesIO(raw))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text and para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            line = " ".join(c for c in cells if c)
            if line:
                parts.append(line)
    return "\n".join(parts)


def extract_pdf_text(raw: bytes) -> str:
    """从 PDF 字节流抽取文本（依赖 pdfplumber）。"""
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
    return "\n".join(parts)


def extract_upload_text(raw: bytes, filename: str) -> str:
    """按扩展名路由到对应解析器；纯文本按常见编码回退解码。"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("docx", "doc"):
        return extract_docx_text(raw)
    if ext == "pdf":
        return extract_pdf_text(raw)
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def agent_info() -> dict:
    try:
        return get_agent().info
    except Exception as exc:  # pragma: no cover
        st.error(f"初始化失败：{exc}")
        return {}


# ======================================================================
# 通用渲染组件
# ======================================================================
def hero(eyebrow: str, title: str, desc: str, tags: list[str]) -> None:
    tag_html = "".join(f"<span>{t}</span>" for t in tags)
    st.markdown(
        f"""<div class="hero">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  <p>{desc}</p>
  <div class="tags">{tag_html}</div>
</div>""",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items: [(数值, 标签, 颜色)]"""
    cols = st.columns(len(items), gap="small")
    for col, (value, label, color) in zip(cols, items):
        with col:
            st.markdown(
                f"""<div class="kpi" style="border-left-color:{color}">
  <div class="v" style="color:{color}">{value}</div>
  <div class="l">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )


def score_bars(report) -> None:
    rows = []
    for ds in report.dimension_scores:
        color = "#2E8B57" if ds.score >= 85 else ("#D2691E" if ds.score >= 60 else "#E54545")
        rows.append(
            f"""<div class="score-bar">
  <div class="name">{ds.dimension.value}</div>
  <div class="track"><div class="fill" style="width:{ds.score}%;background:{color}"></div></div>
  <div class="num" style="color:{color}">{ds.score}</div>
</div>"""
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def issue_card(issue) -> str:
    m = SEV_META[issue.severity.value]
    parts = [
        f"<div class='issue' style='border-left-color:{m['border']};background:{m['bg']}'>",
        "<div class='top'>",
        f"<span class='chip' style='background:{m['color']}'>{m['label']}</span>",
        f"<span class='msg'>{issue.message}</span></div>",
        f"<div class='meta'>第 {issue.line} 行 · 第 {issue.paragraph + 1} 段 · "
        f"字符 {issue.span.start}&ndash;{issue.span.end} · 规则 "
        f"<code>{issue.rule_id}</code> · 置信度 {issue.confidence} · "
        f"检出 <code>{issue.agent}</code></div>",
    ]
    if issue.original:
        parts.append(
            "<div class='diff2'>"
            f"<div><div class='h'>原文</div><span class='old'>{_esc(issue.original)}</span></div>"
            f"<div><div class='h'>建议</div><span class='new'>{_esc(issue.suggestion)}</span></div>"
            "</div>"
        )
    elif issue.suggestion:
        parts.append(f"<div class='meta'>建议：<span class='new'>{_esc(issue.suggestion)}</span></div>")
    if issue.explain:
        parts.append(f"<div class='meta'>依据：{_esc(issue.explain)}</div>")
    if issue.source:
        parts.append(f"<div class='meta'>出处：{_esc(issue.source)}</div>")
    parts.append("</div>")
    return "".join(parts)


def _esc(text: str) -> str:
    return (str(text or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def plotly_base(height: int, title: str = "") -> dict:
    layout = dict(
        height=height,
        margin=dict(l=10, r=16, t=40 if title else 12, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="-apple-system, Microsoft YaHei, sans-serif", size=12, color="#38445C"),
        showlegend=False,
    )
    if title:
        layout["title"] = dict(text=title, x=0.01, font=dict(size=13.5, color="#0A1F44"))
    return layout


def chart_dimensions(report) -> go.Figure:
    dims = [d.dimension.value for d in report.dimension_scores][::-1]
    scores = [d.score for d in report.dimension_scores][::-1]
    colors = ["#2E8B57" if s >= 85 else ("#D2691E" if s >= 60 else "#E54545") for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=dims, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}" for s in scores], textposition="outside",
        textfont=dict(size=12, color="#38445C"), width=0.58,
    ))
    fig.update_layout(**plotly_base(260))
    fig.update_xaxes(range=[0, 108], showgrid=True, gridcolor="#EEF1F7",
                     zeroline=False, ticksuffix=" ", showticklabels=False,
                     title=None)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12.5, color="#38445C"))
    return fig


def chart_severity(report) -> go.Figure:
    counts = report.counts
    labels = ["致命", "严重", "一般", "提示"]
    values = [counts["FATAL"], counts["MAJOR"], counts["MINOR"], counts["INFO"]]
    colors = ["#E54545", "#E08A3C", "#4CAF7D", "#6A9BD8"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+value", textfont=dict(size=12.5),
        sort=False,
    ))
    layout = plotly_base(260)
    layout.update(showlegend=False, margin=dict(l=6, r=6, t=12, b=6))
    fig.update_layout(**layout)
    fig.add_annotation(
        text=f"<b>{len(report.issues)}</b><br><span style='font-size:11px;color:#6B7688'>问题总数</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="#0A1F44"),
    )
    return fig


def chart_categories(report) -> go.Figure:
    grouped = report.by_category()
    cats = [c for c in CATEGORY_ORDER if c in grouped]
    cats += [c for c in grouped if c not in cats]
    values = [len(grouped[c]) for c in cats]
    fig = go.Figure(go.Bar(
        x=cats, y=values, marker=dict(color="#0A1F44", line=dict(width=0)), width=0.5,
        text=values, textposition="outside", textfont=dict(size=12, color="#38445C"),
    ))
    fig.update_layout(**plotly_base(250))
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F7", zeroline=False, showticklabels=False)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12))
    return fig


# ======================================================================
# 页面：概览
# ======================================================================
def page_overview() -> None:
    info = agent_info()
    hero(
        "LOCAL LLM · MULTI-AGENT PROOFREADING",
        "智能审校 Agent",
        "基于本地部署大语言模型的多 Agent 协作校审系统，面向调研专报与学术期刊："
        "政治术语精准校验、语文规范审核、逻辑严密性评估、专业文本深度审校与风格迁移。"
        "稿件不出域，本地模型不可达时自动降级为规则引擎，流程不中断。",
        ["政治术语三级匹配", "LangGraph 12 节点编排", "5 路并行审校",
         "数据自洽复算", "跨稿件查重", "GB/T 7714 校验"],
    )

    c = st.columns(6, gap="small")
    items = [
        (str(info.get("graph_nodes") and len(info["graph_nodes"]) or 12), "Agent 节点", "#0A1F44"),
        (str(info.get("terminology_entries", 30)), "术语库条目", "#C9A227"),
        (str(info.get("banned_rules", 12)), "禁用词规则", "#D2691E"),
        ("3", "术语匹配层级", "#2E8B57"),
        ("5", "审校引擎", "#3A6FB0"),
        ("49", "自动化测试", "#8A6BB0"),
    ]
    for col, (v, l, color) in zip(c, items):
        with col:
            st.markdown(
                f"""<div class="kpi" style="border-left-color:{color}">
  <div class="v" style="color:{color}">{v}</div><div class="l">{l}</div></div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### 能力矩阵")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        cards = [
            ("政治术语精准审校引擎",
             "三级匹配：Aho-Corasick 精确匹配 → 字符二元组倒排索引 + 编辑距离近似匹配 → "
             "本地嵌入向量语义匹配。覆盖称谓固定表述、组合提法顺序、提法演进、涉台涉港涉澳红线、禁用慎用词。",
             "#C9A227"),
            ("专业文本深度审校",
             "数据合规：分项—合计—占比算术自洽复算；格式规范：章节序号、图表引用、序号体例；"
             "重复发表：SimHash + MinHash + TF-IDF 三层查重（文内自重复与跨稿件）；"
             "参考文献：GB/T 7714-2015 逐项校验与引文双向对应。",
             "#3A6FB0"),
            ("语文规范与逻辑严密性",
             "错别字与词语误用、标点（GB/T 15834）、数字单位、冗余欧化、长句可读性；"
             "推论越界、因果跳步、绝对化断言、论证强度冲突、指代不明、数量词与数据矛盾。",
             "#2E8B57"),
            ("风格迁移",
             "文体量化画像（平均句长、口语密度、绝对化密度、数据标注率、段旨句命中率）"
             "与目标区间比对，输出可执行的迁移动作清单与受控改写示范。",
             "#D2691E"),
        ]
        for title, desc, color in cards:
            st.markdown(
                f"""<div class="card glass" style="border-left:4px solid {color}">
  <h4>{title}</h4><div class="desc">{desc}</div></div>""",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            f"""<div class="card">
  <h4>运行环境</h4>
  <div class="meta" style="font-size:12.5px">
    编排引擎：<code>{info.get('engine', '-')}</code><br>
    推理后端：<code>{info.get('llm_backend', '-')}</code> / {info.get('llm_model', '-')}<br>
    嵌入模式：<code>{info.get('embedding_mode', '-')}</code><br>
    术语库版本：<code>{info.get('terminology_version', '-')}</code><br>
    默认文体预设：{info.get('style_preset', '-')}
  </div></div>""",
            unsafe_allow_html=True,
        )
        if info.get("degraded"):
            st.warning(
                "当前未连接本地大模型，审校由**纯规则引擎**完成（`degraded=True`）。"
                "规则层覆盖全部硬性校验；模型层负责的语义复核与改写示范会跳过。",
                icon=None,
            )
        else:
            st.success("本地大模型已连接，语义复核与改写示范已启用。", icon=None)

        st.markdown(
            """<div class="card">
  <h4>阅读顺序建议</h4>
  <div class="desc">
  ① 在「在线审校」页上传或粘贴稿件，选择目标文体后运行；<br>
  ② 先看结论与签发门禁，再看划改稿定位问题；<br>
  ③ 「问题清单」按类别逐条核对原文与建议；<br>
  ④ 数据类结论务必查看「数据复算」中的公式与信源；<br>
  ⑤ 「批评修正」记录系统主动驳回的误报，可据此判断是否申诉。
  </div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### 审校流水线")
    nodes = get_agent().describe_agents() if info else []
    if nodes:
        for row in range(0, len(nodes), 4):
            cols = st.columns(4, gap="small")
            for col, node in zip(cols, nodes[row:row + 4]):
                with col:
                    st.markdown(
                        f"""<div class="card" style="padding:14px 16px;min-height:118px">
  <div style="font-size:13px;font-weight:700;color:#0A1F44">{node['label']}</div>
  <div class="desc" style="font-size:12px;margin-top:6px">{node['role']}</div>
  <div class="meta mono" style="margin-top:8px">{node['node']}</div>
</div>""",
                        unsafe_allow_html=True,
                    )


# ======================================================================
# 页面：在线审校
# ======================================================================
def run_audit(text: str, source_name: str, doc_type: str, preset: str) -> None:
    agent = get_agent()
    doc = Document.from_text(text, doc_type=doc_type, source_path=source_name,
                             title=source_name)
    with st.status("多 Agent 协作审校进行中…", expanded=True) as status:
        st.write("受理预检：解析文档结构、统计特征、生成病灶画像")
        report = agent.audit_document(doc, doc_type=doc_type, style_preset=preset)
        trace = report.to_dict().get("trace", [])
        for t in trace:
            flag = "（本轮未激活）" if t.get("skipped") else ""
            el = t.get("elapsed_ms")
            st.write(f"{t.get('agent')} · {t.get('summary','')}{flag}"
                     f"{f'（{el} ms）' if el is not None else ''}")
            time.sleep(0.1)
        status.update(
            label=f"审校完成：综合 {report.overall_score} 分 · 门禁 {report.release_gate}",
            state="complete", expanded=False,
        )
    st.session_state.update(report=report, report_text=text,
                            source_name=source_name, doc_type=doc_type, preset=preset)
    _push_history(report, source_name, doc_type)


def _push_history(report, source_name: str, doc_type: str) -> None:
    """把本次审校的轻量记录追加到会话级历史，供「审校历史」页对比。

    同一 doc_id 只保留最新一次，避免重复堆积；记录仅存于当前会话，
    刷新页面即清空（遵循 Streamlit 会话状态语义）。
    """
    hist = st.session_state.get("audit_history", [])
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "doc_id": report.doc_id,
        "source": Path(source_name).name,
        "doc_type": doc_type,
        "score": report.overall_score,
        "gate": report.release_gate,
        "FATAL": report.counts["FATAL"],
        "MAJOR": report.counts["MAJOR"],
        "MINOR": report.counts["MINOR"],
        "INFO": report.counts["INFO"],
        "issue_count": len(report.issues),
        "dimensions": {d.dimension.value: d.score for d in report.dimension_scores},
    }
    hist = [h for h in hist if h.get("doc_id") != report.doc_id]
    hist.append(rec)
    st.session_state["audit_history"] = hist


def render_report(report, text: str, source_name: str) -> None:
    counts = report.counts
    gate_color = {"BLOCK": "#E54545", "REVIEW": "#D2691E", "PASS": "#2E8B57"}[report.release_gate]
    gate_desc = {"BLOCK": "存在致命级问题，必须修改后重新审校",
                 "REVIEW": "存在需要修改的问题，建议修改后再审",
                 "PASS": "未发现阻断性问题"}[report.release_gate]

    st.markdown(
        f"""<div class="card" style="padding:20px 24px">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <span class="gate" style="background:{gate_color}">
      <span class="dot"></span>签发门禁 {report.release_gate}</span>
    <span style="font-size:13.5px;color:#5A6478">{gate_desc}</span>
    <span style="margin-left:auto;font-size:13px;color:#6B7688">
      综合评分 <b style="font-size:22px;color:{gate_color}">{report.overall_score}</b> / 100</span>
  </div>
  <div class="meta" style="margin-top:12px">
    稿件 {_esc(report.doc_title)} &nbsp;|&nbsp; 文体 {report.doc_type}
    &nbsp;|&nbsp; 风格预设 {report.style_preset or '未指定'}
    &nbsp;|&nbsp; 正文 {report.char_count} 字 / {report.paragraph_count} 段
    &nbsp;|&nbsp; 术语库 <code>{report.terminology_version}</code>
    &nbsp;|&nbsp; 推理后端 <code>{report.llm_backend}</code>
    {'（已降级为规则引擎）' if report.degraded else ''}
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    kpi_row([
        (str(counts["FATAL"]), "致命问题", SEV_META["FATAL"]["color"]),
        (str(counts["MAJOR"]), "严重问题", SEV_META["MAJOR"]["color"]),
        (str(counts["MINOR"]), "一般问题", SEV_META["MINOR"]["color"]),
        (str(counts["INFO"]), "提示", SEV_META["INFO"]["color"]),
        (str(len(report.issues)), "问题总数", "#0A1F44"),
        (f"{report.overall_score}", "综合评分", gate_color),
    ])
    st.write("")

    c1, c2, c3 = st.columns([1.1, 1, 1.15], gap="medium")
    with c1:
        st.plotly_chart(chart_dimensions(report), **STRETCH,
                        config={"displayModeBar": False})
    with c2:
        st.plotly_chart(chart_severity(report), **STRETCH,
                        config={"displayModeBar": False})
    with c3:
        st.plotly_chart(chart_categories(report), **STRETCH,
                        config={"displayModeBar": False})

    with st.expander("维度得分明细与审校结论", expanded=False):
        score_bars(report)
        st.markdown("")
        st.markdown(f"<div class='summary-box'>{_esc(report.summary)}</div>",
                    unsafe_allow_html=True)

    tabs = st.tabs(["划改稿", "问题清单", "数据复算", "批评修正",
                    "修订留痕", "执行轨迹", "导出"])

    with tabs[0]:
        st.markdown(
            "<div class='legend'>"
            f"<span><i style='background:#FBD5D5;border:1px solid #E54545'></i>致命</span>"
            f"<span><i style='background:#FAE0C6;border:1px solid #E08A3C'></i>严重</span>"
            f"<span><i style='background:#D8EEDF;border:1px solid #4CAF7D'></i>一般</span>"
            f"<span><i style='background:#DCE8F8;border:1px solid #6A9BD8'></i>提示</span>"
            "<span style='margin-left:auto'>鼠标悬停色块可查看判定理由</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<pre class='manuscript'>{highlight_manuscript(text, report.issues)}</pre>",
            unsafe_allow_html=True,
        )

    with tabs[1]:
        fc1, fc2, fc3 = st.columns([1.6, 1.2, 1], gap="medium")
        with fc1:
            sev_pick = st.multiselect(
                "严重度筛选", ["致命", "严重", "一般", "提示"],
                default=["致命", "严重", "一般", "提示"], key="sev_filter")
        with fc2:
            grouped_all = report.by_category()
            cat_pick = st.multiselect("类别筛选", list(grouped_all.keys()),
                                      default=list(grouped_all.keys()), key="cat_filter")
        with fc3:
            order = st.selectbox("排序", ["按严重度", "按位置"], key="order_by")

        label_to_sev = {v["label"]: k for k, v in SEV_META.items()}
        wanted = {label_to_sev[x] for x in sev_pick}
        shown = [i for i in report.issues
                 if i.severity.value in wanted and i.category.value in cat_pick]
        shown.sort(key=(lambda i: (-i.severity.weight, i.span.start))
                   if order == "按严重度" else (lambda i: i.span.start))
        st.caption(f"符合筛选条件的问题 {len(shown)} / {len(report.issues)} 条")
        for issue in shown:
            st.markdown(issue_card(issue), unsafe_allow_html=True)
        if not shown:
            st.info("当前筛选条件下没有命中问题。", icon=None)

    with tabs[2]:
        records = report.meta.get("fact_check") or []
        if not records:
            st.info("本次无数据类或文献类结论需要复算。", icon=None)
        else:
            st.caption("复算由确定性算法完成，不依赖模型，结果可作签发依据。")
            for r in records:
                badge = {"confirmed_error": ("已确认为错误", "#E54545"),
                         "needs_human": ("需人工核对", "#D2691E")}.get(
                    r.get("verdict"), (str(r.get("verdict")), "#6B7688"))
                st.markdown(
                    f"""<div class="card" style="padding:15px 18px">
  <div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap">
    <span class="chip" style="background:{badge[1]}">{badge[0]}</span>
    <span class="mono">{_esc(r.get('issue_rule'))}</span>
    <span style="font-size:13px;color:#38445C">{_esc(r.get('statement'))[:110]}</span>
  </div>
  <div class="meta">复算过程：<code>{_esc(r.get('formula'))}</code></div>
  <div class="meta">处理建议：{_esc(r.get('advice'))}</div>
  <div class="meta">信源：{_esc(r.get('source',''))} · 置信度 {r.get('confidence')}</div>
</div>""",
                    unsafe_allow_html=True,
                )

    with tabs[3]:
        notes = report.meta.get("critic_notes") or []
        if not notes:
            st.success("未产生需要驳回或降级的审校意见。", icon=None)
        else:
            st.caption("批评修正 Agent 以「驳回者」视角复核并行召回的结论，"
                       "用于把误报压到最低。被驳回的问题仍保留理由，支持申诉。")
            for n in notes:
                st.markdown(
                    f"""<div class="card" style="padding:14px 18px">
  <div style="font-size:13px;color:#38445C">
    <span class="mono">{_esc(n.get('issue_id'))}</span>
    &nbsp;→&nbsp;<b>{_esc(n.get('action'))}</b>
    &nbsp;<span class="meta" style="display:inline">（{_esc(n.get('by'))}）</span></div>
  <div class="meta">{_esc(n.get('reason'))}</div>
</div>""",
                    unsafe_allow_html=True,
                )

    with tabs[4]:
        rev = (report.meta.get("llm_notes") or {}).get("revision") or {}
        if not rev:
            st.info("本次未生成修订稿。", icon=None)
        else:
            st.markdown(
                f"自动应用 **{rev.get('changed', 0)}** 处安全修改；"
                f"保留 **{len(rev.get('skipped', []))}** 处高风险问题交由人工处理。"
                "（致命政治性问题、数据矛盾、逻辑缺陷一律只出建议不改稿。）"
            )
            edits = rev.get("edits") or []
            if edits:
                st.dataframe(
                    pd.DataFrame([{"规则": e["rule_id"], "原文": e["before"],
                                   "改为": e["after"]} for e in edits]),
                    **STRETCH, hide_index=True, height=min(420, 60 + 35 * len(edits)),
                )
            skipped = rev.get("skipped") or []
            if skipped:
                with st.expander(f"保留待人工处理的 {len(skipped)} 处问题"):
                    st.dataframe(
                        pd.DataFrame([{"严重度": SEV_META[s["severity"]]["label"],
                                       "类别": s["category"], "原因": s.get("skip_reason", ""),
                                       "问题": s["message"]} for s in skipped]),
                        **STRETCH, hide_index=True,
                    )
            if rev.get("revised_preview"):
                with st.expander("修订稿预览（前 400 字）"):
                    st.markdown(
                        f"<pre class='manuscript'>{_esc(rev['revised_preview'])}</pre>",
                        unsafe_allow_html=True)

    with tabs[5]:
        for t in report.to_dict().get("trace", []):
            el = t.get("elapsed_ms")
            st.markdown(
                f"""<div class="trace-item{' skipped' if t.get('skipped') else ''}">
  <div style="font-size:13.5px;font-weight:700;color:#0A1F44">{_esc(t.get('agent'))}
    <span class="mono" style="font-weight:400">{_esc(t.get('node'))}</span></div>
  <div class="t">{_esc(t.get('ts'))}{f' · {el} ms' if el is not None else ''}
    {' · 本轮未激活' if t.get('skipped') else ''}</div>
  <div class="s">{_esc(t.get('summary'))}</div>
</div>""",
                unsafe_allow_html=True,
            )

    with tabs[6]:
        stem = Path(source_name).stem or "document"
        d1, d2, d3 = st.columns(3, gap="medium")
        with d1:
            st.download_button("下载 Markdown 报告", render_markdown(report, text),
                               file_name=f"{stem}_审校报告.md", mime="text/markdown",
                               **STRETCH)
        with d2:
            st.download_button("下载 JSON 结果", render_json(report),
                               file_name=f"{stem}_审校结果.json", mime="application/json",
                               **STRETCH)
        with d3:
            st.download_button("下载 HTML 报告", render_html(report, text),
                               file_name=f"{stem}_审校报告.html", mime="text/html",
                               **STRETCH)
        c4, c5 = st.columns(2, gap="medium")
        with c4:
            st.download_button("下载 Word 报告", export_docx(report, text),
                               file_name=f"{stem}_审校报告.docx",
                               mime="application/vnd.openxmlformats-"
                                    "officedocument.wordprocessingml.document",
                               **STRETCH)
        with c5:
            st.download_button("下载 PDF 报告", export_pdf(report, text),
                               file_name=f"{stem}_审校报告.pdf",
                               mime="application/pdf", **STRETCH)
        with st.expander("JSON 结构预览（前 2000 字符）"):
            st.code(render_json(report)[:2000], language="json")


def page_audit() -> None:
    hero(
        "INTERACTIVE REVIEW",
        "在线审校",
        "上传或粘贴待审稿件，选择目标文体后运行。系统会并行执行五个专业审校 Agent，"
        "经批评修正与事实复算后输出带定位、依据与置信度的完整报告。",
        ["文档结构解析", "病灶画像路由", "5 路并行审校", "批评修正环", "确定性复算"],
    )

    with st.container(border=False):
        mode = st.radio("稿件来源", ["使用内置示例稿件", "上传文件", "粘贴文本"],
                        horizontal=True, label_visibility="collapsed")

    text, source_name = "", ""
    if mode == "使用内置示例稿件":
        text = load_sample()
        source_name = "demo_report.md"
        st.caption("内置示例稿件：1393 字，人工埋入 30 余处类型化错误，用于验证各引擎检出能力。")
    elif mode == "上传文件":
        up = st.file_uploader(
            "上传稿件（支持 Markdown / 纯文本 / Word / PDF）",
            type=["md", "txt", "markdown", "docx", "pdf"],
            help="Word 与 PDF 会自动抽取正文文本后送审；PDF 需包含可复制文本（扫描件/加密件暂不支持）。",
        )
        if up is not None:
            raw = up.read()
            try:
                text = extract_upload_text(raw, up.name)
                source_name = up.name
                if not text.strip():
                    st.warning("未能从文件中提取到文本内容，请确认文件未加密且包含可复制文本。")
                else:
                    st.caption(f"已读取 {up.name}，{len(text)} 字符。")
            except Exception as exc:  # 解析失败时给出友好提示而非崩溃
                text = ""
                st.error(f"文件解析失败：{exc}")
    else:
        text = st.text_area("粘贴稿件正文", height=300,
                            placeholder="将调研专报或论文正文粘贴到此处…")
        source_name = "粘贴文本.md"

    c1, c2, c3 = st.columns([1, 1, 1.2], gap="medium")
    with c1:
        doc_type = st.selectbox("稿件类型", ["调研专报", "学术期刊", "研究报告",
                                            "新闻通稿", "公文", "论文"], index=0)
    with c2:
        preset = st.selectbox("目标文体预设", ["公文专报", "学术期刊", "新闻通稿"], index=0)
    with c3:
        st.write("")
        st.caption("目标文体用于风格量化诊断；稿件类型用于调度决策。")

    disabled = not (text and text.strip())
    if st.button("开始审校", type="primary", disabled=disabled):
        run_audit(text, source_name, doc_type, preset)

    if st.session_state.get("report") is not None:
        st.write("")
        st.markdown("---")
        render_report(st.session_state["report"], st.session_state["report_text"],
                      st.session_state.get("source_name", "document"))
    elif disabled:
        st.info("请先提供待审稿件。", icon=None)


# ======================================================================
# 页面：术语库
# ======================================================================
def page_terminology() -> None:
    agent = get_agent()
    store = agent.store
    hero(
        "TERMINOLOGY GOVERNANCE",
        "术语库管理",
        "术语库的生命力在于动态更新。系统支持从官方文件、内网接口与本单位台账抽取候选术语，"
        "经影子校验与人工审批后入库，并保证历史审校结论可复现。",
        [f"版本 {store.version}", f"标准条目 {len(store.entries)}",
         f"禁用模式 {len(store.banned)}", "影子校验", "热重载"],
    )

    tab1, tab2, tab3 = st.tabs(["术语条目", "禁用与慎用模式", "动态更新（影子校验）"])

    with tab1:
        f1, f2 = st.columns([1.4, 1], gap="medium")
        with f1:
            kw = st.text_input("检索术语（支持标准表述、变体、标签）", key="term_kw").strip()
        with f2:
            sev_f = st.multiselect("严重度", ["FATAL", "MAJOR", "MINOR", "INFO"],
                                   default=["FATAL", "MAJOR", "MINOR", "INFO"], key="term_sev")
        rows = []
        for e in store.entries:
            if e.severity.value not in sev_f:
                continue
            hay = f"{e.term} {' '.join(e.variants)} {' '.join(e.tags)} {e.source}"
            if kw and kw not in hay:
                continue
            rows.append({
                "编号": e.id,
                "严重度": SEV_META[e.severity.value]["label"],
                "类型": {"literal": "固定表述", "ordering": "组合提法顺序",
                         "semantic": "语义匹配", "deprecated": "提法演进"}.get(e.type, e.type),
                "标准表述": e.term,
                "错误变体数": len(e.variants),
                "标签": "、".join(e.tags),
                "出处": e.source,
            })
        st.caption(f"命中 {len(rows)} / {len(store.entries)} 条")
        st.dataframe(pd.DataFrame(rows), **STRETCH, hide_index=True,
                     height=430)
        pick = st.selectbox("查看条目详情", [e.id for e in store.entries], key="term_detail")
        entry = store.by_id(pick)
        if entry:
            st.markdown(
                f"""<div class="card">
  <h4>{_esc(entry.term)}</h4>
  <div class="meta">编号 <code>{entry.id}</code> · 严重度
    <b style="color:{SEV_META[entry.severity.value]['color']}">
    {SEV_META[entry.severity.value]['label']}</b> · 置信度 {entry.confidence}
    · 出处 {_esc(entry.source)}</div>
  <div class="desc" style="margin-top:10px">{_esc(entry.explain)}</div>
  <div class="meta" style="margin-top:10px">错误变体：
    {"、".join(f"<code>{_esc(v)}</code>" for v in entry.variants) or "（无）"}</div>
  {f"<div class='meta'>规范顺序：{' → '.join(entry.order)}</div>" if entry.order else ""}
</div>""",
                unsafe_allow_html=True,
            )

    with tab2:
        rows = [{"编号": b.id, "严重度": SEV_META[b.severity.value]["label"],
                 "类别": b.category, "说明": b.message,
                 "正则": b.pattern, "处置建议": b.suggestion, "出处": b.reference}
                for b in store.banned]
        st.dataframe(pd.DataFrame(rows), **STRETCH, hide_index=True,
                     height=420)

    with tab3:
        st.markdown(
            "**更新流程**：句式抽取候选 → 剥离动词噪声 → 包含关系去重 → 写入待审区 → "
            "与线上版本做影子校验（diff）→ 人工审批后写入最高优先级的本地台账。"
            "整个过程不直接改动中央术语库，避免机器改错术语污染全量稿件。"
        )
        sample = st.text_area(
            "输入一段政策文件正文，演示候选术语抽取",
            value=("要坚持以人民为中心的发展思想，坚持不懈用习近平新时代中国特色社会主义思想凝心铸魂，"
                   "坚持稳中求进工作总基调，牢牢把握高质量发展这个首要任务，"
                   "牢固树立绿水青山就是金山银山的理念，铸牢中华民族共同体意识。"),
            height=140,
        )
        if st.button("抽取候选术语", type="primary"):
            updater = TerminologyUpdater(store, agent.settings)
            with st.status("执行抽取与影子校验…", expanded=True) as status:
                st.write("按政策文件句式模板扫描正文")
                cands = updater.extract_candidates(sample, "streamlit-demo", "在线演示输入")
                st.write(f"得到 {len(cands)} 条候选，正在去重与打分")
                time.sleep(0.2)
                status.update(label=f"抽取完成：{len(cands)} 条候选（未生效）", state="complete")
            if cands:
                st.dataframe(
                    pd.DataFrame([{"候选术语": c.term, "承载句式": c.cue,
                                   "置信度": c.confidence, "建议严重度": c.severity}
                                  for c in cands]),
                    **STRETCH, hide_index=True,
                )
                st.caption("这些候选会进入 data/terminology/pending/ 待审区，"
                           "需人工核对后执行 terms approve 才会生效。")
            else:
                st.info("未从该文本中抽取出候选术语。", icon=None)


# ======================================================================
# 页面：Agent 架构
# ======================================================================
def page_architecture() -> None:
    agent = get_agent()
    hero(
        "MULTI-AGENT ORCHESTRATION",
        "Agent 架构",
        "基于 LangGraph 构建的多 Agent 协作体系：动态路由决定激活哪些专业 Agent，"
        "并行 fan-out 提升吞吐，批评修正环持续压制误报，事实复算保证结论可作签发依据。",
        ["12 节点", "条件边动态路由", "并行 fan-out/fan-in", "批评修正环",
         "确定性复算", "全链路 trace"],
    )

    st.markdown("#### 工作流结构")
    st.code(agent.graph_mermaid(), language="mermaid")
    st.caption("该 Mermaid 源码可直接粘贴到支持 Mermaid 的编辑器中渲染。")

    st.markdown("#### 节点职责")
    nodes = agent.describe_agents()
    for row in range(0, len(nodes), 3):
        cols = st.columns(3, gap="medium")
        for col, node in zip(cols, nodes[row:row + 3]):
            with col:
                st.markdown(
                    f"""<div class="card glass" style="min-height:150px">
  <div style="font-size:13.5px;font-weight:700;color:#0A1F44">{node['label']}</div>
  <div class="mono" style="margin:6px 0">{node['node']}</div>
  <div class="desc" style="font-size:12.5px">{node['role']}</div>
  <div class="meta">实现类 <code>{node['type']}</code></div>
</div>""",
                    unsafe_allow_html=True,
                )

    st.write("")
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.markdown(
            """<div class="card">
  <h4>为什么是「规则 + 模型」混合，而不是纯 LLM 审校</h4>
  <div class="desc">
  纯 LLM 做审校在生产环境有三个致命问题：<b>不可复现</b>（两次运行结果不同，编辑无法信任、
  无法申诉）、<b>幻觉</b>（会发明并不存在的术语规范，或漏掉明确红线）、
  <b>不可审计</b>（无法回答「为什么判这句有问题」）。<br><br>
  因此本系统把审校拆成两类任务：有确定答案的判定（术语对错、算术自洽、文献格式）
  交给<b>确定性规则与算法</b>；需要语义判断的（某条批评是否成立、如何改写更严谨）
  交给<b>本地模型</b>，且模型产出的每条判断都必须经过批评修正 Agent 复核。
  模型不可用时整条链路自动降级，不产生半成品结论。
  </div></div>""",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            """<div class="card">
  <h4>误报压制：审校系统能不能被用起来的关键</h4>
  <div class="desc">
  审校系统的信任成本极高——一次误报就会让编辑放弃使用。系统做了五层压制：<br><br>
  <b>长引述豁免</b>：仅引号内容 ≥ 20 字才算原文引用，短引号里的术语短语照报；<br>
  <b>否定语境豁免</b>：「不得使用 X」中的 X 不是错误；<br>
  <b>参考文献区豁免</b>：文献题名不可改；<br>
  <b>标准表述保护</b>：命中片段本身是标准表述时降级为提示，防止把正确的改错；<br>
  <b>重叠消解</b>：同区间只保留严重度、匹配长度、规则特异性最高的一条。
  </div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 评分与签发门禁")
    st.markdown(
        """<div class="card">
  <div class="desc">
  <b>维度得分</b>：<code>score = 100 · exp(−penalty / scale)</code>，penalty 为该维度所有问题的严重度权重之和
  （致命 100 / 严重 20 / 一般 5 / 提示 1）。指数衰减避免「问题多就一律零分」的退化。<br><br>
  <b>综合评分</b>：按维度加权——政治 0.35、专业 0.25、语文 0.18、逻辑 0.17、风格 0.05。
  政治权重最高，因为它是「一票否决」维度。<br><br>
  <b>签发门禁</b>：存在任一致命问题 → <span style="color:#E54545">BLOCK</span>（不予签发）；
  综合分 &lt; 75 或严重问题 &gt; 5 → <span style="color:#D2691E">REVIEW</span>（退改）；
  其余 → <span style="color:#2E8B57">PASS</span>。
  </div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 数据自洽性复算示例")
    st.markdown(
        """<div class="card">
  <div class="desc">
  以段落（而非句子）为分析单元——中文公文中「分项—合计—占比」三者常跨句表述：<br><br>
  <code style="display:block;padding:12px;line-height:1.9">
  2026 年上半年，共归集数据 862 万条……其中，民政数据 312 万条，社保数据 268 万条，<br>
  卫健数据 196 万条，三项合计 776 万条，占归集总量的 95.8%。</code><br>
  算法用「其中 / 分别为」与「合计」提示词切分<b>分项集合</b>（这一步避免把上一句的
  「总费用 420 万元」误当作「三项合计 440 万元」的分项），复算得
  <code>776 ÷ 862 = 90.0%</code>，与文中 95.8% 不符，即报为严重问题并附带复算公式。
  </div></div>""",
        unsafe_allow_html=True,
    )


# ======================================================================
# 页面：审校历史与对比
# ======================================================================
def page_history() -> None:
    hero(
        "AUDIT HISTORY",
        "审校历史与对比",
        "同一会话内多次审校的轻量记录与横向对比。每次审校自动留痕，可比较综合分、"
        "问题严重度分布与各维度趋势。（记录保存在当前会话，刷新页面会清空。）",
        ["分数对比", "问题分布", "维度趋势"],
    )

    hist = st.session_state.get("audit_history", [])
    if not hist:
        st.info("还没有审校记录。前往「在线审校」上传稿件跑一次，记录会自动出现在这里。",
                icon="🕒")
        return

    rename = {"ts": "时间", "source": "来源", "doc_type": "文体", "score": "综合分",
              "gate": "门禁", "FATAL": "致命", "MAJOR": "严重", "MINOR": "一般",
              "INFO": "提示", "issue_count": "问题数"}
    show_cols = ["ts", "source", "doc_type", "score", "gate",
                 "FATAL", "MAJOR", "MINOR", "INFO", "issue_count"]
    raw = pd.DataFrame(hist)
    df = raw[show_cols].rename(columns=rename)
    st.markdown("#### 历次审校记录")
    st.dataframe(df, **STRETCH, hide_index=True)

    c1, c2 = st.columns([1, 1], gap="large")
    gate_color = {"BLOCK": "#E54545", "REVIEW": "#D2691E", "PASS": "#2E8B57"}
    with c1:
        st.markdown("#### 综合分对比")
        fig = go.Figure()
        fig.add_bar(x=raw["source"], y=raw["score"],
                    marker_color=[gate_color.get(g, "#888") for g in raw["gate"]],
                    text=raw["score"], textposition="outside")
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20), yaxis_range=[0, 105],
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Noto Sans SC, sans-serif"))
        st.plotly_chart(fig, **STRETCH)
    with c2:
        st.markdown("#### 问题严重度分布")
        sev = [("FATAL", "致命", "#E54545"), ("MAJOR", "严重", "#D2691E"),
               ("MINOR", "一般", "#2E8B57"), ("INFO", "提示", "#2F54EB")]
        fig2 = go.Figure()
        for key, label, col in sev:
            fig2.add_bar(name=label, x=raw["source"], y=raw[key], marker_color=col)
        fig2.update_layout(barmode="stack", margin=dict(l=20, r=20, t=10, b=20),
                           legend=dict(orientation="h", y=-0.18),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Noto Sans SC, sans-serif"))
        st.plotly_chart(fig2, **STRETCH)

    dims = list(hist[0]["dimensions"].keys())
    if dims:
        st.markdown("#### 维度得分趋势")
        fig3 = go.Figure()
        for d_ in dims:
            fig3.add_trace(go.Scatter(
                x=raw["source"], y=[h["dimensions"].get(d_) for h in hist],
                mode="lines+markers", name=d_))
        fig3.update_layout(margin=dict(l=20, r=20, t=10, b=20), yaxis_range=[0, 105],
                           legend=dict(orientation="h", y=-0.2),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Noto Sans SC, sans-serif"))
        st.plotly_chart(fig3, **STRETCH)

    st.caption(f"当前会话共 {len(hist)} 条审校记录。")
    if st.button("清除历史记录", **STRETCH):
        st.session_state["audit_history"] = []
        st.rerun()


# ======================================================================
# 页面：使用说明
# ======================================================================
def page_about() -> None:
    hero(
        "OPERATIONS & BOUNDARIES",
        "使用说明与边界",
        "系统定位是「辅助审校」——它把编辑从机械比对中解放出来，但签发责任仍在编辑。"
        "每一条问题都附带规则编号、依据出处与置信度，支持申诉与回溯。",
        ["本地部署", "数据不出域", "可复现结论", "可申诉留痕"],
    )
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.markdown(
            """<div class="card">
  <h4>本地部署与模型接入</h4>
  <div class="desc">
  全部推理可运行在内网/本机，<b>稿件不出域</b>。修改 <code>config/settings.yaml</code> 即可切换后端：<br><br>
  <b>Ollama</b>：<code>ollama pull qwen2.5:14b-instruct</code>，单机最简，适合科室级部署；<br>
  <b>vLLM</b>：<code>vllm serve Qwen2.5-14B-Instruct --port 8000</code>，高并发，适合全院/全社共享；<br>
  <b>LM Studio / Xinference</b>：图形化运维，开放 OpenAI 兼容端口即可。<br><br>
  也可用环境变量覆盖，例如
  <code>PFRD_LLM__OPENAI_COMPATIBLE__BASE_URL</code>。<br>
  本地模型不可达时自动降级为纯规则引擎，报告上标注 <code>degraded=True</code>，工作流不中断。
  </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="card">
  <h4>术语库治理岗位职责</h4>
  <div class="desc">
  <code>core_terms.yaml</code> 是示例库，覆盖高频易错项。生产部署必须按
  「<code>terms update</code> 抽取候选 → 影子校验 → <code>terms approve</code> 审批入库」
  的流程对接权威语料持续增量，并明确责任岗位与复核周期。术语库会随时间失效——
  这是长期运营工作，不是一次性交付物。
  </div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="card" style="border-left:4px solid #E54545">
  <h4>边界与已知限制</h4>
  <div class="desc">
  1. <b>术语库需持续运营</b>，示例库不能直接作为生产口径。<br>
  2. <b>语义匹配依赖嵌入模型质量</b>。未部署嵌入模型时退化为词形近似
  （<code>embedding_mode=lexical</code>），能捕获改写型错误，但召回率会下降，报告中已显式标注。<br>
  3. <b>自动修订只处理安全修改</b>。致命政治性问题、数据矛盾、逻辑缺陷一律只出建议不改稿——
  这类问题必须由人决定，机器擅自改写会带来更大风险。<br>
  4. <b>重复发表检测是启发式的</b>。输出「疑似」与相似度分值，不等于学术不端认定，必须由人复核。<br>
  5. <b>逻辑缺陷检测给出的是线索</b>而非判定，置信度普遍在 0.70–0.88 区间，报告中如实呈现。<br>
  6. <b>系统不为结论背书</b>。签发责任仍在编辑与终审人。
  </div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 命令行等价操作")
    st.code(
        """# 查看运行环境与 Agent 清单
python -m proofread_agent.cli info

# 审校稿件并导出三种格式
python -m proofread_agent.cli audit 稿件.md --doc-type 调研专报 --format markdown html json

# 术语库动态更新（影子模式，人工审批）
python -m proofread_agent.cli terms update
python -m proofread_agent.cli terms approve --file data/terminology/pending/pending_xxx.yaml

# 运行测试套件
python -m unittest discover -s tests -v""",
        language="bash",
    )
    st.caption("界面操作与命令行走的是同一套引擎与流水线，结果完全一致。")


# ======================================================================
# 侧边栏与路由
# ======================================================================
NAV_PAGES = ["概览", "在线审校", "审校历史", "术语库", "Agent 架构", "使用说明"]


def _nav_default_index() -> int:
    """支持 `?page=在线审校` 深链接，便于把某个页面直接发给同事。"""
    try:
        wanted = st.query_params.get("page", "")
    except Exception:
        try:
            raw_q = st.experimental_get_query_params().get("page", "")
            wanted = raw_q[0] if isinstance(raw_q, list) else raw_q
        except Exception:
            return 0
    return NAV_PAGES.index(wanted) if wanted in NAV_PAGES else 0


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """<div style="padding:8px 4px 18px">
  <div style="font-size:11.5px;letter-spacing:2.2px;color:#E8C766;font-weight:600">
    MULTI-AGENT PROOFREADING</div>
  <div style="font-family:'Noto Serif SC',serif;font-size:23px;font-weight:700;
              color:#FFFFFF;margin-top:8px;line-height:1.35">智能审校工作台</div>
  <div style="font-size:12px;color:#AFC2E8;margin-top:8px;line-height:1.7">
    本地大模型 · 稿件不出域<br>政治术语 / 语文规范 / 逻辑 / 专业深度 / 风格</div>
</div>""",
            unsafe_allow_html=True,
        )
        page = option_menu(
            menu_title=None,
            options=["概览", "在线审校", "审校历史", "术语库", "Agent 架构", "使用说明"],
            icons=["grid-1x2", "shield-check", "clock-history", "journal-text",
                   "diagram-3", "info-circle"],
            default_index=_nav_default_index(),
            styles={
                "container": {"padding": "8px 6px", "background-color": "#0E2450",
                              "border-radius": "14px",
                              "border": "1px solid rgba(201,162,39,0.35)"},
                "icon": {"color": "#E8C766", "font-size": "15px"},
                "nav-link": {"font-size": "14.5px", "color": "#FFFFFF", "font-weight": "600",
                             "margin": "3px 0", "padding": "10px 14px",
                             "border-radius": "11px",
                             "--hover-color": "rgba(201,162,39,0.22)",
                             "border": "1px solid transparent"},
                "nav-link-selected": {
                    "background": "linear-gradient(90deg, rgba(201,162,39,0.35), rgba(201,162,39,0.10))",
                    "border": "1px solid rgba(201,162,39,0.75)", "color": "#FFE9A8",
                    "font-weight": "800", "border-left": "4px solid #E8C766"},
            },
        )
        st.markdown(
            "<div style='height:1px;background:rgba(201,162,39,0.28);margin:16px 4px'></div>",
            unsafe_allow_html=True,
        )
        info = agent_info()
        st.caption(
            f"编排引擎 {info.get('engine', '-')}\n\n"
            f"术语库 {info.get('terminology_version', '-')}"
            f"（{info.get('terminology_entries', '-')} 条）\n\n"
            f"推理后端 {info.get('llm_backend', '-')}\n\n"
            f"嵌入模式 {info.get('embedding_mode', '-')}"
        )
        if info.get("degraded"):
            st.caption("当前为纯规则引擎模式")
        if st.button("重新初始化引擎", **STRETCH):
            get_agent.clear()
            st.rerun()
        st.markdown(
            """<div style="font-size:11px;color:#7E90B8;margin-top:22px;line-height:1.8">
  系统为辅助审校工具，<br>签发责任仍在编辑与终审人。</div>""",
            unsafe_allow_html=True,
        )
    return page


def main() -> None:
    page = render_sidebar()
    {
        "概览": page_overview,
        "在线审校": page_audit,
        "审校历史": page_history,
        "术语库": page_terminology,
        "Agent 架构": page_architecture,
        "使用说明": page_about,
    }.get(page, page_overview)()


main()
