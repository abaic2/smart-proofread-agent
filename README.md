# 智能审校 Agent

> 基于**本地部署大语言模型**的多 Agent 协作校审系统，面向**调研专报、学术期刊**等专业文本，
> 提供政治术语精准校验、语文规范审核、逻辑严密性评估、专业文本深度审校与风格迁移能力。

全部推理可运行在内网/本机（Ollama、vLLM、LM Studio、Xinference），**稿件不出域**；
本地模型不可达时自动降级为纯规则引擎，工作流仍能端到端跑通，不会因模型故障中断校审流程。

---

## 一、能力矩阵

| 模块 | 能力 | 典型检出项 | 实现要点 |
|---|---|---|---|
| **政治术语精准审校引擎** | 称谓与固定表述、组合提法顺序、提法演进（时态）、涉台涉港涉澳红线、禁用慎用词 | 「以习近平总书记为核心的党中央」→「以习近平同志为核心的党中央」；「四个意识」顺序错误；「一带一路战略」→「共建'一带一路'」 | **三级匹配**：AC 自动机精确匹配 → 倒排索引+编辑距离近似匹配 → 本地嵌入向量语义匹配 |
| **语文规范审核** | 错别字、易混词误用、标点（GB/T 15834）、数字与单位、冗余欧化、长句、全半角 | 「下降了 3.2 倍」；「近 200 余人」；「截止目前」→「截至目前」 | 规则库 + 语境判断，零误报优先 |
| **逻辑严密性评估** | 推论越界（以偏概全）、因果跳步、绝对化断言、论证强度冲突、指代不明、数量词与数据矛盾 | 「12 个乡镇使用率 62% → 因此全国已全面建成」；「众所周知」缺据 | 论点—论据—保证三段式信号检测 + 范围一致性推理 |
| **专业文本深度审校** | 数据合规、格式规范、核心观点重复发表、参考文献（GB/T 7714-2015） | 分项之和 ≠ 合计；占比 108%；776/862=90.0% 却写 95.8%；文献年份 2031；正文引 [5] 却无条目 | 数值三元组解析 + 算术自洽复算；SimHash→MinHash→TF-IDF 三层查重；文献著录项逐条比对 |
| **风格迁移** | 文体量化画像、差距诊断、迁移动作清单、受控改写 | 数据标注率 0.143（目标 ≥0.8）；专报缺"存在问题/工作建议"段落 | 量化指标（句长/口语密度/绝对化密度/标注率）+ do/avoid 词表 + 目标区间比对 |

**架构核心**：基于 **LangGraph** 构建 12 节点多 Agent 协作图，支持并行 fan-out/fan-in、
动态路由、批评修正环与确定性事实复算，全流程带审计轨迹。

---

## 二、快速开始

### 2.1 Web 工作台（Streamlit，推荐）

```bash
pip install -r requirements.txt
streamlit run app.py            # 浏览器打开 http://localhost:8501
```

工作台含六个页面：**概览**（能力矩阵与运行环境）、**在线审校**（上传/粘贴稿件 →
运行 → 划改稿、问题清单、数据复算、批评修正、修订留痕、执行轨迹、导出）、
**审校历史**（会话内多次审校的分数/严重度/维度趋势对比）、
**术语库**（条目检索、禁用模式、动态更新与影子校验演示）、**Agent 架构**
（Mermaid 工作流、节点职责、设计取舍）、**使用说明**（本地部署、岗位职责、边界限制）。

稿件支持 **Markdown / 纯文本 / Word(.docx) / PDF** 上传（Word/PDF 自动抽取正文与表格；
PDF 需含可复制文本）；审校结果可导出 **Markdown / JSON / HTML / Word(.docx) / PDF**
五种格式（Word 与 PDF 由 `report.exporters` 生成，PDF 内嵌 STSong-Light 中文字体）。

### 2.2 命令行

```bash
# 查看运行环境（含 Agent 清单、术语库版本、模型后端）
python -m proofread_agent.cli info           # 需 PYTHONPATH=src，或先 pip install -e .

# 跑一遍内置示例（含 30+ 处人工埋点错误）
python -m proofread_agent.cli demo --limit 80

# 审校自己的稿件
python -m proofread_agent.cli audit 你的稿件.md \
    --doc-type 调研专报 --preset 公文专报 --format markdown html json
```

> Windows PowerShell 下建议先设置路径：`$env:PYTHONPATH="src"`

输出：`outputs/<稿件名>_audit.{md,html,json}`——HTML 报告含**总览 / 划改稿 / 问题清单 /
数据复算记录 / 批评修正记录 / 自动修订留痕 / Agent 执行轨迹**七个视图。

### 作为库使用

```python
from proofread_agent import ProofreadAgent

agent  = ProofreadAgent()                                   # 读取 config/settings.yaml
report = agent.audit_file("data/samples/demo_report.md", doc_type="调研专报")

print(report.overall_score, report.release_gate)            # 48.8 BLOCK
for issue in report.sorted_issues()[:10]:
    print(issue.severity.label, issue.line, issue.message, "→", issue.suggestion)

agent.export(report, "outputs")                             # 导出 md / html / json
print(agent.graph_mermaid())                                # 导出工作流图
```

---

## 三、接入本地大模型

修改 `config/settings.yaml`（也可用环境变量覆盖，如 `PFRD_LLM__OPENAI_COMPATIBLE__BASE_URL`）：

```yaml
llm:
  backend: openai_compatible          # 或 ollama
  openai_compatible:                  # vLLM / LM Studio / Xinference / TGI
    base_url: "http://127.0.0.1:8000/v1"
    model: "Qwen2.5-14B-Instruct"
  ollama:
    base_url: "http://127.0.0.1:11434"
    model: "qwen2.5:14b-instruct"
  fallback_to_rule_engine: true       # 模型不可达时降级，不中断流程
  embedding:
    backend: auto                     # auto | ollama | openai_compatible | lexical
    model: "bge-m3"                   # 语义级术语匹配与段落查重使用
```

| 部署方式 | 命令示例 | 说明 |
|---|---|---|
| Ollama | `ollama pull qwen2.5:14b-instruct` | 单机最简，适合科室级部署 |
| vLLM | `vllm serve Qwen2.5-14B-Instruct --port 8000` | 高并发，适合全院/全社共享 |
| LM Studio / Xinference | 启动后开放 OpenAI 兼容端口 | 图形化运维 |

模型在系统中的分工是**"判断"而非"生成"**：复核低置信度命中、给出改写示范、
撰写审校结论。所有硬性事实（术语对错、数据算术、文献格式）由确定性规则与算法负责，
避免幻觉污染审校结论。

---

## 四、术语库与动态更新

术语库分三层，优先级从高到低：

```
data/terminology/local_ledger.yaml    本单位增量台账（最高优先级，可覆写中央口径）
data/terminology/core_terms.yaml      中央政策术语标准库（30 条示例，含变体/顺序/语义种子）
data/terminology/banned_patterns.yaml 禁用词/慎用词模式库
```

**更新流程（影子模式，人工审批）**

```bash
# ① 从官方文件 / 内网接口 / 本单位台账拉取候选，写入待审区并做影子校验
python -m proofread_agent.cli terms update

# ② 人工核对 pending/*.yaml 后审批入库（写入最高优先级台账并热重载）
python -m proofread_agent.cli terms approve --file data/terminology/pending/pending_xxx.yaml

# 查看当前术语库
python -m proofread_agent.cli terms show --limit 30
```

设计要点：
* **版本可追溯**：每次加载/转正生成 `checksum` 与历史记录，报告记录
  `terminology_version`，保证历史审校结论可复现（编辑申诉时可回溯当时依据的版本）；
* **影子校验**：新版本先与线上版本比对差异，人工确认后才转正，避免"机器改错术语"污染全量稿件；
* **零停机热更新**：审批通过即重建 AC 索引与语义索引，无需重启服务。

---

## 五、目录结构

```
smart-proofread-agent/
├── app.py                          Streamlit Web 工作台（五页）
├── .streamlit/config.toml          主题与服务器配置
├── config/settings.yaml            全局配置（模型、匹配阈值、抑制策略、图编排参数）
├── data/
│   ├── terminology/                术语库（core_terms / banned_patterns / semantic_clusters / local_ledger）
│   ├── references/gbt7714.yaml     GB/T 7714-2015 著录规则
│   ├── corpus/                     历史稿件语料（用于跨稿件观点重复检测）
│   └── samples/demo_report.md      示例稿件（内置 30+ 处埋点错误）
├── src/proofread_agent/
│   ├── llm/client.py               本地模型适配（Ollama / OpenAI 兼容 / 离线兜底）+ 嵌入客户端
│   ├── terminology/                术语库存储（版本化/影子校验）与动态更新器
│   ├── text/                       Aho-Corasick、编辑距离、SimHash/MinHash/TF-IDF、分句分段分词
│   ├── document.py                 文档对象（偏移映射、引述区间、参考文献区识别）
│   ├── engines/                    五大审校引擎（纯规则，零外部依赖，可独立测试）
│   ├── agents/                     LangGraph 节点：受理/调度/5 专家/汇聚/批评/复算/修订/报告
│   ├── report/                     问题模型（Issue/Span/Severity）与 md/html/json 渲染
│   └── cli.py                      命令行入口
├── tests/test_suite.py             49 个用例（算法、引擎、更新器、端到端）
└── docs/architecture.md            架构设计与技术决策说明
```

### 部署到 Streamlit Community Cloud

仓库根目录已备好 `app.py`、`requirements.txt`、`.streamlit/config.toml`、`runtime.txt`（Python 3.11）：

1. 把仓库推到 GitHub（Public）；
2. 打开 `https://share.streamlit.io/deploy?repository=abaic2/smart-proofread-agent&branch=main&mainModule=app.py`
   （新版控制台也可从 `https://streamlit.io/cloud` 连接仓库）；
3. 用 GitHub 账号授权并点击 Deploy（这一步需本人完成 OAuth）；
4. 之后每次 `git push`，云端 1–2 分钟自动重新部署。

云端无本地模型时自动以纯规则引擎运行（页面会明确提示 `degraded`），
所有硬性校验能力不受影响。

#### 通过 Secrets 启用云端大模型审校

默认配置指向本机 `127.0.0.1:8000`，云端不可达，故走规则引擎兜底。
若想让云端应用也做 LLM 分析，在 Streamlit Cloud 的 **App → Settings → Secrets**
填入你的 OpenAI 兼容端点（任意云上 vLLM / LM Studio / 厂商 API 均可）：

```toml
[llm]
backend = "openai_compatible"

[llm.openai_compatible]
base_url = "https://你的端点/v1"
api_key = "sk-..."
model = "模型名"
temperature = "0.1"
max_tokens = "2048"
timeout = "120"
```

应用启动时会把上述 Secrets 自动映射为 `PFRD_LLM__*` 环境变量并覆盖配置，
无需改代码。`fallback_to_rule_engine` 保持 `true` 即可在端点偶发不可达时自动降级。

---

## 六、工作流

```
START → 受理预检 ──► 调度 Supervisor ─┬─► 政治术语 Agent ─┐
                                     ├─► 语文规范 Agent ─┤
                                     ├─► 逻辑严密 Agent ─┼─► 结果汇聚 join
                                     ├─► 专业深度 Agent ─┤
                                     └─► 风格迁移 Agent ─┘
                                                          │
                    ┌──────────── 批评修正环 ◄────────────┘
                    ▼   （误报压制 / 一致性消解 / 模型复核，最多 2 轮）
              事实与数据复算 ──► 修订稿生成 ──► 报告聚合 ──► END
```

* **并行 fan-out/fan-in**：五个专家 Agent 在同一超步并发执行，共享 State 通过
  reducer（`merge_dict` / `extend_list`）安全归并；
* **动态路由**：Supervisor 依据"病灶画像"（政治词密度、数据点数量、是否含参考文献等）
  决定激活哪些 Agent，短稿不必跑全量；
* **批评修正环**：Critic 以"驳回者"角色反向工作，专门删掉站不住脚的问题——审校系统的
  信任成本极高，一次误报就会让编辑放弃使用；
* **降级可用**：未安装 LangGraph 时自动切换内置顺序执行器（同样节点、同样状态语义）。

详见 [`docs/architecture.md`](docs/architecture.md)。

---

## 七、测试与自检

```bash
python -m unittest discover -s tests -v      # 49 passed
```

覆盖：AC 自动机重叠匹配、编辑距离剪枝、SimHash/MinHash 近重复判定、文档结构解析、
五大引擎的正例与**反例**（如"合计正确时不得误报""局部样本推全域必须报""短引号内术语不得豁免"）、
术语库影子校验、修订稿偏移正确性、端到端门禁与轨迹完整性。

---

## 八、边界与已知限制（务必阅读）

1. **术语库是示例库**。`core_terms.yaml` 覆盖高频易错项，生产部署必须按
   "动态更新"流程对接权威语料持续增量，并明确责任岗位。
2. **语义匹配依赖嵌入模型质量**。未部署嵌入模型时退化为词形近似（`embedding_mode=lexical`），
   能捕获改写型错误（如"以人民为核心"），但**召回率会下降**，报告中有显式标注。
3. **自动修订只处理"安全修改"**。致命政治性问题、数据矛盾、逻辑缺陷一律**只出建议不改稿**——
   这类问题必须由人决定。
4. **重复发表检测是启发式的**。输出的是"疑似"与相似度分值，不等于学术不端认定，
   必须由人复核。
5. **不为结论背书**。系统是"辅助审校"，签发责任仍在编辑。报告中的每一条问题都附带
   规则 ID、依据出处与置信度，支持申诉与回溯。

---

## 九、许可证

内部交付项目，著作权归委托方所有。示例术语库内容来自公开权威文件的规范表述整理，
引用来源已在条目的 `source` 字段标注。
