"""智能审校 Agent 测试套件（纯标准库 unittest，无需额外依赖）。

运行::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofread_agent import ProofreadAgent, Document  # noqa: E402
from proofread_agent.engines import (  # noqa: E402
    AcademicEngine,
    LogicEngine,
    NormEngine,
    StyleEngine,
    TermEngine,
)
from proofread_agent.report import Severity  # noqa: E402
from proofread_agent.text import AhoCorasick, levenshtein  # noqa: E402
from proofread_agent.text.similarity import (  # noqa: E402
    MinHasher,
    combined_similarity,
    compare_fingerprints,
    paragraph_fingerprint,
    simhash_similarity,
)

DEMO = ROOT / "data" / "samples" / "demo_report.md"


class TestTextAlgorithms(unittest.TestCase):
    def test_aho_corasick_finds_all_and_overlaps(self):
        ac = AhoCorasick.from_patterns(["四个意识", "意识", "核心意识"])
        hits = ac.find_all("牢固树立四个意识，核心意识最重要")
        found = {p for _, _, p in hits}
        self.assertEqual(found, {"四个意识", "意识", "核心意识"})

    def test_aho_corasick_no_false_positive(self):
        ac = AhoCorasick.from_patterns(["以习近平同志为核心的党中央"])
        self.assertEqual(ac.find_all("我们坚决做到两个维护"), [])

    def test_levenshtein_bounds(self):
        self.assertEqual(levenshtein("部署", "布署", 3), 1)
        self.assertEqual(levenshtein("部署", "部署", 3), 0)
        self.assertGreater(levenshtein("完全不同的内容", "另外一句话", 2), 2)

    def test_simhash_near_duplicates(self):
        a = "通过对12个乡镇的实地调研发现，平台功能使用率仅为62%。"
        b = "通过对12个乡镇的实地调研发现，平台功能使用率仅为62%左右。"
        c = "中共中央关于加强党的政治建设的意见明确了两个维护的具体内涵。"
        fa = paragraph_fingerprint(a)
        fb = paragraph_fingerprint(b)
        fc = paragraph_fingerprint(c)
        self.assertGreater(combined_similarity(compare_fingerprints(fa, fb)), 0.85)
        self.assertLess(combined_similarity(compare_fingerprints(fa, fc)), 0.35)

    def test_minhash_jaccard_estimate(self):
        h = MinHasher(num_perm=128, shingle_size=3)
        s1 = h.signature(["a", "b", "c", "d", "e", "f"])
        s2 = h.signature(["a", "b", "c", "d", "e", "x"])
        self.assertGreater(MinHasher.estimate_jaccard(s1, s2), 0.5)
        self.assertLess(simhash_similarity(0, (1 << 64) - 1), 0.01)


class TestDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = Document.from_file(DEMO)

    def test_structure(self):
        self.assertGreater(len(self.doc.paragraphs), 10)
        self.assertGreater(len(self.doc.sentences), 20)
        self.assertEqual(len(self.doc.ref_items), 3)
        self.assertGreater(self.doc.ref_start, 0)

    def test_body_excludes_references(self):
        self.assertNotIn("电子政务, 2031", self.doc.body_text)
        self.assertIn("电子政务, 2031", self.doc.reference_text)
        self.assertTrue(self.doc.body_text.startswith("# 关于县域数字治理"))

    def test_line_and_paragraph_lookup(self):
        pos = self.doc.text.find("布署")
        self.assertGreater(self.doc.line_of(pos), 0)
        self.assertGreaterEqual(self.doc.paragraph_of(pos)["index"], 0)

    def test_quote_spans_detected(self):
        self.assertGreater(len(self.doc.quote_spans), 0)


class TestTermEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ProofreadAgent()
        cls.engine = TermEngine(cls.agent.settings, cls.agent.store, cls.agent.embedding)
        cls.doc = Document.from_file(DEMO)
        cls.result = cls.engine.run(cls.doc)
        cls.ids = {i.rule_id for i in cls.result.issues}

    def test_l1_exact_term_error(self):
        self.assertIn("PTR-0012", self.ids)          # 五位一体战略布局
        self.assertIn("PTR-0033", self.ids)          # 一带一路战略
        self.assertIn("PTR-0020", self.ids)          # 布署 -> 部署

    def test_ordering_detects_permutation_and_rebuilds(self):
        order = [i for i in self.result.issues if i.rule_id == "PTR-0010-ORDER"]
        self.assertEqual(len(order), 1)
        self.assertEqual(order[0].severity, Severity.FATAL)
        self.assertIn("政治意识、大局意识、核心意识、看齐意识", order[0].suggestion)

    def test_fuzzy_catches_semantic_word_swap(self):
        hits = [i for i in self.result.issues if i.rule_id.startswith("PTR-0060")]
        self.assertTrue(hits, "应通过近似/语义匹配捕获「以人民为核心」")
        self.assertIn("为中心", hits[0].suggestion)

    def test_no_false_positive_on_correct_standard_term(self):
        # 标准表述「习近平新时代中国特色社会主义思想」本身不得被报错
        for i in self.result.issues:
            self.assertNotEqual(i.original, "习近平新时代中国特色社会主义思想")

    def test_reference_section_and_quotes_exempt(self):
        doc = Document.from_text("参考文献\n[1] 张明. 关于五位一体战略布局的研究[J]. 求是, 2020(1): 1-2.")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.original == "五位一体战略布局"], [])

    def test_negated_context_exempt(self):
        doc = Document.from_text("文稿中不得使用五位一体战略布局这一提法。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.original == "五位一体战略布局"], [])

    def test_short_quoted_term_is_not_exempt(self):
        """术语级短语加引号不算"原文引述"，必须仍然报错（防止漏报红线）。"""
        doc = Document.from_text('文件要求落实"五位一体战略布局"。')
        res = self.engine.run(doc)
        self.assertTrue([i for i in res.issues if "五位一体" in (i.original or "")])

    def test_long_citation_is_exempt(self):
        long_quote = ("“全党必须增强政治意识、大局意识、核心意识、看齐意识，"
                      "坚决维护党中央权威和集中统一领导”")
        doc = Document.from_text(f"中央文件指出，{long_quote}，这是根本政治要求。")
        res = self.engine.run(doc)
        self.assertFalse([i for i in res.issues if "核心意识" in (i.original or "")])

    def test_suppression_statistics_present(self):
        self.assertIn("L1_exact_hits", self.result.stats)
        self.assertIn("embedding_mode", self.result.stats)


class TestNormEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ProofreadAgent()
        cls.engine = NormEngine(cls.agent.settings, cls.agent.store)
        cls.doc = Document.from_file(DEMO)
        cls.result = cls.engine.run(cls.doc)
        cls.ids = {i.rule_id for i in cls.result.issues}

    def test_multiple_used_for_decrease(self):
        hits = [i for i in self.result.issues if "下降了3.2倍" in (i.original or "")]
        self.assertTrue(hits)
        self.assertEqual(hits[0].severity, Severity.MAJOR)

    def test_near_and_yu_conflict(self):
        self.assertTrue(any("近200余人" in (i.original or "") for i in self.result.issues))

    def test_wrong_character(self):
        hits = [i for i in self.result.issues if i.original == "截止目前"]
        self.assertTrue(hits)
        self.assertEqual(hits[0].suggestion, "截至目前")

    def test_halfwidth_quote_normalization(self):
        hits = [i for i in self.result.issues if i.rule_id == "FMT-QUOTE"]
        self.assertTrue(hits)
        self.assertTrue(hits[0].suggestion.startswith("“"))

    def test_no_issue_inside_reference_section(self):
        for i in self.result.issues:
            self.assertFalse(self.doc.in_reference_section(i.span.start),
                             f"{i.rule_id} 不应在参考文献区报错")

    def test_fullwidth_quote_skips_english(self):
        doc = Document.from_text('The term "digital governance" is widely used.')
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "FMT-QUOTE"], [],
                         "纯英文引号不应报全角问题")

    def test_fullwidth_quote_flags_chinese(self):
        doc = Document.from_text('落实"一件事一次办"场景。')
        res = self.engine.run(doc)
        self.assertTrue([i for i in res.issues if i.rule_id == "FMT-QUOTE"],
                        "含中文的直引号应提示改用全角弯引号")

    def test_cjk_latin_allowlist(self):
        doc = Document.from_text("通过GitHub仓库管理代码，输出PDF报告。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "PUN-CJK-LATIN"], [],
                         "技术缩写（GitHub/PDF）与中文粘连属通行写法，不应提示")


class TestLogicEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ProofreadAgent()
        cls.engine = LogicEngine(cls.agent.settings, cls.agent.store)
        cls.doc = Document.from_file(DEMO)
        cls.result = cls.engine.run(cls.doc)

    def test_overgeneralization(self):
        hits = [i for i in self.result.issues if i.rule_id == "LOGIC-OVERGEN"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].severity, Severity.MAJOR)
        self.assertIn("调研范围", hits[0].suggestion)

    def test_absolute_claims(self):
        # 内嵌的绝对化硬断言应被捕获（即便无证据也提示）
        doc = Document.from_text(
            "这一改革举措必然导致系统稳定性全面下降，必须引起高度警惕。")
        res = self.engine.run(doc)
        hits = [i for i in res.issues if i.rule_id == "LOGIC-ABSOLUTE"]
        self.assertTrue(hits, "应捕获内嵌的绝对化断言")
        terms = {t for i in hits for t in i.meta.get("terms", [])}
        self.assertIn("必然导致", terms)
        self.assertEqual(hits[0].severity, Severity.MAJOR)

    def test_absolute_opener_is_not_flagged(self):
        """句首修辞性开场白（众所周知，…/显而易见，…）不是事实断言，应跳过。"""
        for opener in ("众所周知，数字治理很重要。", "显而易见，这项工作需要推进。"):
            doc = Document.from_text(opener)
            res = self.engine.run(doc)
            self.assertEqual(
                [i for i in res.issues if i.rule_id == "LOGIC-ABSOLUTE"], [],
                f"开场白不应被判绝对化：{opener}")

    def test_vague_reference(self):
        # 无先行语的多字短语应被标记
        doc = Document.from_text("会议讨论了预算安排。该做法显著提升了整体效率。")
        res = self.engine.run(doc)
        hits = [i for i in res.issues if i.rule_id == "LOGIC-VAGUE-REF"]
        self.assertTrue(any(i.meta.get("word") == "该做法" for i in hits))

    def test_vague_reference_has_antecedent_is_skipped(self):
        """前文出现「二次录入现象」这类先行语时，该做法并非指代不明。"""
        doc = Document.from_text("上线后出现了二次录入现象。该做法亟待优化解决。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "LOGIC-VAGUE-REF"], [])

    def test_single_char_demonstrative_not_flagged(self):
        """单字代词（其/此/该/这些/那些）句首几乎必有上下文，不应自动告警。"""
        doc = Document.from_text("平台已上线运行。其稳定性在测试中表现良好。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "LOGIC-VAGUE-REF"], [])

    def test_causal_leap_with_in_sentence_cause_is_skipped(self):
        """本句内已陈述因（逗号前有条件），不构成因果跳步。"""
        doc = Document.from_text("由于垂管系统未实现对接，导致业务办理中出现二次录入。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "LOGIC-CAUSAL-LEAP"], [])

    def test_causal_leap_at_sentence_start_flagged(self):
        """连接词在句首、前文无证据支撑，才是真正的因果跳步。"""
        doc = Document.from_text("群众普遍反映办事不便。由此可见，该政策已经彻底失败。")
        res = self.engine.run(doc)
        self.assertTrue([i for i in res.issues if i.rule_id == "LOGIC-CAUSAL-LEAP"])

    def test_no_overgeneralization_when_scope_is_consistent(self):
        doc = Document.from_text(
            "本次调研覆盖全省 12 个县区。因此，全省范围内的平台建设已进入精细化管理阶段。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "LOGIC-OVERGEN"], [])


class TestAcademicEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ProofreadAgent()
        cls.engine = AcademicEngine(cls.agent.settings, cls.agent.store, cls.agent.embedding)
        cls.doc = Document.from_file(DEMO)
        cls.result = cls.engine.run(cls.doc)

    def test_sum_consistency(self):
        hits = [i for i in self.result.issues if i.rule_id == "DATA-SUM"]
        self.assertTrue(hits)
        self.assertEqual(hits[0].meta["computed"], 52)

    def test_percent_range(self):
        self.assertTrue(any(i.rule_id == "DATA-PCT-RANGE" for i in self.result.issues))

    def test_percent_consistency(self):
        hits = [i for i in self.result.issues if i.rule_id == "DATA-PCT-CONSISTENCY"]
        self.assertTrue(hits, "应算出 776/862 = 90.0% 与文中 95.8% 不符")
        self.assertAlmostEqual(hits[0].meta["computed"], 90.02, places=1)

    def test_no_false_positive_when_sum_is_correct(self):
        doc = Document.from_text(
            "平台年度运维费用约需420万元，其中硬件采购约240万元，软件授权约150万元，"
            "人员培训约50万元，三项合计440万元。")
        res = self.engine.run(doc)
        self.assertEqual([i for i in res.issues if i.rule_id == "DATA-SUM"], [])

    def test_reference_checks(self):
        ids = {i.rule_id for i in self.result.issues}
        self.assertIn("REF-001", ids)
        self.assertIn("REF-002", ids)
        self.assertIn("REF-004", ids)

    def test_yoy_without_number(self):
        self.assertTrue(any(i.rule_id == "DATA-YOY-VAGUE" for i in self.result.issues))

    def test_cross_document_duplicate(self):
        hits = [i for i in self.result.issues if i.rule_id == "DUP-CROSS"]
        self.assertTrue(hits, "应与历史语料库比对出疑似重复发表")
        self.assertIn("2025_03", hits[0].meta["corpus"])

    def test_claims_extraction(self):
        claims = self.engine.extract_claims(self.doc)
        self.assertIsInstance(claims, list)


class TestStyleEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ProofreadAgent()
        cls.engine = StyleEngine(cls.agent.settings, cls.agent.store, "公文专报")
        cls.doc = Document.from_file(DEMO)
        cls.result = cls.engine.run(cls.doc)

    def test_profile_metrics(self):
        prof = self.result.stats["profile"]
        for key in ("平均句长", "口语词密度", "绝对化词密度", "数据标注率", "段旨句命中率"):
            self.assertIn(key, prof)

    def test_metric_gap_detected(self):
        hits = [i for i in self.result.issues if i.rule_id == "STYLE-METRIC"]
        self.assertTrue(hits)
        self.assertTrue(any(i.meta["metric"] == "数据标注率" for i in hits))

    def test_transfer_plan_generated(self):
        plan = self.engine.transfer_plan()
        self.assertTrue(plan)
        self.assertTrue(all("动作" in p and "说明" in p for p in plan))

    def test_academic_preset_differs(self):
        eng = StyleEngine(self.agent.settings, self.agent.store, "学术期刊")
        res = eng.run(self.doc)
        self.assertEqual(res.stats["preset"], "学术期刊")


class TestPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = ProofreadAgent()
        cls.report = cls.agent.audit_file(DEMO)
        cls.data = cls.report.to_dict()

    def test_report_shape(self):
        self.assertEqual(self.report.release_gate, "BLOCK")   # 存在致命问题
        self.assertGreaterEqual(self.report.counts["FATAL"], 1)
        self.assertGreater(len(self.report.issues), 20)
        self.assertEqual(len(self.report.dimension_scores), 5)

    def test_issue_contract_positions(self):
        for i in self.report.issues:
            self.assertGreater(i.line, 0)
            if i.span.length and i.span.end <= len(self.data.get("source_text", "")) + 10 ** 9:
                self.assertGreaterEqual(i.span.end, i.span.start)

    def test_trace_covers_all_agents(self):
        nodes = {t["node"] for t in self.data["trace"]}
        for expected in ("intake_agent", "supervisor_agent", "terminology_agent",
                         "language_agent", "logic_agent", "academic_agent",
                         "style_agent", "join_agent", "critic_agent",
                         "fact_check_agent", "revision_agent", "report_agent"):
            self.assertIn(expected, nodes)

    def test_critic_loop_ran_twice(self):
        rounds = [t for t in self.data["trace"] if t["node"] == "critic_agent"]
        self.assertGreaterEqual(len(rounds), 2)

    def test_fact_check_records(self):
        records = self.data["fact_check"]
        self.assertTrue(records)
        for r in records:
            self.assertIn("formula", r)
            self.assertIn("verdict", r)

    def test_revision_applies_safe_fixes_only(self):
        rev = self.data["llm_notes"]["revision"]
        self.assertGreater(rev["changed"], 0)
        # 修订稿中不应残留可自动修复的错别字
        self.assertNotIn("布署", rev["revised_preview"])
        # 高风险问题必须保留待人工处理
        self.assertTrue(any("高风险" in s.get("skip_reason", "") for s in rev["skipped"]))

    def test_degraded_flag_when_no_local_model(self):
        self.assertTrue(self.report.degraded)
        self.assertEqual(self.report.llm_backend, "mock")

    def test_report_rendering(self):
        md = self.agent.render(self.report, "", "markdown")
        html = self.agent.render(self.report, "", "html")
        self.assertIn("智能审校报告", md)
        self.assertIn("<html", html)
        self.assertIn("签发门禁", html)


class TestTerminologyUpdater(unittest.TestCase):
    def test_extract_candidates_from_policy_text(self):
        from proofread_agent.terminology import TerminologyUpdater
        agent = ProofreadAgent()
        updater = TerminologyUpdater(agent.store, agent.settings)
        text = ("要坚持以人民为中心的发展思想，坚持稳中求进工作总基调，"
                "牢固树立绿水青山就是金山银山的理念，深入推进全面从严治党。")
        cands = updater.extract_candidates(text, "unit-test")
        self.assertTrue(cands)
        self.assertTrue(all(c.confidence > 0 for c in cands))

    def test_shadow_diff_and_promote(self):
        from proofread_agent.terminology import TerminologyVersion
        agent = ProofreadAgent()
        before = agent.store.version
        beta = TerminologyVersion(
            version=before + "+test", loaded_at="now",
            entries=list(agent.store.entries), banned=list(agent.store.banned))
        diff = agent.store.stage_shadow(beta)
        self.assertEqual(diff["new_version"], before + "+test")
        self.assertEqual(agent.store.version, before, "影子版本不应直接影响线上版本")
        promoted = agent.store.promote_shadow(approved=True)
        self.assertIsNotNone(promoted)
        self.assertEqual(agent.store.version, before + "+test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
