"""Run with:  python -m unittest discover -s tests -v   (from the project root)"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analyst import analyzer, report
from analyst.sample import make_sample_data
from analyst.sandbox import SandboxError, run_code


def _prepared():
    df, notes = analyzer.prepare_dataframe(make_sample_data())
    return df, analyzer.profile_dataset(df, notes)


class LoadingTests(unittest.TestCase):
    def test_csv_with_semicolons_and_latin1(self):
        raw = "name;city;amount\nJosé;Zürich;1,5\nAna;Köln;2,5\n".encode("latin-1")
        sheets = analyzer.read_table(raw, "x.csv")
        df = sheets["Data"]
        self.assertEqual(list(df.columns), ["name", "city", "amount"])
        self.assertEqual(len(df), 2)

    def test_excel_multi_sheet(self):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(xw, sheet_name="One", index=False)
            pd.DataFrame({"c": [5, 6]}).to_excel(xw, sheet_name="Two", index=False)
        sheets = analyzer.read_table(buf.getvalue(), "book.xlsx")
        self.assertEqual(set(sheets), {"One", "Two"})

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            analyzer.read_table(b"", "a.csv")
        with self.assertRaises(ValueError):
            analyzer.read_table(b"abc", "a.pdf")

    def test_text_numbers_and_dates_detected(self):
        raw = pd.DataFrame({
            "price": ["$1,200", "$950", "$1,100", "$40", None] * 6,
            "when": ["2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20", "2024-05-25"] * 6,
            "zip": ["02134", "10001", "94105", "02135", "60601"] * 6,
            "label": ["a", "b", "c", "d", "e"] * 6,
        })
        df, notes = analyzer.prepare_dataframe(raw)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["price"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["when"]))
        self.assertFalse(pd.api.types.is_numeric_dtype(df["zip"]))  # leading zeros preserved
        self.assertTrue(notes)

    def test_duplicate_and_blank_column_names(self):
        raw = pd.DataFrame([[1, 2, 3]], columns=["a", "a", " "])
        df, _ = analyzer.prepare_dataframe(raw)
        self.assertEqual(len(set(df.columns)), 3)


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.p = _prepared()

    def test_shape_and_duplicates(self):
        self.assertEqual(self.p["rows"], 1218)
        self.assertEqual(self.p["duplicates"]["count"], 18)

    def test_missing_detected(self):
        names = {d["name"]: d["missing"] for d in self.p["missing"]["columns_with_missing"]}
        self.assertGreater(names["satisfaction"], 50)
        self.assertGreater(names["customer_age"], 20)
        self.assertIn("region", names)

    def test_kinds(self):
        k = self.p["kinds"]
        self.assertEqual(k["order_id"], "identifier")
        self.assertEqual(k["order_date"], "datetime")
        self.assertEqual(k["region"], "categorical")
        self.assertEqual(k["revenue"], "numeric")

    def test_outliers_and_trend(self):
        self.assertIn("revenue", self.p["outliers"])
        t = {x["column"]: x for x in self.p["trends"]}
        self.assertEqual(t["revenue"]["direction"], "increasing")  # sample data grows over time
        self.assertEqual(t["revenue"]["agg"], "sum")

    def test_quality_score_in_range_and_json_safe(self):
        import json
        self.assertTrue(0 <= self.p["quality_score"] < 100)
        json.dumps(self.p)

    def test_profile_text(self):
        txt = analyzer.profile_to_text(self.p, self.df)
        self.assertIn("revenue", txt)
        self.assertIn("DUPLICATES: 18", txt)
        self.assertIn("SAMPLE ROWS", txt)
        self.assertNotIn("SAMPLE ROWS", analyzer.profile_to_text(self.p, self.df, include_sample=False))

    def test_edge_cases_dont_crash(self):
        analyzer.profile_dataset(pd.DataFrame({"a": [1]}))
        analyzer.profile_dataset(pd.DataFrame({"a": [None, None], "b": ["x", "y"]}))
        analyzer.profile_dataset(pd.DataFrame({"a": pd.Series([], dtype=float)}))


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.df, _ = _prepared()

    def test_basic_aggregation(self):
        out = run_code("result = df.groupby('region')['revenue'].sum().sort_values(ascending=False)", self.df)
        self.assertIn("East", out["output"])

    def test_trailing_expression_and_print(self):
        self.assertIn("1218", run_code("len(df)", self.df)["output"])
        self.assertIn("hello", run_code("print('hello')", self.df)["output"])

    def test_original_dataframe_not_mutated(self):
        run_code("df['new'] = 1\nresult = df.shape", self.df)
        self.assertNotIn("new", self.df.columns)

    def test_pd_np_allowlisted_helpers_work(self):
        out = run_code("result = np.round(np.mean(df['revenue']), 0)", self.df)
        self.assertTrue(out["output"].strip())
        run_code("result = pd.to_datetime(df['order_date']).dt.year.value_counts()", self.df)

    def test_blocked_code(self):
        bad = [
            "import os",
            "from os import path",
            "open('/etc/passwd').read()",
            "pd.read_csv('/etc/passwd')",
            "df.to_csv('/tmp/x.csv')",
            "df.to_pickle('/tmp/x.pkl')",
            "df.__class__",
            "().__class__.__bases__",
            "while True: pass",
            "df.query('revenue > 5')",
            "'{0.__class__}'.format(df)",
            "np.load('x.npy')",
            "pd.io",
            "eval('1+1')",
            "getattr(df, 'shape')",
            "exec('1')",
            "__import__('os')",
            "with open('x') as f: pass",
            "class A: pass",
            "df._mgr",
        ]
        for code in bad:
            with self.subTest(code=code):
                with self.assertRaises(SandboxError):
                    run_code(code, self.df)

    def test_runtime_errors_become_sandbox_errors(self):
        with self.assertRaises(SandboxError) as cm:
            run_code("result = df['no_such_column'].sum()", self.df)
        self.assertIn("KeyError", str(cm.exception))

    def test_timeout(self):
        with self.assertRaises(SandboxError) as cm:
            run_code("for i in range(10**10):\n    pass", self.df, timeout=1)
        self.assertIn("too long", str(cm.exception))

    def test_output_truncated(self):
        out = run_code("result = df.to_string()", self.df)
        self.assertLessEqual(len(out["output"]), 6100)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.df, self.p = _prepared()

    def test_html_report(self):
        html = report.build_html_report(
            "demo.csv", self.df, self.p, figures=None,
            insights_md="## Summary\n- **Revenue** is up\n<script>alert(1)</script>",
            qa_log=[{"role": "user", "content": "Top region?"}, {"role": "assistant", "content": "East."}],
        )
        self.assertIn("Data Analysis Report", html)
        self.assertIn("Top region?", html)
        self.assertNotIn("<script>alert(1)</script>", html)  # model/user text is escaped
        self.assertIn("Quality score", html)

    def test_excel_report_roundtrip(self):
        data = report.build_excel_report(self.df, self.p, insights_md="line1\nline2")
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        for name in ("Summary", "Columns", "Missing", "Duplicates", "Numeric stats", "Trends", "Insights", "Data"):
            self.assertIn(name, sheets)
        self.assertEqual(len(sheets["Data"]), len(self.df))


if __name__ == "__main__":
    unittest.main()
