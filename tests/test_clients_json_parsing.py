from __future__ import annotations

import unittest

from book_cpt.services.clients import parse_json_array


class JsonArrayParsingTests(unittest.TestCase):
    def test_repairs_invalid_backslashes_inside_json_strings(self) -> None:
        response = r"""[
          {
            "instruction": "请解读这张图。",
            "answer": "启动条件包含 $55.3\%$ 门槛，\alpha 增大。\n第二行仍应保留为换行。"
          }
        ]"""

        rows = parse_json_array(response)

        self.assertEqual(rows[0]["answer"], "启动条件包含 $55.3\\%$ 门槛，\\alpha 增大。\n第二行仍应保留为换行。")


if __name__ == "__main__":
    unittest.main()


class BareObjectMembersTests(unittest.TestCase):
    """模型把 [{...}] 写成 [ "k": v, ... ]，少了内层花括号。

    实测是 JSON 解析失败里最常见的一种。json 会在第 2 行那个冒号处报
    Expecting ',' delimiter —— 数组里一个字符串元素后面只能跟逗号或右括号。
    内容本身是对的，补括号就能救回整条样本。
    """

    def test_repairs_the_shape_seen_in_production(self) -> None:
        response = """[
  "instruction": "请分析该表格中7475牌号（T7351状态）铝合金在不同热暴露条件下的力学性能变化。",
  "chart_subject": "7475牌号（T7351状态）铝合金型材在不同热暴露条件下的力学性能数据",
  "trends": "随着热暴露温度升高，抗拉强度（σb）与屈服强度（σ0.2）下降，延伸率（δ）上升",
  "conclusion": "热暴露温度是主导因素"
]"""
        rows = parse_json_array(response)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            sorted(rows[0]),
            ["chart_subject", "conclusion", "instruction", "trends"],
        )
        self.assertIn("7475", rows[0]["chart_subject"])

    def test_repairs_it_without_the_closing_bracket(self) -> None:
        rows = parse_json_array('[\n  "topic": "跨页归纳",\n  "combined_summary": "两页互补"\n')
        self.assertEqual(rows, [{"topic": "跨页归纳", "combined_summary": "两页互补"}])

    def test_repairs_it_together_with_a_trailing_comma(self) -> None:
        rows = parse_json_array('[\n  "summary": "要点",\n  "keywords": ["a", "b"],\n]')
        self.assertEqual(rows, [{"summary": "要点", "keywords": ["a", "b"]}])

    def test_a_real_object_array_is_left_alone(self) -> None:
        self.assertEqual(parse_json_array('[{"a": 1}, {"a": 2}]'), [{"a": 1}, {"a": 2}])

    def test_a_plain_string_array_is_not_mangled_into_an_object(self) -> None:
        # ["a", "b"] 开头也是 [" ，但后面不是冒号，不该被当成缺括号的对象。
        # 它本身是合法 JSON，parse_json_array 过滤掉非 dict 项后返回空表。
        from book_cpt.services.clients import _wrap_bare_object_members

        self.assertEqual(_wrap_bare_object_members('["a", "b"]'), '["a", "b"]')
        self.assertEqual(parse_json_array('["a", "b"]'), [])

    def test_nested_array_value_does_not_confuse_the_repair(self) -> None:
        from book_cpt.services.clients import _wrap_bare_object_members

        # 第一个 token 是数组而非 "键":，不该动
        self.assertEqual(_wrap_bare_object_members('[["a"], ["b"]]'), '[["a"], ["b"]]')

    def test_escaped_quote_in_the_key_is_handled(self) -> None:
        from book_cpt.services.clients import _wrap_bare_object_members

        repaired = _wrap_bare_object_members('[\n  "a\\"b": 1\n]')
        self.assertTrue(repaired.startswith("[{"))

    def test_a_bare_object_without_brackets_still_works(self) -> None:
        rows = parse_json_array('{"instruction": "问", "summary": "答"}')
        self.assertEqual(rows, [{"instruction": "问", "summary": "答"}])

    def test_error_preview_is_long_enough_to_diagnose(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_json_array("这不是 JSON，" + "填充" * 300)
        # 200 字符分不出"被截断"还是"格式写错"，预览要够长
        self.assertGreater(len(str(ctx.exception)), 300)
