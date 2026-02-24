"""
Agent 专用数值计算工具
支持增长率、差值、占比等运算，用于 CALC 意图的查询
"""
from typing import Literal, Union

OpType = Literal["growth", "diff", "ratio"]


class CalculatorTool:
    """数值计算工具：增长率、差值、占比"""

    def exec(
        self,
        op_type: OpType,
        a: Union[int, float],
        b: Union[int, float],
    ) -> dict:
        """
        执行数值运算。

        Args:
            op_type: 运算类型 — "growth" 增长率, "diff" 差值, "ratio" 占比
            a: 第一个数（增长率/占比时为当期，差值为被减数）
            b: 第二个数（增长率/占比时为基期，差值为减数）

        Returns:
            包含 result (数值，异常时为 None) 和 formula (公式字符串) 的字典。
            除零时 result 为 None，formula 为友好错误提示。
        """
        a, b = float(a), float(b)

        if op_type == "growth":
            return self._growth(a, b)
        if op_type == "diff":
            return self._diff(a, b)
        if op_type == "ratio":
            return self._ratio(a, b)
        return {
            "result": None,
            "formula": f"未知运算类型: {op_type}，支持 growth / diff / ratio",
        }

    @staticmethod
    def _fmt(x: float) -> str:
        """数值在公式中的显示：整数不显示小数部分"""
        return str(int(x)) if x == int(x) else str(x)

    def _growth(self, a: float, b: float) -> dict:
        """增长率: (a - b) / b"""
        if b == 0:
            return {
                "result": None,
                "formula": "增长率计算时除数（基期）不能为零，请检查数据。",
            }
        value = (a - b) / b
        formula = f"({self._fmt(a)} - {self._fmt(b)}) / {self._fmt(b)} = {value * 100:.2f}%"
        return {"result": value, "formula": formula}

    def _diff(self, a: float, b: float) -> dict:
        """差值: a - b"""
        value = a - b
        formula = f"{self._fmt(a)} - {self._fmt(b)} = {self._fmt(value) if value == int(value) else value}"
        return {"result": value, "formula": formula}

    def _ratio(self, a: float, b: float) -> dict:
        """占比: a / b（按百分比展示）"""
        if b == 0:
            return {
                "result": None,
                "formula": "占比计算时除数（总量）不能为零，请检查数据。",
            }
        value = a / b
        formula = f"{self._fmt(a)} / {self._fmt(b)} = {value * 100:.2f}%"
        return {"result": value, "formula": formula}


if __name__ == "__main__":
    tool = CalculatorTool()

    # 增长率: (100 - 80) / 80 = 25%
    out = tool.exec("growth", 100, 80)
    print("growth(100, 80):", out)
    assert out["result"] == 0.25
    assert "25.00%" in out["formula"]

    # 差值: 100 - 80 = 20
    out = tool.exec("diff", 100, 80)
    print("diff(100, 80):", out)
    assert out["result"] == 20
    assert "100 - 80 = 20" in out["formula"] or "= 20" in out["formula"]

    # 占比: 30 / 100 = 30%
    out = tool.exec("ratio", 30, 100)
    print("ratio(30, 100):", out)
    assert out["result"] == 0.3
    assert "30.00%" in out["formula"]

    # 除零：增长率
    out = tool.exec("growth", 100, 0)
    print("growth(100, 0):", out)
    assert out["result"] is None
    assert "不能为零" in out["formula"]

    # 除零：占比
    out = tool.exec("ratio", 10, 0)
    print("ratio(10, 0):", out)
    assert out["result"] is None
    assert "不能为零" in out["formula"]

    print("All tests passed.")
