import pytest

from audit_env.calculator import CalculatorError, evaluate
from audit_env.money import convert_minor, format_rate, local_date, parse_offset


@pytest.mark.parametrize("expr,value", [
    ("1+2*3", "7"), ("(1+2)*3", "9"), ("10/4", "2.5"), ("-3 + 5", "2"), ("0.1+0.2", "0.3"),
    ("1083.70 * 1.0837", "1174.40569"),
])
def test_calculator_arithmetic(expr, value):
    assert evaluate(expr) == value


@pytest.mark.parametrize("expr", [
    "__import__('os')", "2**10", "open('x')", "a+1", "1/0", "", "(" * 40 + "1" + ")" * 40, "1" * 400, "1e5",
    "[1,2]", "1;2",
])
def test_calculator_rejects_code_and_bad_input(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_conversion_respects_exponents_and_rounding():
    assert convert_minor(100000, "EUR", "USD", "1.0837") == 108370
    assert convert_minor(12345, "JPY", "USD", "0.0067") == 8271  # exp 0 -> 2
    assert convert_minor(1000, "KWD", "USD", "3.25") == 325  # 1.000 KWD -> 3.25 USD
    assert convert_minor(1, "USD", "USD", "999") == 1
    assert convert_minor(5, "EUR", "USD", "0.5") == 3  # 2.5 -> half_up
    assert convert_minor(5, "EUR", "USD", "0.5", "half_even") == 2


def test_format_rate_is_plain_decimal():
    from decimal import Decimal
    assert format_rate(Decimal("0.000646153846")) == "0.000646154"
    assert "e" not in format_rate(Decimal("1E-7")).lower()


def test_local_date_uses_company_offset():
    tz = parse_offset("+05:00")
    assert str(local_date("2026-10-31T20:30:00+00:00", tz)) == "2026-11-01"
    assert str(local_date("2026-11-01T02:00:00-05:00", parse_offset("-05:00"))) == "2026-11-01"
