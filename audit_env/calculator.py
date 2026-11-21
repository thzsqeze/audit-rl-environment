"""Safe decimal calculator: numbers, + - * / and parentheses only. No names, calls or code."""

from __future__ import annotations

import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

MAX_EXPRESSION_LEN = 300
MAX_DEPTH = 32
_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?|\.\d+)|(.))")


class CalculatorError(ValueError):
    pass


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    expr = expr.rstrip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            raise CalculatorError("invalid token")
        number, op = m.groups()
        if number is not None:
            tokens.append(number)
        elif op in "+-*/()":
            tokens.append(op)
        else:
            raise CalculatorError("only numbers, + - * / and parentheses are allowed")
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0
        self.depth = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise CalculatorError("unexpected end of expression")
        self.i += 1
        return tok

    def expr(self) -> Decimal:
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self) -> Decimal:
        value = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            rhs = self.factor()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise CalculatorError("division by zero")
                value = value / rhs
        return value

    def factor(self) -> Decimal:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalculatorError("expression too deeply nested")
        try:
            tok = self.take()
            if tok == "-":
                return -self.factor()
            if tok == "+":
                return self.factor()
            if tok == "(":
                value = self.expr()
                if self.take() != ")":
                    raise CalculatorError("missing closing parenthesis")
                return value
            if tok in "*/)":
                raise CalculatorError("unexpected operator")
            return Decimal(tok)
        finally:
            self.depth -= 1


def evaluate(expression: str) -> str:
    """Evaluate an arithmetic expression and return a plain decimal string."""
    if not isinstance(expression, str):
        raise CalculatorError("expression must be a string")
    if len(expression) > MAX_EXPRESSION_LEN:
        raise CalculatorError("expression too long")
    tokens = _tokenize(expression)
    if not tokens:
        raise CalculatorError("empty expression")
    with localcontext() as ctx:
        ctx.prec = 40
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            parser = _Parser(tokens)
            value = parser.expr()
            if parser.peek() is not None:
                raise CalculatorError("unexpected trailing input")
            value = +value  # apply context precision
        except (InvalidOperation, DivisionByZero, OverflowError) as exc:
            raise CalculatorError("arithmetic error") from exc
    if abs(value) >= Decimal(10) ** 30:
        raise CalculatorError("result out of range")
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "") else text
