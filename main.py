import gradio as gr
import pandas as pd
import sympy as sp
from lark import Lark, Transformer
from pprint import pprint
from qbee import *


def get_eq_grammar():
    with open("equations.lark", 'r') as f:
        return f.read()


diff_eq_parser = Lark(get_eq_grammar())


def eval_quadratization(system_str: str) -> str:
    tree = diff_eq_parser.parse(system_str)
    system = DiffEqTransformer().transform(tree)
    res = polynomialize_and_quadratize(system)
    return str(res)


def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Quadratize** to see the output.")
        with gr.Column():
            inp = gr.Textbox(placeholder="Enter a system of equations")
            out = gr.Textbox()
        btn = gr.Button("Quadratize")
        btn.click(fn=eval_quadratization, inputs=inp, outputs=out)

    demo.launch(share=True)


def is_symbol(expr: sp.Expr):
    return isinstance(expr, sp.Symbol)


def decide_vars_and_params(equations: list[sp.Symbol, sp.Expr]) -> list[sp.Function, sp.Expr]:
    lhs, rhs = list(zip(*equations))

    def decide(e: sp.Symbol):
        return functions(e.name) if e in lhs else parameters(e.name)

    lhs_res = [functions(expr.name) for expr in lhs]
    rhs_res = [expr.replace(is_symbol, decide)
               for expr in rhs]
    return list(zip(lhs_res, rhs_res))


class DiffEqTransformer(Transformer):
    def start(self, equations):
        return decide_vars_and_params(equations)

    def equation(self, eq):
        lhs, rhs = eq
        return (lhs, rhs)

    def number(self, n):
        return sp.Number(n[0])

    def varname(self, name):
        (name,) = name
        match name:
            case "e":
                return sp.E
            case "pi":
                return sp.pi
            case _:
                return sp.Symbol(name)

    def sum(self, terms):
        return sp.Add(terms[0], terms[1])

    def diff(self, terms):
        return sp.Add(terms[0], -terms[1])

    def mul(self, terms):
        return sp.Mul(terms[0], terms[1])

    def div(self, terms):
        return sp.Mul(terms[0], sp.Number(1) / terms[1])

    def pow(self, terms):
        return sp.Pow(terms[0], terms[1])

    def braced(self, expr):
        return expr[0]

    def function(self, expr):
        fname, *args = expr
        return functions(fname.name)

    def ln(self, expr):
        return sp.ln(expr[0])

    def log(self, expr):
        return sp.log(expr[0])

    def sin(self, expr):
        return sp.sin(expr[0])

    def cos(self, expr):
        return sp.cos(expr[0])

    def tan(self, expr):
        return sp.tan(expr[0])

    def cot(self, expr):
        return sp.cot(expr[0])

    def asin(self, expr):
        return sp.asin(expr[0])

    def acos(self, expr):
        return sp.acos(expr[0])

    def atan(self, expr):
        return sp.atan(expr[0])

    def acot(self, expr):
        return sp.acot(expr[0])

    def sinh(self, expr):
        return sp.sinh(expr[0])

    def cosh(self, expr):
        return sp.cosh(expr[0])

    def tanh(self, expr):
        return sp.tanh(expr[0])

    def coth(self, expr):
        return sp.coth(expr[0])

    def asinh(self, expr):
        return sp.asinh(expr[0])

    def acosh(self, expr):
        return sp.acosh(expr[0])

    def atanh(self, expr):
        return sp.atanh(expr[0])

    def acoth(self, expr):
        return sp.acoth(expr[0])

    def exp(self, expr):
        return sp.exp(expr[0])

    def sqrt(self, expr):
        return sp.sqrt(expr[0])


if __name__ == '__main__':
    launch_gradio()
