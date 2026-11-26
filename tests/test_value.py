"""Unit tests for the scalar Value class.

Run with: python -m pytest tests/test_value.py -v
"""

import math

import pytest

from minitorch.core.value import Value


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---- Forward pass ----

def test_add_forward():
    assert approx((Value(2.0) + Value(3.0)).data, 5.0)


def test_mul_forward():
    assert approx((Value(2.0) * Value(3.0)).data, 6.0)


def test_scalar_ops_with_python_numbers():
    assert approx((Value(2.0) + 1).data, 3.0)
    assert approx((1 + Value(2.0)).data, 3.0)
    assert approx((5 - Value(2.0)).data, 3.0)
    assert approx((Value(6.0) / 2).data, 3.0)
    assert approx((6 / Value(2.0)).data, 3.0)


def test_pow_forward():
    assert approx((Value(3.0) ** 2).data, 9.0)


def test_tanh_and_relu_forward():
    assert approx(Value(0.5).tanh().data, math.tanh(0.5))
    assert approx(Value(-1.0).relu().data, 0.0)
    assert approx(Value(2.0).relu().data, 2.0)


# ---- Backward pass ----

def test_add_grad():
    a, b = Value(2.0), Value(3.0)
    out = a + b
    out.backward()
    assert approx(a.grad, 1.0)
    assert approx(b.grad, 1.0)


def test_mul_grad():
    a, b = Value(2.0), Value(3.0)
    out = a * b
    out.backward()
    assert approx(a.grad, 3.0)
    assert approx(b.grad, 2.0)


def test_grad_accumulates_when_variable_used_twice():
    # f = x * x  ->  df/dx = 2x
    x = Value(4.0)
    out = x * x
    out.backward()
    assert approx(x.grad, 8.0)


def test_chain_rule_micrograd_example():
    # Same example as micrograd's README.
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    assert approx(g.data, 24.7041, 1e-4)
    assert approx(a.grad, 138.8338, 1e-4)
    assert approx(b.grad, 645.5773, 1e-4)


def test_exp_log_tanh_grads():
    x = Value(0.3)
    out = x.exp().log().tanh()  # equals tanh(x)
    out.backward()
    expected = 1 - math.tanh(0.3) ** 2
    assert approx(x.grad, expected, 1e-6)


def test_relu_grad_zero_for_negative_input():
    x = Value(-2.0)
    x.relu().backward()
    assert approx(x.grad, 0.0)
