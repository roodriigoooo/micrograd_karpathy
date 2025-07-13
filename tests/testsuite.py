import pytest
import numpy as np
from engine import Value
from nn import MLP, SGD
from utils import numerical_gradient


# --- Test Core Operations ---

def test_add():
    a = Value([1, 2, 3])
    b = Value([4, 5, 6])
    c = a + b
    c.backward()
    assert np.allclose(c.data, [5, 7, 9])
    assert np.allclose(a.grad, [1, 1, 1])
    assert np.allclose(b.grad, [1, 1, 1])


def test_mul():
    a = Value([1, 2, 3])
    b = Value([4, 5, 6])
    c = a * b
    c.backward()
    assert np.allclose(c.data, [4, 10, 18])
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)


def test_pow():
    a = Value([2, 3, 4])
    c = a ** 3
    c.backward()
    assert np.allclose(c.data, [8, 27, 64])
    assert np.allclose(a.grad, 3 * a.data ** 2)


def test_relu():
    a = Value([-1, 0, 1])
    b = a.relu()
    b.backward()
    assert np.allclose(b.data, [0, 0, 1])
    assert np.allclose(a.grad, [0, 0, 1])


# --- Test Broadcasting ---

def test_add_broadcast():
    a_data = np.random.randn(3, 4)
    b_data = np.random.randn(4)

    # Test grad wrt a
    a1 = Value(a_data.copy()); b1 = Value(b_data.copy())
    (a1 + b1).sum().backward()
    num_grad_a = numerical_gradient(lambda x: (x + b1).sum(), a1)
    assert np.allclose(a1.grad, num_grad_a, atol=1e-3)

    # Test grad wrt b
    a2 = Value(a_data.copy()); b2 = Value(b_data.copy())
    (a2 + b2).sum().backward()
    num_grad_b = numerical_gradient(lambda x: (a2 + x).sum(), b2)
    assert np.allclose(b2.grad, num_grad_b, atol=1e-3)

def test_add_broadcast_leading_dims():
    a = Value(3.0).reshape((1,1,1))
    b = Value(np.random.randn(5, 4, 3).astype(np.float32))
    (a + b).sum().backward()

    #we expect
    assert a.grad.shape == (1,1,1)
    assert b.grad.shape == (5,4,3)
    assert np.allclose(a.grad, np.array([[[60.0]]], dtype = np.float32))
    assert np.allclose(b.grad, np.ones_like(b.data))

    # a is a scalar broadcast so check on b only
    num_grad_b = numerical_gradient(lambda x: (a + x).sum(), b)
    assert np.allclose(b.grad, num_grad_b, atol=1e-3)

def test_mul_broadcast():
    a_data = np.random.randn(3, 4)
    b_data = np.random.randn(3, 1)

    # Test grad wrt a
    a1 = Value(a_data.copy()); b1 = Value(b_data.copy())
    (a1 * b1).sum().backward()
    num_grad_a = numerical_gradient(lambda x: (x * b1).sum(), a1)
    assert np.allclose(a1.grad, num_grad_a, atol=1e-3)

    # Test grad wrt b
    a2 = Value(a_data.copy()); b2 = Value(b_data.copy())
    (a2 * b2).sum().backward()
    num_grad_b = numerical_gradient(lambda x: (a2 * x).sum(), b2)
    assert np.allclose(b2.grad, num_grad_b, atol=1e-3)

def test_mul_broadcast_multiple_singletons():
    a = Value(np.random.randn(7, 1, 5).astype(np.float32))
    b = Value(np.random.randn(1, 4, 1).astype(np.float32))
    (a * b).sum().backward()

    # they should broadcast to (7, 4, 5)
    assert a.grad.shape == a.data.shape
    assert b.grad.shape == b.data.shape

    num_grad_a = numerical_gradient(lambda x: (x * b).sum(), a)
    num_grad_b = numerical_gradient(lambda x: (a * x).sum(), b)
    assert np.allclose(a.grad, num_grad_a, atol=1e-3)
    assert np.allclose(b.grad, num_grad_b, atol=1e-3)

# --- Test Tensor Operations ---

def test_matmul():
    a_data = np.random.randn(2, 3)
    b_data = np.random.randn(3, 4)

    # Test grad wrt a
    a1 = Value(a_data.copy());
    b1 = Value(b_data.copy())
    (a1 @ b1).sum().backward()
    num_grad_a = numerical_gradient(lambda x: (x @ b1).sum(), a1)
    assert np.allclose(a1.grad, num_grad_a, atol=1e-3)

    # Test grad wrt b
    a2 = Value(a_data.copy());
    b2 = Value(b_data.copy())
    (a2 @ b2).sum().backward()
    num_grad_b = numerical_gradient(lambda x: (a2 @ x).sum(), b2)
    assert np.allclose(b2.grad, num_grad_b, atol=1e-3)

def test_matmul_vector_matrix():
    a = Value(np.random.randn(4).astype(np.float32))
    b = Value(np.random.randn(4, 1).astype(np.float32))
    (a @ b).sum().backward()

    num_grad_a = numerical_gradient(lambda x: (x @ b).sum(), a)
    num_grad_b = numerical_gradient(lambda x: (a @ x).sum(), b)
    assert np.allclose(a.grad, num_grad_a, atol=1e-3)
    assert np.allclose(b.grad, num_grad_b, atol=1e-3)


def test_slice():
    a = Value(np.array([[1, 2, 3], [4, 5, 6]]))
    b = a[0, :2]  # Slice -> [1, 2]
    b.sum().backward()

    expected_grad = np.array([[1, 1, 0], [0, 0, 0]])
    assert np.allclose(b.data, [1, 2])
    assert np.allclose(a.grad, expected_grad)

def test_slice_backward():
    data = np.random.randn(6).astype(np.float32)
    x = Value(data.copy())
    x[::2].sum().backward()

    expected = np.zeros_like(data)
    expected[::2] = 1.0
    assert np.allclose(x.grad, expected)


def test_gradient_accumulation():
    x = Value(np.random.randn(5).astype(np.float32))

    loss1 = x.sum()          # partial / partial x = 1
    loss1.backward(retain_graph=True)
    grad_after_first = x.grad.copy()

    loss2 = (x * 2).sum()    # partial/partial x = 2
    loss2.backward()

    expected = grad_after_first + 2 * np.ones_like(x.data)
    assert np.allclose(x.grad, expected)

def test_softplus_numerical_stability():
    x = Value(np.full((10,), 100.0, dtype=np.float32))
    x.softplus().sum().backward()

    assert not np.isnan(x.grad).any(), "Gradient contains NaNs"
    assert not np.isnan(x.data).any(), "Forward pass produced NaNs"

# --- Test Full Network and Optimizer ---

def test_mlp_forward():
    model = MLP(3, [4, 1])
    xs = Value(np.random.randn(10, 3))  # Batch of 10
    ypred = model(xs)
    assert ypred.shape == (10, 1)


def test_training_step():
    # Setup
    model = MLP(2, [3, 1])
    optimizer = SGD(model.parameters(), lr=0.1)
    x = Value(np.array([[0.5, -0.5]]))
    y_true = Value(np.array([[0.9]]))

    # Get initial parameter values
    initial_params = [p.data.copy() for p in model.parameters()]

    # Training Step
    optimizer.zero_grad()
    y_pred = model(x)
    loss = (y_pred - y_true) ** 2
    loss = loss.sum()  # Ensure loss is a scalar for backward
    loss.backward()
    optimizer.step()

    # Check if parameters have been updated
    for i, p in enumerate(model.parameters()):
        assert not np.allclose(initial_params[i], p.data), "Parameter was not updated"

    # Check if gradients were computed
    for p in model.parameters():
        assert not np.allclose(p.grad, 0), "Gradient was not computed"