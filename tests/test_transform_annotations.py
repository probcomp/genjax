"""Return annotations survive transformations that stage user functions."""

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.extend.core import ClosedJaxpr

from genjax import gen, normal
from genjax.adev import Dual, expectation
from genjax.pjax import (
    FlatSamplerCache,
    InitialStylePrimitive,
    SamplerConfig,
    initial_style_bind,
    modular_vmap,
    seed,
    stage,
)
from genjax.state import namespace, save, state


def annotated_square(x: jax.Array) -> jax.Array:
    """Square the input array."""
    return x**2


def test_stage_annotated_function():
    wrapped = stage(annotated_square)
    jaxpr, _ = wrapped(jnp.array(3.0))
    assert isinstance(jaxpr, ClosedJaxpr)
    assert wrapped.__name__ == annotated_square.__name__
    assert wrapped.__doc__ == annotated_square.__doc__
    assert wrapped.__module__ == annotated_square.__module__
    assert annotated_square.__annotations__["return"] is jax.Array
    assert wrapped.__annotations__["return"] != jax.Array


def test_seed_annotated_function():
    @gen
    def model():
        return normal(0.0, 1.0) @ "x"

    def plain():
        return model.simulate().get_retval()

    def annotated() -> jax.Array:
        return model.simulate().get_retval()

    key = jrand.key(0)
    expected = seed(plain)(key)
    assert jnp.array_equal(seed(annotated)(key), expected)
    assert jnp.array_equal(jax.jit(seed(annotated))(key), expected)


def test_modular_vmap_annotated_function():
    xs = jnp.arange(4.0)
    mapped = modular_vmap(annotated_square)
    assert "return" not in mapped.__annotations__
    assert jnp.array_equal(mapped(xs), xs**2)


def test_modular_vmap_scalar_return_annotation():
    def constant() -> float:
        return 1.0

    assert jnp.array_equal(
        modular_vmap(constant, in_axes=None, axis_size=3)(), jnp.ones(3)
    )


def test_initial_style_bind_annotated_function():
    primitive = InitialStylePrimitive("annotated_square_test")
    wrapped = initial_style_bind(primitive)(annotated_square)
    assert wrapped(jnp.array(3.0)) == 9.0


def test_flat_sampler_annotated_function():
    cache = FlatSamplerCache(SamplerConfig(annotated_square))
    flatten = cache._make_flat(annotated_square)
    assert "return" not in flatten.__annotations__
    flat, _ = flatten(jnp.array(3.0))
    assert flat(jnp.array(3.0), num_consts=0)[0] == 9.0


def test_state_annotated_function():
    def computation(x: jax.Array) -> jax.Array:
        save(input=x)
        return x**2

    collect = state(computation)
    assert "return" not in collect.__annotations__
    result, collected = collect(jnp.array(3.0))
    assert result == 9.0
    assert collected["input"] == 3.0


def test_namespace_annotated_function():
    def computation(x: jax.Array) -> jax.Array:
        save(input=x)
        return x**2

    result, collected = state(namespace(computation, "inner"))(jnp.array(3.0))
    assert result == 9.0
    assert collected["inner"]["input"] == 3.0


def test_expectation_annotated_function():
    objective = expectation(annotated_square)
    dual = objective.jvp_estimate(Dual(jnp.array(3.0), jnp.array(1.0)))
    assert dual.primal == 9.0
    assert dual.tangent == 6.0
    assert objective.estimate(jnp.array(3.0)) == 9.0
    assert objective.grad_estimate(jnp.array(3.0)) == 6.0


def test_stage_lambda_metadata():
    source = lambda x: x**2  # noqa: E731
    wrapped = stage(source)
    jaxpr, _ = wrapped(jnp.array(3.0))
    assert isinstance(jaxpr, ClosedJaxpr)
    assert wrapped.__name__ == source.__name__
    assert wrapped.__qualname__ == source.__qualname__


def test_expectation_lambda():
    objective = expectation(lambda x: x**2)
    dual = objective.jvp_estimate(Dual(jnp.array(3.0), jnp.array(1.0)))
    assert dual.primal == 9.0
    assert dual.tangent == 6.0
