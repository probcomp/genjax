"""Conditional SMC retained-particle behavior."""

import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

from genjax.core import const, gen
from genjax.distributions import normal
from genjax.inference.smc import extend_csmc, init, init_csmc
from genjax.pjax import seed


@gen
def normal_model():
    return normal(0.0, 1.0) @ "x"


@gen
def shifted_normal_model(loc):
    return normal(loc, 1.0) @ "x"


def test_init_csmc_retains_choices_and_trace():
    retained_choices = {"x": jnp.array(5.0)}
    particles = seed(init_csmc)(
        jrand.key(0), normal_model, (), const(4), {}, retained_choices
    )

    retained_log_density, _ = normal_model.assess(retained_choices)
    assert jnp.array_equal(
        particles.traces.get_choices()["x"][0], retained_choices["x"]
    )
    retained_trace = jtu.tree_map(lambda leaf: leaf[0], particles.traces)
    assert jnp.allclose(retained_trace.get_score(), -retained_log_density)
    proposal_trace = seed(normal_model.simulate)(jrand.key(3))
    assert jtu.tree_structure(retained_trace) == jtu.tree_structure(proposal_trace)


def test_extend_csmc_retains_choices_and_trace():
    particles = seed(init)(
        jrand.key(1), shifted_normal_model, (jnp.array(0.0),), const(4), {}
    )
    retained_choices = {"x": jnp.array(5.0)}
    extended = seed(extend_csmc)(
        jrand.key(2),
        particles,
        shifted_normal_model,
        jnp.zeros(4),
        {},
        retained_choices,
    )

    retained_log_density, _ = shifted_normal_model.assess(retained_choices, 0.0)
    assert jnp.array_equal(extended.traces.get_choices()["x"][0], retained_choices["x"])
    retained_trace = jtu.tree_map(lambda leaf: leaf[0], extended.traces)
    assert jnp.allclose(retained_trace.get_score(), -retained_log_density)
    old_trace = jtu.tree_map(lambda leaf: leaf[0], particles.traces)
    assert jtu.tree_structure(retained_trace) == jtu.tree_structure(old_trace)
