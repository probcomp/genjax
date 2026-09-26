"""Conditional SMC retained-particle behavior."""

import time

import jax
import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax.scipy.special import logsumexp

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


@gen
def observed_normal_model(loc, scale):
    x = normal(loc, 1.0) @ "x"
    normal(x, scale) @ "y"
    return x


@gen
def initial_proposal(constraints, loc, scale):
    return normal(constraints["y"] + 0.3 * loc, 0.7) @ "x"


@gen
def extension_proposal(constraints, old_choices, loc, scale):
    return normal(0.25 * old_choices["x"] + loc + 0.1 * constraints["y"], 0.6) @ "x"


def normal_logpdf(value, loc, scale):
    return (
        -0.5 * ((value - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)
    )


@pytest.mark.parametrize("custom", [False, True])
def test_init_csmc_importance_weight(custom):
    loc, scale, y = 0.3, 0.4, 2.0
    particles = seed(init_csmc)(
        jrand.key(3),
        observed_normal_model,
        (loc, scale),
        const(4),
        {"y": jnp.array(y)},
        {"x": jnp.array(1.1), "y": jnp.array(y)},
        initial_proposal if custom else None,
    )
    xs = particles.traces.get_choices()["x"]
    expected = normal_logpdf(y, xs, scale)
    if custom:
        expected += normal_logpdf(xs, loc, 1.0) - normal_logpdf(xs, y + 0.3 * loc, 0.7)
    assert jnp.allclose(particles.log_weights, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(particles.diagnostic_weights, expected - logsumexp(expected))


@pytest.mark.parametrize("custom", [False, True])
def test_extend_csmc_importance_weight(custom):
    particles = seed(init)(
        jrand.key(4), observed_normal_model, (0.1, 0.5), const(4), {"y": jnp.array(1.0)}
    )
    locs, scale, y = jnp.linspace(0.3, 0.6, 4), 0.4, 2.0
    extended = seed(extend_csmc)(
        jrand.key(5),
        particles,
        observed_normal_model,
        (locs, jnp.full(4, scale)),
        {"y": jnp.array(y)},
        {"x": jnp.array(1.1), "y": jnp.array(y)},
        extension_proposal if custom else None,
    )
    xs = extended.traces.get_choices()["x"]
    expected = particles.log_weights + normal_logpdf(y, xs, scale)
    if custom:
        q_locs = 0.25 * particles.traces.get_choices()["x"] + locs + 0.1 * y
        expected += normal_logpdf(xs, locs, 1.0) - normal_logpdf(xs, q_locs, 0.6)
    assert jnp.allclose(extended.log_weights, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(extended.diagnostic_weights, expected - logsumexp(expected))


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("extension", [False, True])
def test_csmc_merges_observations(custom, extension):
    constraints, retained = {"y": jnp.array(2.0)}, {"x": jnp.array(1.1)}
    if extension:
        previous = seed(init)(
            jrand.key(6), observed_normal_model, (0.0, 0.4), const(4), constraints
        )
        particles = seed(extend_csmc)(
            jrand.key(7),
            previous,
            observed_normal_model,
            (jnp.zeros(4), jnp.full(4, 0.4)),
            constraints,
            retained,
            extension_proposal if custom else None,
        )
    else:
        particles = seed(init_csmc)(
            jrand.key(8),
            observed_normal_model,
            (0.0, 0.4),
            const(4),
            constraints,
            retained,
            initial_proposal if custom else None,
        )
    choices = particles.traces.get_choices()
    assert choices["x"][0] == retained["x"]
    assert jnp.all(choices["y"] == constraints["y"])
    expected_score = -normal_logpdf(1.1, 0.0, 1.0) - normal_logpdf(2.0, 1.1, 0.4)
    retained_trace = jtu.tree_map(lambda leaf: leaf[0], particles.traces)
    assert jnp.allclose(retained_trace.get_score(), expected_score)


def test_csmc_posterior_invariance():
    # Fixed before evaluating the implementation: 128 batches of 512 draws,
    # 1,024 burn-in steps, and a four-batch-means-SE tolerance for each moment.
    # The mean SE must resolve the reported 0.12 bias by more than eight SE.
    y, scale = 2.0, 0.2
    posterior_mean = y / (1.0 + scale**2)
    posterior_variance = scale**2 / (1.0 + scale**2)

    @jax.jit
    def chain(key):
        def step(x, key):
            sample_key, select_key = jrand.split(key)
            particles = seed(init_csmc)(
                sample_key,
                observed_normal_model,
                (0.0, scale),
                const(4),
                {"y": jnp.array(y)},
                {"x": x, "y": jnp.array(y)},
            )
            index = jrand.categorical(select_key, particles.log_weights)
            x = particles.traces.get_choices()["x"][index]
            return x, x

        _, draws = jax.lax.scan(
            step, jnp.array(posterior_mean), jrand.split(key, 1024 + 128 * 512)
        )
        return draws[1024:]

    start = time.perf_counter()
    draws = np.asarray(chain(jrand.key(2026)))
    elapsed = time.perf_counter() - start
    batches = draws.reshape(128, 512)
    mean, variance = draws.mean(), draws.var()
    mean_se = batches.mean(axis=1).std(ddof=1) / np.sqrt(128)
    variance_se = ((batches - mean) ** 2).mean(axis=1).std(ddof=1) / np.sqrt(128)
    print(
        f"mean={mean:.6f} expected={posterior_mean:.6f} SE={mean_se:.6f}; "
        f"variance={variance:.6f} expected={posterior_variance:.6f} SE={variance_se:.6f}; "
        f"compile+run={elapsed:.3f}s"
    )
    assert 8 * mean_se < 0.12
    assert abs(mean - posterior_mean) < 4 * mean_se
    assert abs(variance - posterior_variance) < 4 * variance_se


@gen
def nested_observed_model():
    return observed_normal_model(0.0, 0.4) @ "step"


@pytest.mark.parametrize("include_observation", [False, True])
def test_csmc_nested_constraints(include_observation):
    retained = {"step": {"x": jnp.array(1.1)}}
    if include_observation:
        retained["step"]["y"] = jnp.array(-10.0)
    particles = seed(init_csmc)(
        jrand.key(9),
        nested_observed_model,
        (),
        const(4),
        {"step": {"y": jnp.array(2.0)}},
        retained,
    )
    choices = particles.traces.get_choices()["step"]
    assert choices["x"][0] == retained["step"]["x"]
    assert jnp.all(choices["y"] == 2.0)
    assert jnp.allclose(particles.log_weights, normal_logpdf(2.0, choices["x"], 0.4))
