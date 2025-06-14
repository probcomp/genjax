from beartype import BeartypeConf
from beartype.claw import beartype_this_package

conf = BeartypeConf(
    is_color=True,
    is_debug=False,
    is_pep484_tower=True,
    violation_type=TypeError,
)

beartype_this_package(conf=conf)

from .adev import (  # noqa: E402
    Dual,
    categorical_enum_parallel,
    expectation,
    flip_enum,
    flip_enum_parallel,
    flip_mvd,
    flip_reinforce,
    geometric_reinforce,
    normal_reinforce,
    normal_reparam,
    multivariate_normal_reparam,
    multivariate_normal_reinforce,
)
from .core import (  # noqa: E402
    GFI,
    Distribution,
    Fn,
    Pytree,
    Vmap,
    gen,
    get_choices,
    modular_vmap,
    seed,
    tfp_distribution,
    trace,
)
from .distributions import (  # noqa: E402
    bernoulli,
    beta,
    categorical,
    flip,
    normal,
    uniform,
    exponential,
    poisson,
    multivariate_normal,
)

__all__ = [
    "GFI",
    "Distribution",
    "Dual",
    "Fn",
    "Pytree",
    "Trace",
    "Vmap",
    "bernoulli",
    "beta",
    "uniform",
    "exponential",
    "poisson",
    "categorical",
    "categorical_enum_parallel",
    "expectation",
    "flip",
    "flip_enum",
    "flip_enum_parallel",
    "flip_mvd",
    "flip_reinforce",
    "gen",
    "geometric_reinforce",
    "get_choices",
    "modular_vmap",
    "normal",
    "normal_reinforce",
    "normal_reparam",
    "multivariate_normal",
    "multivariate_normal_reparam",
    "multivariate_normal_reinforce",
    "seed",
    "tfp_distribution",
    "trace",
]
