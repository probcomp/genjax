# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GenJAX is a probabilistic programming language embedded in Python centered on programmable inference: automation which allows users of GenJAX to express and customize Bayesian inference algorithms.

## Critical Development Constraints

### Python Control Flow - ABSOLUTELY FORBIDDEN

**CRITICAL**: When writing GenJAX code (`@gen` functions, combinators, etc.), Python control flow will break JAX compilation:

- **ABSOLUTELY NEVER use Python `if...else` statements** - fundamentally incompatible with JAX compilation
- **ABSOLUTELY NEVER use Python `for` loops** - break JAX vectorization and compilation  
- **ABSOLUTELY NEVER use Python `while` loops** - not supported in JAX
- **ALWAYS use JAX control flow**: `jax.lax.cond`, `jax.lax.scan` for deterministic logic
- **ALWAYS use GenJAX combinators**: `Cond(branch1, branch2)`, `Scan(callee)` for generative functions

**Violating these constraints causes JAX compilation errors.**

### Documentation Policy

**NEVER create documentation files (.md, README, etc.) unless explicitly requested by the user.**

## Key concepts

### Datatypes: Generative Functions & Traces

- **Generative Function**: A probabilistic program that implements the Generative Function Interface (GFI)
- **Trace**: A recording of execution containing random choices sampled during the execution, the arguments of the execution, the return value of the execution, and the score of the execution (the score is `log 1 / P(random choices)`, the reciprocal of the density in logspace).

### Generative Function Interface

Generative functions are probabilistic programs which bundle together a set of probabilistic ingredients (measures and deterministic functions), and expose a computational interface that provides automation for doing computations with those ingredients.

In the following, we use tags:

- (**MATH**) indicates a description in terms of the abstract mathematical ingredients.

The (**MATH**) ingredients of generative functions are as follows:

- A measure kernel $P(dx; a \in A)$ over a measurable space $X$ given arguments $a \in A$ (informally called: the `P` distribution).
- A measurable function $f(x, a \in A) \rightarrow R$ (informally called: the return value function).
- An indexed family of measure kernels $Q(dX; a \in A, x' \in X')$ given arguments $A$ and sample $x' \in X' \subset X$ (informally called _the internal proposal distribution family_)

#### Computational Interface

The GFI is defined in `src/core.py` with these key methods:

**GFI Methods**:
- `simulate(args) -> Trace` - Sample and create trace
- `assess(args, choices) -> (Density, ReturnValue)` - Evaluate log density
- `generate(args, partial_choices) -> (Trace, Weight)` - Propose completion
- `update(args, trace, changes) -> (Trace, Weight, OldChoices)` - Update trace
- `regenerate(args, trace, selection) -> (Trace, Weight, OldChoices)` - Regenerate parts

**Trace Methods**:
- `get_retval()` - Return value
- `get_choices()` - Random choices
- `get_score()` - Log reciprocal density
- `get_args()` - Arguments
- `get_gen_fn()` - Source GFI

### Generative Function Languages

#### Distributions: Probabilistic Building Blocks

**IMPORTANT**: Distributions are generative functions that implement a `sample` and `logpdf` interface (the interface of probability distributions). They are the fundamental building blocks of probabilistic programs in GenJAX.

**Key Concepts**:

- **Distributions are Generative Functions**: Every distribution implements the GFI (simulate, assess, generate, update, regenerate)
- **Wrapper for TensorFlow Probability distributions**: Most distributions wrap TensorFlow Probability distributions for robustness and reliability
- **Parameter Convention**: Distribution parameters are passed as arguments to the GFI methods, not as constructor arguments

**Common Distributions**:

```python
from genjax import normal, beta, categorical, exponential, uniform, bernoulli

# Continuous distributions
normal(mu, sigma)              # Normal(mu, sigma)
beta(alpha, beta_param)        # Beta(alpha, beta)
exponential(rate)              # Exponential(rate)
uniform(low, high)             # Uniform(low, high)

# Discrete distributions
flip(p)                        # Bernoulli(p)
categorical(logits)            # Categorical(logits)
```

**Custom Distributions**:

```python
# Using TFP wrapper
student_t = tfp_distribution(
    lambda df, loc, scale: tfp.distributions.StudentT(df, loc, scale))

# Custom from scratch
custom_dist = distribution(sampler_fn, logpdf_fn, name="custom")
```

#### `Fn` Generative Functions

**IMPORTANT**: The `@gen` decorator is GenJAX's primary mechanism for creating `Fn` generative functions from JAX-compatible Python programs. `Fn` is the core generative function type that implements complex probabilistic models through function composition.

**What `@gen` Does**:

- **Transforms JAX Python Functions**: Converts ordinary JAX-compatible Python functions into `Fn` generative functions that implement the full GFI
- **Enables Probabilistic Addressing**: Adds the `@` operator for addressing the random choices made by distributions and other generative functions

##### `@gen` Decorator: Transform JAX-Compatible Python Functions into `Fn`

```python
import jax.numpy as jnp
from genjax import beta, flip, gen

# Example: beta-bernoulli model
@gen
def beta_ber():
    # define the hyperparameters that control the Beta prior
    alpha0 = jnp.array(10.0)
    beta0 = jnp.array(10.0)
    # sample f from the Beta prior
    f = beta(alpha0, beta0) @ "latent_fairness"
    return flip(f) @ "obs"

# The @gen decorator creates an Fn generative function
print(type(beta_ber))  # <class 'genjax.core.Fn'>
```

**String Addressing**: The `@` operator creates addressed random choices, enabling hierarchical addressing through function composition.

**Implementation**: `@gen` uses handler stacks to intercept `@` calls and manage trace state.

**Advanced `Fn` Patterns**:

```python
# ✅ CORRECT: Use Scan for iteration
@gen
def step_function(carry, x):
    state = normal(carry, 1.0) @ "state"
    return state, state

@gen
def time_series_model(T):
    scan_fn = Scan(step_function, length=T)
    final_carry, states = scan_fn((0.0, None)) @ "steps"
    return states

# ❌ WRONG: Python for loops break JAX compilation
@gen
def bad_model(T):
    for t in range(T):  # CRITICAL ERROR
        x = normal(0.0, 1.0) @ f"x_{t}"  # Dynamic addressing also breaks
```

**Key Principles**:
- **Static Structure**: Addressing must be statically determinable
- **JAX Compatibility**: Use JAX operations, not Python operations
- **Deterministic Execution**: Same sequence of @ calls given same arguments

#### Generative Function Combinators

**IMPORTANT**: Combinators compose generative functions while preserving GFI and JAX semantics.

**Key Combinators**:
- `Scan`: Sequential iteration (like `jax.lax.scan`)
- `Vmap`: Vectorization (like `jax.vmap`)
- `Cond`: Conditional branching (like `jax.lax.cond`)

##### Scan Usage

**CRITICAL**: Use static addressing only.

```python
# ✅ CORRECT: Static addressing
@gen
def step(carry, _):
    x = normal(carry, 1.0) @ "x"  # Static address
    return x, x

scan_fn = Scan(step, length=10)
result = scan_fn((0.0, None)) @ "scan"

# ❌ WRONG: Dynamic addressing breaks Scan
@gen  
def bad_step(carry, t):
    x = normal(0.0, 1.0) @ f"x_{t}"  # Dynamic - will break!
```

##### Vmap Usage

**Basic Pattern**:

```python
# Vectorize distributions
vectorized_normal = normal.vmap(in_axes=(0, None))
traces = vectorized_normal.simulate((mus, 1.0))

# Vectorize generative functions
batch_fn = Vmap(single_fn)
traces = batch_fn.simulate((vectorized_args,))

# Convenient independent sampling
batch_sampler = single_fn.repeat(10)
traces = batch_sampler.simulate(())
```

**Key Points**:
- `in_axes=(0, None)`: vectorize first arg, broadcast second
- `.repeat(n)`: shorthand for `Vmap(fn, axis_size=n)`
- `axis_size`: batch size when not inferrable from inputs

##### Cond Usage

```python
@gen
def branch_a():
    return exponential(1.0) @ "value"

@gen
def branch_b():
    return exponential(2.0) @ "value"

@gen
def conditional_model():
    x = normal(0.0, 1.0) @ "x"
    cond_fn = Cond(branch_a, branch_b)
    result = cond_fn((x > 0,)) @ "conditional"
    return result
```

**Constraints**: Both branches evaluated (JAX requirement), compatible return types, same addressing.

**Key Principles**:
- **Scan/Vmap**: Preserve addressing, vectorize leaf values
- **Cond**: Merge trace structures, select based on condition
- **Composition**: Rules apply recursively for nested combinators

## GenJAX API Patterns

**CRITICAL**: GenJAX generative functions specific API patterns that must be followed exactly:

**Generative Function Usage in `@gen` Functions**:

```python
# ✅ CORRECT: Use @ "address" syntax
x = normal(mu, sigma) @ "x"
y = exponential(rate) @ "y" 

# ❌ WRONG: Don't call methods directly
x = normal(mu, sigma)  # Won't be traced

# ❌ WRONG: Can't use kwargs 
x = normal(mu=mu, sigma=sigma) @ "x"
```

**Generative Function Interface APIs**:

```python
# ✅ CORRECT: Argument parameters as tuple
log_density, retval = normal.assess((mu, sigma), sample_value)
log_density, retval = exponential.assess((rate,), sample_value)

# ❌ WRONG: Argument parameters as constructor arguments  
log_density, retval = normal(mu, sigma).assess((), sample_value)  # Invalid
log_density, retval = exponential(rate).assess((), sample_value)  # Invalid
```

## JAX Integration

### JAX Compatibility Requirements

**ALWAYS use JAX patterns:**
- `jax.lax.cond()` for conditionals
- `jax.lax.scan()` for iteration  
- JAX arrays, not Python lists
- JAX operations, not Python built-ins

### PJAX: Probabilistic JAX

**IMPORTANT**: PJAX extends JAX with probabilistic primitives (`assume_p`, `log_density_p`).

**Key Transformations**:
- **`seed`**: Eliminates PJAX primitives → enables standard JAX transformations → requires explicit PRNG keys
- **`modular_vmap`**: Preserves PJAX primitives → specialized vectorization → automatic key management

**Usage**:
```python
# seed for jit, grad, custom inference
seeded_fn = seed(model.simulate)
jit_model = jax.jit(seeded_fn)

# modular_vmap for probabilistic vectorization
vmap_fn = modular_vmap(model.simulate, in_axes=(0,))
```


### Vectorization

**Key Principle**: Use JAX's automatic Pytree vectorization. GenJAX datatypes inherit from `Pytree` - use single vectorized instances, not Python lists.

## Development Overview

### Codebase Commands

The codebase is managed through [`pixi`](https://pixi.sh/latest/). All toplevel codebase commands should go through `pixi`.

#### Key Commands

```bash
pixi install              # Setup
pixi run format           # Format code
pixi run python examples/simple.py  # Run examples
pixi run betaber-timing   # Generate figures
```

### Codebase Architecture

```
src/genjax/
├── core.py           # PJAX, GFI, Trace, Distribution, Fn, combinators
├── distributions.py  # Standard distributions
├── adev.py          # Automatic differentiation of expected values
└── stdlib.py        # Inference utilities
```

### Development Workflow

1. Research existing patterns first
2. Follow established conventions
3. Run `pixi run format` before concluding
4. Verify JAX compilation compatibility


#### Testing Patterns

**Density Validation**:
```python
def test_model():
    trace = model.simulate(())
    choices = trace.get_choices()
    
    # Use Distribution.assess for validation
    expected_density = sum([dist.assess(params, choice)[0] for dist, params, choice in distributions])
    actual_density, _ = model.assess((), choices)
    assert jnp.allclose(actual_density, expected_density)
    assert jnp.allclose(trace.get_score(), -actual_density)
```

**Combinator Testing**:
- Compare against manual JAX implementations
- Test `simulate`/`assess` consistency
- Validate density computations

## Known Issues

- `vmap` within ADEV `@expectation` has undefined semantics
- Not all gradient strategies support batching
- Some JAX transformations may not compose cleanly

## Glossary

- **GFI**: Generative Function Interface (simulate, assess, generate, update, regenerate)
- **PJAX**: stands for _Probabilistic_ JAX, an extension of JAX with custom primitives (`sample_p`, `log_density_p`) which GenJAX uses as an intermediate representation of its probabilistic computations.
- **ADEV**: Automatic Differentiation of Expected Values
- **Trace**: Recording of the execution of a generative function.
- **Address**: String identifier for random variables (e.g., `"x"`, `"alpha"`, etc)
- **Score**: Negative log probability (log of the reciprocal `1/P(choices)`)
- **Programmable Inference**: Ability to customize inference algorithms

## Further Reading

- [GenJAX Documentation](https://probcomp.github.io/genjax/)
- [ADEV Paper](https://dl.acm.org/doi/10.1145/3571198) - Automatic differentiation of expected values
- [Programmable VI Paper](https://dl.acm.org/doi/10.1145/3656463) - Programmable variational inference
- [Gen Paper](https://dl.acm.org/doi/10.1145/3314221.3314642) - Original Gen system (Julia version)

---

_This codebase implements cutting-edge research in vectorized probabilistic programming. The design prioritizes flexibility and performance while maintaining mathematical correctness._
