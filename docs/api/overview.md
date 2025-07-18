# Core API Overview

GenJAX provides a powerful and composable API for probabilistic programming. The core concepts are:

## Generative Functions

The fundamental abstraction in GenJAX is the **generative function** - a probabilistic program that can be executed, scored, and manipulated through the Generative Function Interface (GFI).

### The `@gen` Decorator

Transform Python functions into generative functions:

```python
from genjax import gen, normal

@gen
def my_model(x):
    # Sample from distributions using @ for addressing
    z = normal(0, 1) @ "z"
    y = normal(z * x, 0.1) @ "y"
    return y
```

### Addressing with `@`

The `@` operator assigns addresses to random choices, creating a hierarchical namespace:

```python
@gen
def hierarchical_model():
    # Top-level choice
    global_mean = normal(0, 1) @ "global_mean"
    
    # Nested choices
    for i in range(3):
        local_mean = normal(global_mean, 0.5) @ f"group_{i}/mean"
        for j in range(5):
            obs = normal(local_mean, 0.1) @ f"group_{i}/obs_{j}"
```

## Generative Function Interface (GFI)

Every generative function implements these core methods:

### `simulate(args...) -> Trace`

Forward sampling from the model:

```python
trace = model.simulate(x=2.0)
choices = trace.get_choices()  # {"z": 0.5, "y": 1.1}
retval = trace.get_retval()    # 1.1
```

### `assess(choices, args...) -> (log_density, retval)`

Evaluate the log probability density:

```python
choices = {"z": 0.5, "y": 1.0}
log_prob, retval = model.assess(choices, x=2.0)
```

### `generate(constraints, args...) -> (trace, weight)`

Generate a trace with some choices constrained:

```python
constraints = {"y": 1.5}  # Fix observation
trace, weight = model.generate(constraints, x=2.0)
# weight = log p(y=1.5, z) / q(z | y=1.5)
```

### `update(trace, constraints, args...) -> (new_trace, weight, discard)`

Update an existing trace with new constraints:

```python
new_constraints = {"y": 2.0}
new_trace, weight, discard = model.update(trace, new_constraints, x=2.0)
```

### `regenerate(trace, selection, args...) -> (new_trace, weight, discard)`

Selectively regenerate parts of a trace:

```python
from genjax import sel

selection = sel("z")  # Regenerate only z
new_trace, weight, discard = model.regenerate(trace, selection, x=2.0)
```

## Traces

Traces record the execution of generative functions:

```python
trace = model.simulate(x=2.0)

# Access components
choices = trace.get_choices()      # Random choices
retval = trace.get_retval()        # Return value
score = trace.get_score()          # log(1/p(choices))
args = trace.get_args()            # Function arguments
gen_fn = trace.get_gen_fn()        # Source generative function
```

## Distributions

Built-in probability distributions that implement the GFI:

```python
from genjax import normal, beta, categorical, bernoulli

# Continuous distributions
x = normal(mu=0, sigma=1) @ "x"
p = beta(alpha=2, beta=2) @ "p"

# Discrete distributions  
k = categorical(probs=jnp.array([0.2, 0.3, 0.5])) @ "k"
b = bernoulli(p=0.7) @ "b"
```

## Combinators

Higher-order generative functions for composition:

### Map/Vmap

Vectorized execution:

```python
# Map model over multiple inputs
vectorized = model.vmap()
traces = vectorized.simulate(jnp.array([1.0, 2.0, 3.0]))
```

### Scan

Sequential execution with state threading:

```python
from genjax import Scan, const

@gen
def step(state, x):
    new_state = normal(state + x, 0.1) @ "state"
    return new_state, new_state

scan_model = Scan(step, const(10))  # 10 steps
trace = scan_model.simulate(init_state=0.0, xs=jnp.ones(10))
```

### Cond

Conditional execution:

```python
from genjax import Cond

@gen
def model_a():
    return normal(0, 1) @ "x"

@gen  
def model_b():
    return normal(5, 2) @ "x"

cond_model = Cond(model_a, model_b)
trace = cond_model.simulate(condition=True)  # Uses model_a
```

## Selections

Target specific addresses for operations:

```python
from genjax import sel, Selection, AllSel

# Select specific addresses
s1 = sel("x")                    # Select "x"
s2 = sel("group_0", "mean")      # Select "group_0/mean"

# Combine selections
s_or = sel("x") | sel("y")       # Select x OR y
s_and = sel("x") & sel("y")      # Select x AND y (intersection)
s_not = ~sel("x")                # Select everything except x

# Select all
s_all = Selection(AllSel())      # Select all addresses
```

## Next Steps

- Learn about [Generative Functions](generative-functions.md) in detail
- Explore available [Distributions](distributions.md)
- Understand [Traces](traces.md) and their structure
- Master [Combinators](combinators.md) for model composition