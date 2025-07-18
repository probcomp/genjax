# genjax.core

Core functionality for GenJAX including the Generative Function Interface, traces, and model construction.

## Mathematical Foundation

The Generative Function Interface (GFI) is based on measure theory. A generative function $g$ defines:

- A measure kernel $P(dx; \text{args})$ over measurable space $X$
- A return value function $f(x, \text{args}) \rightarrow R$
- Internal proposal family $Q(dx'; \text{args}, x)$

The importance weight from `generate` is:

$$w = \log \frac{P(\text{all\_choices})}{Q(\text{free\_choices} | \text{constrained\_choices})}$$

::: genjax.core
    options:
      show_source: true
      show_bases: true
      show_signature_annotations: true
      members_order: source
      filters:
        - "!^_"  # Exclude private members
      docstring_section_style: google
      show_docstring_attributes: true
      show_docstring_description: true
      show_docstring_examples: true
      show_docstring_parameters: true
      show_docstring_returns: true
      show_docstring_yields: true
      show_docstring_raises: true
      show_docstring_warns: true
      show_docstring_other_parameters: true

## Live Examples

### Basic Model Definition

```python exec="true" source="material-block"
import jax
import jax.numpy as jnp
from genjax import gen, distributions

@gen
def coin_flip_model(n_flips):
    """A simple coin flipping model with unknown bias."""
    bias = distributions.beta(1.0, 1.0) @ "bias"
    flips = []
    for i in range(n_flips):
        flip = distributions.bernoulli(bias) @ f"flip_{i}"
        flips.append(flip)
    return jnp.array(flips)

print("Model defined successfully!")
```

### Assessing Log Probability

```python exec="true" source="material-block"
import jax
import jax.numpy as jnp
from genjax import gen, distributions

@gen
def coin_flip_model(n_flips):
    """A simple coin flipping model with unknown bias."""
    bias = distributions.beta(1.0, 1.0) @ "bias"
    flips = []
    for i in range(n_flips):
        flip = distributions.bernoulli(bias) @ f"flip_{i}"
        flips.append(flip)
    return jnp.array(flips)

# Assess the log probability of specific choices
choices = {"bias": 0.7, "flip_0": 1, "flip_1": 1, "flip_2": 0}
log_prob, retval = coin_flip_model.assess(choices, 3)

print(f"Given choices: {choices}")
print(f"Log probability: {log_prob:.3f}")
print(f"Return value (flips): {retval}")
```

### Using Selections

```python exec="true" source="material-block"
from genjax import sel, Selection

# Create various selections
s1 = sel("bias")  # Select only bias
s2 = sel("flip_0", "flip_1")  # Select two flips
s3 = sel("bias") | sel("flip_2")  # Select bias OR flip_2

print(f"Selection s1 targets: bias")
print(f"Selection s2 targets: flip_0, flip_1")
print(f"Selection s3 targets: bias or flip_2")
```