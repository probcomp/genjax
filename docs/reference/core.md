# genjax.core

Core functionality for GenJAX including the Generative Function Interface, traces, and model construction.

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

### Basic Model

```pycon exec="true" source="material-block"
>>> import jax
>>> import jax.numpy as jnp
>>> from genjax import gen, distributions
>>> 
>>> @gen
... def coin_flip_model(n_flips):
...     bias = distributions.beta(1.0, 1.0) @ "bias"
...     flips = []
...     for i in range(n_flips):
...         flip = distributions.bernoulli(bias) @ f"flip_{i}"
...         flips.append(flip)
...     return jnp.array(flips)
... 
>>> key = jax.random.PRNGKey(42)
>>> trace = coin_flip_model.simulate(key, (3,))
>>> print(f"Sampled bias: {trace['bias']:.3f}")
>>> print(f"Flips: {trace.retval}")
```

### Using the GFI

```pycon exec="true" source="material-block"
>>> # Assess density at specific choices
>>> choices = {"bias": 0.7, "flip_0": 1, "flip_1": 1, "flip_2": 0}
>>> log_prob, retval = coin_flip_model.assess(choices, (3,))
>>> print(f"Log probability: {log_prob:.3f}")
>>> print(f"Return value: {retval}")
```

### Constrained Generation

```pycon exec="true" source="material-block"
>>> # Fix some choices and sample others
>>> constraints = {"flip_0": 1, "flip_1": 1}
>>> trace = coin_flip_model.generate(key, constraints, (3,))
>>> print(f"Generated bias: {trace['bias']:.3f}")
>>> print(f"All flips: {trace.retval}")
>>> print(f"Score: {trace.score:.3f}")
```