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

## Key Classes and Functions

### GenerativeFunction
The base class for all generative functions in GenJAX.

### Trace
Represents an execution trace of a generative function.

### @gen Decorator
Transform Python functions into generative functions.

## Usage Examples

```python
from genjax import gen, normal

@gen
def my_model(x):
    z = normal(0, 1) @ "z"
    y = normal(z * x, 0.1) @ "y"
    return y

# Use the generative function
trace = my_model.simulate(key, (2.0,))
```