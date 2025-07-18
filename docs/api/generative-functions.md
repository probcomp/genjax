# Generative Functions

Generative functions are the core abstraction in GenJAX. They represent probabilistic computations that can be executed, scored, and manipulated through a unified interface.

## Creating Generative Functions

### The `@gen` Decorator

Transform regular Python functions into generative functions:

```python
from genjax import gen, normal, bernoulli

@gen
def weather_model(temp_yesterday):
    # Sample today's temperature
    temp_today = normal(temp_yesterday, 5.0) @ "temp"
    
    # Determine if it rains based on temperature
    rain_prob = 1 / (1 + jnp.exp(0.1 * (temp_today - 20)))
    rains = bernoulli(rain_prob) @ "rains"
    
    return {"temperature": temp_today, "rains": rains}
```

### Addressing Random Choices

Use the `@` operator to assign addresses to random choices:

```python
@gen
def model():
    # Simple address
    x = normal(0, 1) @ "x"
    
    # Hierarchical addresses
    for i in range(3):
        # Creates addresses: "group_0", "group_1", "group_2"
        group_mean = normal(0, 1) @ f"group_{i}"
        
        for j in range(5):
            # Creates: "obs_0_0", "obs_0_1", ..., "obs_2_4"
            obs = normal(group_mean, 0.1) @ f"obs_{i}_{j}"
```

!!! warning "Avoid Address Collisions"
    Each address at the same level must be unique. GenJAX will raise an error if you reuse addresses:
    
    ```python
    @gen
    def bad_model():
        x = normal(0, 1) @ "x"
        y = normal(1, 1) @ "x"  # Error: address "x" already used!
    ```

## The Generative Function Interface

All generative functions implement these methods:

### simulate

Forward sampling from the generative function:

```python
# Without arguments
trace = model.simulate()

# With arguments
trace = weather_model.simulate(temp_yesterday=25.0)

# Access the trace
choices = trace.get_choices()
return_value = trace.get_retval()
```

### assess

Evaluate the log probability density at given choices:

```python
choices = {
    "temp": 22.0,
    "rains": True
}

log_prob, retval = weather_model.assess(choices, temp_yesterday=25.0)
# log_prob = log p(temp=22.0, rains=True | temp_yesterday=25.0)
```

### generate

Generate a trace with some choices constrained:

```python
# Observe that it rained
constraints = {"rains": True}

trace, weight = weather_model.generate(constraints, temp_yesterday=25.0)
# weight = log p(rains=True, temp) / q(temp | rains=True)
```

The weight is the incremental importance weight, useful for:
- Importance sampling
- Particle filtering
- MCMC acceptance probabilities

### update

Update an existing trace with new constraints:

```python
# Original trace
trace = weather_model.simulate(temp_yesterday=25.0)

# Update with new observation
new_constraints = {"rains": False}
new_trace, weight, discard = weather_model.update(
    trace, 
    new_constraints, 
    temp_yesterday=26.0  # Can also change arguments
)
```

### regenerate

Selectively regenerate parts of a trace:

```python
from genjax import sel

# Regenerate only the temperature
selection = sel("temp")
new_trace, weight, discard = weather_model.regenerate(
    trace,
    selection,
    temp_yesterday=25.0
)
```

## Composing Generative Functions

### Calling Other Generative Functions

```python
@gen
def prior():
    mean = normal(0, 10) @ "mean"
    std = gamma(1, 1) @ "std"
    return mean, std

@gen
def model(n_obs):
    # Call another generative function
    mean, std = prior() @ "prior"
    
    # Use the results
    observations = []
    for i in range(n_obs):
        obs = normal(mean, std) @ f"obs_{i}"
        observations.append(obs)
    
    return jnp.array(observations)
```

### Using Fixed Values

Wrap deterministic values to preserve them during trace operations:

```python
from genjax import Fixed

@gen
def model_with_fixed():
    # This value won't be regenerated
    fixed_param = Fixed(1.0) @ "param"
    
    # This can be regenerated
    x = normal(fixed_param, 1.0) @ "x"
    
    return x
```

## Advanced Patterns

### Mixture Models

```python
@gen
def mixture_model(data):
    # Mixture weights
    weights = dirichlet(jnp.ones(3)) @ "weights"
    
    # Component parameters
    means = []
    for k in range(3):
        mean = normal(0, 10) @ f"mean_{k}"
        means.append(mean)
    
    # Assign data to components
    for i, datum in enumerate(data):
        component = categorical(weights) @ f"z_{i}"
        obs = normal(means[component], 1.0) @ f"obs_{i}"
```

### Recursive Models

```python
@gen
def geometric(p, max_depth=100):
    """Sample from geometric distribution recursively."""
    flip = bernoulli(p) @ f"flip_0"
    
    if flip:
        return 0
    else:
        # Recursive call
        rest = geometric(p, max_depth-1) @ "rest"
        return 1 + rest
```

### State Space Models

```python
@gen
def state_space_model(T, observations):
    # Initial state
    state = normal(0, 1) @ "state_0"
    
    states = [state]
    for t in range(1, T):
        # State transition
        state = normal(state, 0.1) @ f"state_{t}"
        states.append(state)
        
        # Observation
        if observations[t] is not None:
            obs = normal(state, 0.5) @ f"obs_{t}"
            # Could add constraint: obs == observations[t]
    
    return jnp.array(states)
```

## Best Practices

1. **Use descriptive addresses**: Make debugging easier with meaningful names
2. **Avoid address collisions**: Each address at the same level must be unique
3. **Minimize Python loops**: Use JAX/GenJAX combinators when possible
4. **Type annotations**: Help with debugging and documentation
5. **Return structured data**: Return dictionaries or named tuples for clarity

## Common Pitfalls

!!! danger "Python Control Flow in JAX"
    Avoid Python `if`/`for` statements when you need JAX compilation:
    
    ```python
    # Bad - won't work with JAX transformations
    @gen
    def bad_model(n):
        for i in range(n):  # Python loop with traced value!
            x = normal(0, 1) @ f"x_{i}"
    
    # Good - use Scan combinator
    from genjax import Scan
    
    @gen
    def step(carry, i):
        x = normal(0, 1) @ "x"
        return carry, x
    
    model = Scan(step, const(n))
    ```

!!! tip "Performance Tips"
    - Use `Fixed` for values that don't need regeneration
    - Batch operations with `vmap` instead of loops
    - Prefer built-in distributions over custom implementations