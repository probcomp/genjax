# Performance benchmark case study

- **Purpose:** Reproduce POPL 2026 Figure 16b across GenJAX, NumPyro, Pyro,
  TensorFlow Probability, hand-coded JAX/PyTorch, and Gen.jl.

## Use

- From `research/genjax/`:

```sh
uv run --locked --group perfbench python examples/perfbench/main.py pipeline
uv run --locked --group perfbench --group cuda python examples/perfbench/main.py pipeline --mode cuda
uv run --locked --group perfbench python examples/perfbench/main.py pipeline --inference is
uv run --locked --group perfbench python examples/perfbench/main.py pipeline --inference hmc
uv run --locked --group perfbench python examples/perfbench/main.py pipeline --frameworks genjax numpyro handcoded_jax
uv run --locked --group perfbench python examples/perfbench/main.py clean
```

- Direct orchestration:

```sh
uv run --locked --group perfbench python examples/perfbench/main.py pipeline --help
```

- CPU output: `data_cpu/` and `figs_cpu/`.
- CUDA output: `data/` and `figs/`.
- Resume with the `--skip-*` flags shown by `--help`.
- Gen.jl lanes require Julia 1.10 or newer.
- Framework-specific uv groups and repeat caps are encoded in the pipeline. Each
  group keeps its own local environment during a pipeline run.

## Code

- [Pipeline](main.py)
- [Benchmark runners](benchmarks/)
- [Framework adapters](benchmarks/src/timing_benchmarks/curvefit_benchmarks/)
- [Result merge and plotting](benchmarks/combine_results.py)
- [Dependency groups](../../pyproject.toml)
- [Parent artifact index](../../README.md)

## References

- [POPL 2026 paper](https://doi.org/10.1145/3776729)
- Imported timing benchmark baseline: `timing-benchmarks@d4433b0`.

## License

Apache-2.0. See [LICENSE](../../LICENSE.md).
