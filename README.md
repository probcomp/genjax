<p align="center">
  <img src="https://raw.githubusercontent.com/a-tiny-project/genjax/main/logo.png" alt="genjax" width="360" />
</p>

[![DOI](https://zenodo.org/badge/971731825.svg)](https://doi.org/10.5281/zenodo.17342547)

- **Purpose:** Vectorized probabilistic programming with generative functions
  and programmable inference in JAX.
- **Formal foundations:** A mechanized Lean model proves product density
  preservation and fundamental vectorization for finite models under explicit
  primitive conformance certificates. Concrete execution maps vectorized
  generative primitives to XLA kernels, separating denotational density
  semantics from machine array compilation.
- **POPL 2026 artifact:**
  [v1.0.10](https://github.com/a-tiny-project/genjax/releases/tag/v1.0.10)

## Use

```sh
git clone https://github.com/a-tiny-project/genjax.git
cd genjax
uv sync --locked --group test
uv run --locked --group test pytest tests/ -o addopts="" -q -n 4
uv run --locked --group test xdoctest src/genjax
```

```python
from genjax import gen, normal

@gen
def model():
    return normal(0.0, 1.0) @ "x"

trace = model.simulate()
choices = trace.get_choices()
```

- Generative functions expose `simulate`, `generate`, `assess`, and `update`.
- `vmap` and `modular_vmap` lift model and inference structure over explicit
  array axes.

## Development

The lock covers Linux x86_64 and macOS arm64. The test group runs on Python
3.12. Coverage uses the same group:

```sh
uv run --locked --group test pytest tests/ -v --cov=src/genjax --cov-report=xml --cov-report=html --cov-report=term
uv run --locked --group test coverage report --show-missing
```

The format group holds the formatting tools. Run the hooks with:

```sh
uv run --locked --group format pre-commit run --all-files
uv run --locked --group format pre-commit run prettier --hook-stage manual --all-files
```

Pre-commit supplies Node for Prettier. To remove generated figures:

```sh
find examples -type f -path '*/figs/*' \( -name '*.pdf' -o -name '*.png' -o -name '*.jpg' -o -name '*.svg' \) -delete
```

## Paper cases

Run these commands from the project root. Each example has its own dependency
group, which uv installs from `uv.lock` as needed.

| Case                      | Figures        | Command                                                                                                                                                                                  |
| ------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fair coin                 | 16a            | `uv run --locked --group faircoin python -m examples.faircoin.main --combined --num-obs 50 --num-samples 2000 --repeats 10`                                                              |
| Curve fitting             | 4–6            | `uv run --locked --group curvefit python -m examples.curvefit.main paper`                                                                                                                |
| Multi-framework benchmark | 16b            | `uv run --locked --group perfbench python examples/perfbench/main.py pipeline`                                                                                                           |
| Game of Life              | 18             | `uv run --locked --group gol python -m examples.gol.main --mode showcase`                                                                                                                |
| Localization              | 19             | `uv run --locked --group localization python -m examples.localization.main paper --include-smc-comparison --n-particles 200 --n-steps 8 --timing-repeats 3 --n-rays 8 --output-dir figs` |
| AIR estimators            | PLDI 2024 port | `uv run --locked python -m examples.air.main compare --small-config --num-examples 256 --epochs 2`                                                                                       |

Create the shared output directory with `mkdir -p figs`, then run the fair coin,
curve fitting, Game of Life, and localization commands in the table to produce
the paper figures. The AIR training command is
`uv run --locked python -m examples.air.main train --estimator enum --small-config --num-examples 256 --epochs 2`.
The cone figures use
`uv run --locked --extra viz python -m examples.cone.main fig2` and
`uv run --locked --extra viz python -m examples.cone.main table4`.

The Game of Life raster is an untracked input. Fetch the pinned copy before
running that example and verify its digest:

```sh
mkdir -p examples/gol/assets
curl -fsSL --retry 3 -o examples/gol/assets/wizards.jpg \
  https://raw.githubusercontent.com/a-tiny-project/genjax/fec393b015e6113585f1cd425d2a6817a842b1d3/examples/gol/assets/wizards.jpg
printf '%s\n' 'c680998ab5c25a505fdf0e1e03181d4275fbde66e1f1eb1da4cad13bf8db5cb1  examples/gol/assets/wizards.jpg' | sha256sum -c -
```

Fetch the multi-MNIST data for AIR with
`uv run --locked --group perfbench-pyro python -m examples.air.main fetch-data --output examples/air/data/multi_mnist_uint8.npz --cache-root /tmp/air-data`.
For GPU figures, add `--group cuda` to the corresponding uv command; the
perfbench CUDA pipeline also takes `--mode cuda`. GPU AIR runs keep their
shared-memory temporary directory and XLA flags:

```sh
TMPDIR=/dev/shm XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0' uv run --locked --group cuda python -m examples.air.main train --dataset multi-mnist --data-path examples/air/data/multi_mnist_uint8.npz --num-examples 2048 --epochs 10 --batch-size 32 --eval-batch-size 128 --learning-rate 1e-4 --estimator enum
TMPDIR=/dev/shm XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0' uv run --locked --group cuda python -m examples.air.main compare --dataset multi-mnist --data-path examples/air/data/multi_mnist_uint8.npz --num-examples 2048 --epochs 10 --batch-size 32 --eval-batch-size 128 --learning-rate 1e-4
```

CPU and GPU execute the same models but have different scaling curves. Figure 19
and paper-scale curve fitting require CUDA-like throughput to match the
published timing/ESS panels. Gen.jl benchmark lanes require Julia 1.10 or newer.
Generated figures are saved in `figs/`; perfbench outputs to separate CPU and
CUDA directories.

## Code

- [Public package](src/genjax/__init__.py)
- [Generative-function core](src/genjax/core.py)
- [Distributions](src/genjax/distributions.py)
- [Probabilistic vectorization](src/genjax/pjax.py)
- [Inference](src/genjax/inference/)
- [ADEV](src/genjax/adev/)
- [Tests](tests/)
- [Examples](examples/)
- [Performance benchmark](examples/perfbench/README.md)
- [Citation metadata](CITATION.cff)
- [Package references](src/genjax/REFERENCES.md)

## References

- [Probabilistic Programming with Vectorized Programmable Inference](https://doi.org/10.1145/3776729)
- [Artifact DOI](https://doi.org/10.5281/zenodo.17342547)
- [Gen: programmable inference](https://doi.org/10.1145/3314221.3314642)
- [ADEV](https://doi.org/10.1145/3571198)
- [Programmable variational inference](https://doi.org/10.1145/3656463)

## Acknowledgments

GenJAX 1.0 continues the GenJAX project, whose 0.x releases were developed in
[genjax-community/genjax](https://github.com/genjax-community/genjax) from 2022
to 2025. GenJAX thanks the 22 people other than the maintainer who contributed
commits to that codebase:

[Matthew Brulhardt](https://github.com/mwbrulhardt),
[Jacob Burnim](https://github.com/jburnim),
[Guillaume Dalle](https://github.com/gdalle),
[Arijit Dasgupta](https://github.com/arijit-dasgupta),
[Cameron Freer](https://github.com/cameronfreer),
[Matin Ghavami](https://github.com/mugamma),
[Alex Hiser](https://github.com/ahiser1117),
[Matt Huebert](https://github.com/mhuebert),
[Mathieu Huot](https://github.com/MathieuHuot),
[Mirko Klukas](https://github.com/mirkoklukas),
[Urs Köster](https://github.com/ursk), [Ben Lee](https://github.com/midfield),
[Ian Limarta](https://github.com/limarta),
[Joao Loula](https://github.com/Joaoloula),
[David R. MacIver](https://github.com/DRMacIver),
[George Matheos](https://github.com/georgematheos),
[Jay Pottharst](https://github.com/sharlaon),
[Sam Ritchie](https://github.com/sritchie), Rif A. Saurous,
[Colin Smith](https://github.com/littleredcomputer),
[Xiaoyan Wang](https://github.com/horizon-blue),
[Fabian Zaiser](https://github.com/fzaiser).

## License

Apache-2.0. See [LICENSE](LICENSE.md).
