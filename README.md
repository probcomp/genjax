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
pixi install
pixi run test-fast
pixi run paper-figures
pixi run paper-figures-gpu
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
- Inspect all Pixi tasks in [pyproject.toml](pyproject.toml).

## Paper cases

| Case                      | Figures        | Command                                        |
| ------------------------- | -------------- | ---------------------------------------------- |
| Fair coin                 | 16a            | `pixi run paper-faircoin-gen`                  |
| Curve fitting             | 4–6            | `pixi run paper-curvefit-gen`                  |
| Multi-framework benchmark | 16b            | `pixi run paper-perfbench`                     |
| Game of Life              | 18             | `pixi run assets && pixi run -e gol gol-paper` |
| Localization              | 19             | `pixi run paper-localization-gen`              |
| AIR estimators            | PLDI 2024 port | `pixi run air-compare`                         |

- Add `--mode cuda` to `paper-perfbench` for its CUDA pipeline.
- CPU and GPU execute the same models but have different scaling curves.
- Figure 19 and paper-scale curve fitting require CUDA-like throughput to match
  the published timing/ESS panels.
- Gen.jl benchmark lanes require Julia 1.10 or newer.
- Generated figures are saved in `figs/`. Perfbench outputs to separate CPU and
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
