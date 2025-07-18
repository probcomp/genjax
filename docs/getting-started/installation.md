# Installation

GenJAX can be installed using pip or conda package managers.

## Using pip

```bash
pip install genjax
```

## Using conda/mamba

```bash
conda install -c conda-forge genjax
```

## Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/femtomc/genjax.git
cd genjax
pip install -e .
```

## Dependencies

GenJAX requires:
- Python >= 3.12
- JAX >= 0.6.0
- TensorFlow Probability (JAX substrate)
- Beartype for runtime type checking
- Penzai for visualization