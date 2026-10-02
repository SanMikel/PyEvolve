# PyEvolve
Porting UPDFevolv from F77 to Python3 and numba. At the moment only LO space-like evolution is available.

## Installation
Requires  python ≥ 3.10

```bash
git clone https://github.com/SanMikel/PyEvolve.git
cd PyEvolve

# Create and activate virtual environment
python3 -m venv .pyevolve
source .pyevolve/bin/activate

# Install PyEvolve and dependecies
pip install -e .
```

## Quick start
For a quick test run
```bash
python pdf_evolution_example.py
```
Flavour indices do not match PDG id at the moment g: 0, u: 1, d: 2, s: 3, c: 4, b: 5, antiparticles with a - sing in front



