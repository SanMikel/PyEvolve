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
```python
import numpy as np
from numba_src.PDFEvolution import PDFEvolution

def dummy_pdf(x, A, B, C):
    return A*(x**B)*(1-x)**C

evo = PDFEvolution(n_samples=1_000_000, n_cores=5) #Builds evolution kernels

xgrid = np.logspace(-5, 0, 100)
# We set the gluon starting distribution gluon id=0
evo.set_starting_distribution(xgrid, dummy_pdf(xgrid), 0)

# Evolve PDF to mu_f of choice
xgrid, pdf, pdf_w2 = evo.evolve(10.0)

g_pdf = pdf[0]; g_pdf_w2 = pdf_w2[0]
```

Flavour indices do not match PDG id at the moment g: 0, u: 1, d: 2, s: 3, c: 4, b: 5, antiparticles with a - sing in front



