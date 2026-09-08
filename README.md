# PyEvolve
Porting UPDFevolv from F77 to python

### Evolution kernel
Parallelisation via numba, however, large fluctuations, investigating resampling or kernel smoothening 

### Space-like evolution
#### LO: 
Implemented. Slight differences from LHAPDF evolution.

#### NLO:
Splitting functions and alpha_s available. 

Evolution Kernel not yet available. The sampling structure has to be changed by removing the full splitting function weight from 
https://github.com/SanMikel/PyEvolve/blob/128d4fe575d225f16e78742f33f8e692c916e96c/numba_src/z_sampling.py#L51
and add it to 

### Time-like evolution
Not yet implemented


### Interpolation
At the moment only linear interpolation is available 
