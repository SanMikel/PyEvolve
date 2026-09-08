# PyEvolve
Porting UPDFevolv from F77 to python

### Evolution kernel
Parallelisation via numba, however, large fluctuations, investigating resampling or kernel smoothening 

### Space-like evolution
LO implemented, slight differences from LHAPDF evolution.

NLO on the way, Splitting functions and alpha_s available at NLO. Evolution not yet available sampling structure has to be changed. Remove the full splitting function weight from z_sampling.py and add it to parton evolution

### Time-like evolution
Not yet implemented


### Interpolation
At the moment only linear interpolation is available 
