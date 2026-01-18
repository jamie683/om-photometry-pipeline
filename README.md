[![DOI](https://zenodo.org/badge/1136586707.svg)](https://doi.org/10.5281/zenodo.18285162)

# OM Photometry Pipeline

This repository contains a Python pipeline for time-series differential photometry
of CCD data obtained at the OM Dark Sky Observatory.

The pipeline performs:
- Gaussian centroid tracking and FWHM estimation
- FWHM-scaled aperture photometry
- RMS-optimised comparison star ensemble construction
- Position- and FWHM-based decorrelation
- Log-linear baseline removal
- Injection–recovery tests for sensitivity estimation
- Noise diagnostics (RMS vs bin size, β factor, autocorrelation)

The code is intended for instrument characterisation and systematics analysis.
It is not used for transit parameter inference.

## Requirements
- numpy
- pandas
- matplotlib
- astropy
- photutils
- scipy
- tqdm

## Usage
Edit the paths to the FITS cube, DS9 region file, and output directory at the
bottom of `OM_photometry.py`, then run:

```bash
python OM_photometry.py