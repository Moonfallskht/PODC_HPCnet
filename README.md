# HPCnet
Robust photon-level single-pixel imaging through diverse scattering media

## Simulation scripts
- **fog.py** — simplified fog simulation script for generating synthetic scattering-degraded images.  
  Parameters are not yet fully adapted to all datasets and may require tuning for different scattering conditions.

- **PODC_preprocess.m** — preprocessing script for scattering compensation based on percentile normalization and local statistical alignment.  
  Used to enhance contrast, suppress background noise, and reduce domain mismatch before network training or evaluation.

## Notes
The fog simulation script provides a baseline degradation model for subsequent preprocessing and network training (e.g., PODC, U-Net, and HPCnet).  
The PODC preprocessing script is applied as a domain-alignment and contrast-compensation step prior to reconstruction.
