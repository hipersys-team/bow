# BOW: Bayesian-Optimized Wavelengths

## 1. Overview
BOW is a bayesian optimization system that optimizes optical wavelengths' Quality of Transmission (QoT) metrics (e.g., OSNR) for wavelength reconfigurations. BOW is built on Python 3.8.8, with Ax (https://ax.dev) as the Bayesian Optimization backend, GNPy (https://gnpy.readthedocs.io/en/master/) as the optical-layer QoT estimator, and FCR (https://github.com/facebookincubator/FCR) as the control interface to optical network devices.

For a full technical description on BOW, please read our OFC 2021 paper: 

> Z. Zhong, M. Ghobadi, M. Balandat, S. Katti, A. Kazerouni, J. Leach, M. McKillop, Y. Zhang, "BOW: First Real-World Demonstration of a Bayesian Optimization System for Wavelength Reconfiguration," OFC, 2021. http://bow.csail.mit.edu/files/OFC-21-BOW-final.pdf

For more details on BOW, please visit our website: http://bow.csail.mit.edu


## 2. Requirement
* Python 3.8
* Ax 0.1.20
* GNPy 2.1

## 3. License
BOW is MIT-licensed.
