## Description

Abstract of NS-DMD (https://ieeexplore.ieee.org/document/10288443):

Many physical processes display complex high-dimensional time-varying behavior, from global weather patterns to brain activity. An outstanding challenge is to express high dimensional data in terms of a dynamical model that reveals their spatiotemporal structure. Dynamic Mode Decomposition is a means to achieve this goal, allowing the identification of key spatiotemporal modes through the diagonalization of a finite dimensional approximation of the Koopman operator. However, DMD methods apply best to time-translationally invariant or stationary data, while in many typical cases, dynamics vary across time and conditions.  To capture this temporal evolution, we developed a method, Non-Stationary Dynamic Mode Decomposition (NS-DMD), that generalizes DMD by fitting global modulations of drifting spatiotemporal modes. This method accurately predicts the temporal evolution of modes in simulations and recovers previously known results from simpler methods. To demonstrate its properties, the method is applied to multi-channel recordings from an awake behaving non-human primate performing a cognitive task.

This repository provides a Python implementations of NS-DMD along with example code.

To start, I recommend using examples/fit_demo.ipynb. Examples from the manuscript, including supplementary ones, are included in the examples folder.

## Installation

### Installing from source

First clone the repository with
```
git clone https://github.com/learning-2-learn/nsdmd.git
```

Then install locally and in development mode via
```
pip install -e .
```
