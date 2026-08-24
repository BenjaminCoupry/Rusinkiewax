# Rusinkiewax

A small JAX-based Python package for working with Rusinkiewicz parameterizations of pairs of directions.

The package provides conversions between Cartesian and spherical coordinates, utilities for constructing local orthonormal bases, and conversions between cartesian direction pairs and the Rusinkiewicz parameterization.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <repository-url>
cd Rusinkiewax
python -m pip install -e .
```

## Usage

The basic conversions between incident/outgoing directions and the Rusinkiewicz parameterization:

```python
from rusinkiewax import (
    cartesian_to_rusinkiewicz,
    rusinkiewicz_to_cartesian,
)

theta_h, phi_h, theta_d, phi_d = cartesian_to_rusinkiewicz(
    w_i,
    w_o,
    normal,
    tangent,
)

w_i, w_o = rusinkiewicz_to_cartesian(
    theta_h,
    phi_h,
    theta_d,
    phi_d,
    normal,
    tangent,
)
```

Here, `normal` and `tangent` define the local reference frame. They only need to be non-collinear; the package constructs an orthonormal basis from them.

## Rusinkiewicz parameterization

For a pair of directions `w_i` and `w_o`, the parameterization is defined using the half-vector

```text
h = normalize(w_i + w_o)
```

and describes the configuration using four angles:

* `theta_h`, `phi_h`: spherical coordinates of the half-vector in the local reference frame.
* `theta_d`, `phi_d`: spherical coordinates of the incident direction in the frame associated with the half-vector.


## Reference

> Rusinkiewicz, S. M. (1998, June). A new change of variables for efficient BRDF representation. In Eurographics Workshop on Rendering Techniques (pp. 11-22). Vienna: Springer Vienna.

