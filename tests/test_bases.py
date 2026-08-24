import jax
import pytest

from rusinkiewax.bases import (
    spherical_local_base,
    orthogonal,
    complete_base,
)


ATOL = 1.5e-5
N_TEST = 15