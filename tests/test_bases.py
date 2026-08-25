import jax
import pytest

from rusinkiewax.bases import (
    spherical_local_base,
    orthogonal,
    complete_base,
)


ATOL = 1.5e-5
N_TEST = 15


def random_spherical(key):
    """Generate a random spherical coordinate."""
    key_rho, key_theta, key_phi = jax.random.split(key, 3)

    rho = jax.random.uniform(
        key_rho,
        minval=0.1,
        maxval=2.0,
    )
    theta = jax.random.uniform(
        key_theta,
        minval=0.0,
        maxval=jax.numpy.pi,
    )
    phi = jax.random.uniform(
        key_phi,
        minval=0.0,
        maxval=2.0 * jax.numpy.pi,
    )

    return jax.numpy.stack([rho, theta, phi])

@pytest.mark.parametrize("seed", range(N_TEST))
def test_spherical_local_base_is_orthonormal(seed):
    key = jax.random.key(seed)
    spherical = random_spherical(key)
    base = spherical_local_base(spherical)
    
    assert jax.numpy.allclose(
        base.T @ base,
        jax.numpy.eye(3),
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        jax.numpy.linalg.det(base),
        1.0,
        atol=ATOL,
    )
    
@pytest.mark.parametrize("seed", range(N_TEST))
def test_orthogonal_is_orthogonal_to_input(seed):
    key = jax.random.key(seed)
    vector = jax.random.normal(key, (3,))
    orthogonal_vector = orthogonal(vector)
    
    assert jax.numpy.allclose(
        jax.numpy.dot(vector, orthogonal_vector),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.linalg.vector_norm(orthogonal_vector) > ATOL
    
@pytest.mark.parametrize("seed", range(N_TEST))
def test_complete_base_is_orthonormal(seed):
    key_u, key_v = jax.random.split(
        jax.random.key(seed),
        2,
    )
    key_u, key_v = jax.random.split(jax.random.key(seed))
    u = jax.random.normal(key_u, (3,))
    v = jax.random.normal(key_v, (3,))
    base = complete_base(jax.numpy.stack([u, v], axis=-1))
    
    assert jax.numpy.allclose(
        base.T @ base,
        jax.numpy.eye(3),
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        jax.numpy.linalg.det(base),
        1.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(base[:, 0], u, atol=ATOL)

