import jax
import pytest

from rusinkiewax import (
    cartesian_to_spherical,
    cartesian_to_rusinkiewicz,
    rusinkiewicz_to_cartesian,
    spherical_to_cartesian,
)


ATOL = 1.5e-5
N_TEST = 15


def angular_error(a, b):
    """Absolute angular error modulo 2π."""
    return jax.numpy.abs(
        jax.numpy.mod(a - b + jax.numpy.pi, 2.0 * jax.numpy.pi)
        - jax.numpy.pi
    )

def random_unit_vector(key):
    """Generate a random unit vector in R³."""
    vector = jax.random.normal(key, (3,))
    return vector / jax.numpy.linalg.norm(vector)


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


def random_rusinkiewicz(key):
    """Generate valid Rusinkiewicz coordinates."""
    key_theta_h, key_phi_h, key_theta_d, key_phi_d = jax.random.split(
        key, 4
    )

    theta_h = jax.random.uniform(
        key_theta_h,
        minval=0.0,
        maxval=jax.numpy.pi / 2.0,
    )
    phi_h = jax.random.uniform(
        key_phi_h,
        minval=0.0,
        maxval=2.0 * jax.numpy.pi,
    )
    theta_d = jax.random.uniform(
        key_theta_d,
        minval=0.0,
        maxval=jax.numpy.pi / 2.0,
    )
    phi_d = jax.random.uniform(
        key_phi_d,
        minval=0.0,
        maxval=2.0 * jax.numpy.pi,
    )

    return theta_h, phi_h, theta_d, phi_d

@pytest.mark.parametrize("seed", range(N_TEST))
def test_spherical_to_cartesian_round_trip(seed):
    key = jax.random.key(seed)

    spherical = random_spherical(key)

    cartesian = spherical_to_cartesian(spherical)
    spherical_test = cartesian_to_spherical(cartesian)

    rho, theta, phi = spherical
    rho_test, theta_test, phi_test = spherical_test

    assert jax.numpy.allclose(rho_test, rho, atol=ATOL)
    assert jax.numpy.allclose(
        angular_error(theta_test, theta),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_test, phi),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_cartesian_to_spherical_round_trip(seed):
    key = jax.random.key(seed)

    cartesian = jax.random.normal(key, (3,))

    spherical = cartesian_to_spherical(cartesian)
    cartesian_test = spherical_to_cartesian(spherical)

    assert jax.numpy.allclose(
        cartesian_test,
        cartesian,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_rusinkiewicz_to_cartesian_round_trip(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    theta_h, phi_h, theta_d, phi_d = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    theta_h_test, phi_h_test, theta_d_test, phi_d_test = (
        cartesian_to_rusinkiewicz(
            w_i,
            w_o,
            normal,
            tangent,
        )
    )

    assert jax.numpy.allclose(
        angular_error(theta_h_test, theta_h),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_h_test, phi_h),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(theta_d_test, theta_d),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_d_test, phi_d),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_cartesian_to_rusinkiewicz_round_trip(seed):
    key_wi, key_wo, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        4,
    )

    w_i = random_unit_vector(key_wi)
    w_o = random_unit_vector(key_wo)
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    theta_h, phi_h, theta_d, phi_d = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )

    w_i_test, w_o_test = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        w_i_test,
        w_i,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        w_o_test,
        w_o,
        atol=ATOL,
    )

@pytest.mark.parametrize("seed", range(N_TEST))
def test_theta_h_zero_reflects_over_normal(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    _, phi_h, theta_d, phi_d = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    theta_h = 0.0
   
    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )
    
    w_i_reflected = -w_i + 2.0 * jax.numpy.dot(w_i, normal) * normal

    assert jax.numpy.allclose(w_o, w_i_reflected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_theta_d_phi_d_half_pi_gives_opposite_and_tangent_directions(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    theta_h, phi_h, _, _ = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    theta_d = jax.numpy.pi / 2.0
    phi_d = jax.numpy.pi / 2.0

    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        w_i,
        -w_o,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        jax.numpy.dot(w_i, normal),
        0.0,
        atol=ATOL,
    )
    
    assert jax.numpy.allclose(
        jax.numpy.dot(w_o, normal),
        0.0,
        atol=ATOL,
    )
    
@pytest.mark.parametrize("seed", range(N_TEST))
def test_theta_h_phi_d_half_pi_gives_tangent_directions(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    _, phi_h, theta_d, _ = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    theta_h = jax.numpy.pi / 2.0
    phi_d = jax.numpy.pi / 2.0

    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        jax.numpy.dot(w_i, normal),
        0.0,
        atol=ATOL,
    )
    
    assert jax.numpy.allclose(
        jax.numpy.dot(w_o, normal),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_phi_d_plus_pi_exchanges_wi_and_wo(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    theta_h, phi_h, theta_d, phi_d = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    w_i_test, w_o_test = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d + jax.numpy.pi,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(w_i_test, w_o, atol=ATOL)
    assert jax.numpy.allclose(w_o_test, w_i, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_theta_d_zero_gives_equal_directions(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    theta_h, phi_h, _, phi_d = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    theta_d = 0.0
    
    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(w_i, w_o, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_phi_d_zero_and_equal_theta_gives_outgoing_normal(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    theta_h, phi_h, _, _ = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    phi_d = 0.0
    theta_d = theta_h

    _, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(w_o, normal, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_phi_d_pi_and_equal_theta_gives_incident_normal(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    theta_h, phi_h, _, _ = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    phi_d = jax.numpy.pi
    theta_d = theta_h

    w_i, _ = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(w_i, normal, atol=ATOL)

@pytest.mark.parametrize("seed", range(N_TEST))
def test_theta_d_theta_h_zero_gives_normal(seed):
    key_rusinkiewicz, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )
    
    _, phi_h, _, phi_d = random_rusinkiewicz(
        key_rusinkiewicz
    )
    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
    
    theta_d = 0.0
    theta_h = 0.0

    w_i, w_o = rusinkiewicz_to_cartesian(
        theta_h,
        phi_h,
        theta_d,
        phi_d,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(w_i, normal, atol=ATOL)
    assert jax.numpy.allclose(w_o, normal, atol=ATOL)
    
    
    
    
    
    
    
    
    
    
    
    
@pytest.mark.parametrize("seed", range(N_TEST))
def test_perfect_reflection_gives_theta_h_zero(seed):
    key_direction, key_normal = jax.random.split(jax.random.key(seed))

    normal = random_unit_vector(key_normal)
    w_i = random_unit_vector(key_direction)

    # Make the direction incident on the surface.
    if jax.numpy.dot(w_i, normal) < 0.0:
        w_i = -w_i

    w_o = -w_i + 2.0 * jax.numpy.dot(w_i, normal) * normal

    theta_h, _, _, _ = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        random_unit_vector(jax.random.key(seed + 100)),
    )

    assert jax.numpy.allclose(
        angular_error(theta_h, 0.0),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_tangent_directions_give_theta_h_absolute_phi_d_half_pi(seed):
    key_i, key_o, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        4,
    )

    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    w_i = random_unit_vector(key_i)
    w_i = w_i - jax.numpy.dot(w_i, normal) * normal
    w_i = w_i / jax.numpy.linalg.norm(w_i)
    w_o = random_unit_vector(key_o)
    w_o = w_o - jax.numpy.dot(w_o, normal) * normal
    w_o = w_o / jax.numpy.linalg.norm(w_o)

    theta_h, _, _, phi_d = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )
    print(jax.numpy.rad2deg(theta_h), jax.numpy.rad2deg(phi_d))
    
    assert jax.numpy.allclose(
        angular_error(theta_h, jax.numpy.pi / 2.0),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_d, jax.numpy.pi / 2.0),
        0.0,
        atol=ATOL,
    ) or jax.numpy.allclose(
        angular_error(phi_d, - jax.numpy.pi / 2.0),
        0.0,
        atol=ATOL,
    )



@pytest.mark.parametrize("seed", range(N_TEST))
def test_exchanging_wi_and_wo_adds_pi_to_phi_d(seed):
    key_i, key_o, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        4,
    )

    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
        
    w_i = random_unit_vector(key_i)
    w_o = random_unit_vector(key_o)
    
    theta_h, phi_h, theta_d, phi_d = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )
    theta_h_test, phi_h_test, theta_d_test, phi_d_test = cartesian_to_rusinkiewicz(
        w_o,
        w_i,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        angular_error(theta_h_test, theta_h),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_h_test, phi_h),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(theta_d_test, theta_d),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_d_test, phi_d + jax.numpy.pi),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_equal_directions_give_theta_d_zero(seed):
    key_i, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)
        
    w_i = random_unit_vector(key_i)
    w_o = w_i
    

    _, _, theta_d, _ = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        angular_error(theta_d, 0.0),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_outgoing_normal_gives_phi_d_zero_theta_equal(seed):
    key_i, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    w_i = random_unit_vector(key_i)
    w_o = normal

    theta_h, _, theta_d, phi_d = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        angular_error(theta_h, theta_d),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_d, 0.0),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_incident_normal_gives_phi_d_pi_theta_equal(seed):
    key_o, key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        3,
    )

    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    w_o = random_unit_vector(key_o)
    w_i = normal

    theta_h, _, theta_d, phi_d = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        angular_error(theta_h, theta_d),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(phi_d, jax.numpy.pi),
        0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_normal_incident_and_outgoing_give_theta_zero(seed):
    key_normal, key_tangent = jax.random.split(
        jax.random.key(seed),
        2,
    )

    normal = random_unit_vector(key_normal)
    tangent = random_unit_vector(key_tangent)

    w_i = normal
    w_o = normal

    theta_h, _, theta_d, _ = cartesian_to_rusinkiewicz(
        w_i,
        w_o,
        normal,
        tangent,
    )

    assert jax.numpy.allclose(
        angular_error(theta_h, 0.0),
        0.0,
        atol=ATOL,
    )
    assert jax.numpy.allclose(
        angular_error(theta_d, 0.0),
        0.0,
        atol=ATOL,
    )