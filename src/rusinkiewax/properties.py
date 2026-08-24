import jax

def helmholtz_symmetry(theta_h, phi_h, theta_d, phi_d):
    """Fold the Rusinkiewicz difference azimuth into its Helmholtz-symmetric range.

    Exploits the Helmholtz reciprocity of the BRDF, which under the
    Rusinkiewicz parametrization is equivalent to invariance under
    ``phi_d -> phi_d + pi (mod 2 * pi)``. As a consequence, ``phi_d`` can be
    folded from its native range ``[0, 2 * pi)`` into ``[0, pi)`` without
    loss of information about the BRDF value.

    Parameters
    ----------
    theta_h : array_like, (, )
        Polar angle of the half-vector. Passed through unchanged.
    phi_h : array_like, (, )
        Azimuthal angle of the half-vector. Passed through unchanged.
    theta_d : array_like, (, )
        Polar angle of the incident direction in the half-vector local
        frame. Passed through unchanged.
    phi_d : array_like, (, )
        Azimuthal angle of the incident direction in the half-vector
        local frame, in the range ``[0, 2 * pi)``.

    Returns
    -------
    theta_h : jax.Array, (, )
    phi_h : jax.Array, (, )
    theta_d : jax.Array, (, )
    phi_d : jax.Array, (, )
        Folded azimuthal angle, in the range ``[0, pi)``.
    """
    phi_d = jax.numpy.mod(phi_d, jax.numpy.pi)
    return theta_h, phi_h, theta_d, phi_d

def isotropic_material(theta_h, phi_h, theta_d, phi_d):
    """Drop the dependence on the half-vector azimuth for isotropic materials.

    For an isotropic material, the BRDF is invariant under rotation about
    the surface normal, so it does not depend on the absolute azimuthal
    orientation of the half-vector. ``phi_h`` can therefore be set to zero
    without loss of information about the BRDF value.

    Parameters
    ----------
    theta_h : array_like, (, )
        Polar angle of the half-vector. Passed through unchanged.
    phi_h : array_like, (, )
        Azimuthal angle of the half-vector. Discarded and replaced by zero.
    theta_d : array_like, (, )
        Polar angle of the incident direction in the half-vector local
        frame. Passed through unchanged.
    phi_d : array_like, (, )
        Azimuthal angle of the incident direction in the half-vector
        local frame. Passed through unchanged.

    Returns
    -------
    theta_h : jax.Array, (, )
    phi_h : jax.Array, (, )
        Azimuthal angle of the half-vector, set to zero.
    theta_d : jax.Array, (, )
    phi_d : jax.Array, (, )
    """
    phi_h = 0.0
    return theta_h, phi_h, theta_d, phi_d