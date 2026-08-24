import jax

def spherical_local_base(spherical):
    """Compute the local orthonormal basis associated with spherical coordinates.

    Parameters
    ----------
    spherical : array_like, (3, )
        Spherical coordinates ``(rho, theta, phi)``. The radial coordinate
        ``rho`` does not affect the resulting basis.

    Returns
    -------
    orthonormal_base : jax.Array, (3, 3)
        Local orthonormal basis whose columns are, in order,

        - ``u_rho``: radial direction,
        - ``u_theta``: direction of increasing polar angle ``theta``,
        - ``u_phi``: direction of increasing azimuthal angle ``phi``.

        The basis is expressed in cartesian coordinates.
    """
    _, theta, phi = jax.numpy.unstack(spherical, axis=-1)
    u_rho = jax.numpy.asarray([
        jax.numpy.sin(theta) * jax.numpy.cos(phi),
        jax.numpy.sin(theta) * jax.numpy.sin(phi),
        jax.numpy.cos(theta),
    ])
    u_theta = jax.numpy.asarray([
        jax.numpy.cos(theta) * jax.numpy.cos(phi),
        jax.numpy.cos(theta) * jax.numpy.sin(phi),
        -jax.numpy.sin(theta),
    ])
    u_phi = jax.numpy.asarray([
        -jax.numpy.sin(phi),
        jax.numpy.cos(phi),
        0,
    ])
    orthonormal_base = jax.numpy.stack(
        [u_rho, u_theta, u_phi],
        axis=-1,
    )
    return orthonormal_base


def orthogonal(vector):
    """Compute a vector orthogonal to the input.

    Parameters
    ----------
    vector : array_like, (3, )
        Input vector. Must be non-zero.

    Returns
    -------
    orthogonal_vector : jax.Array, (3, )
        A vector orthogonal to ``vector``. Not normalized.

    Notes
    -----
    The orthogonal vector is obtained as the cross product of ``vector``
    with the standard basis vector corresponding to the smallest-magnitude
    component of ``vector``, which avoids the degenerate case of a
    vanishing cross product when ``vector`` is aligned with an arbitrarily
    chosen fixed axis.
    """
    index = jax.numpy.argmin(jax.numpy.abs(vector), axis=-1)
    basis = jax.numpy.eye(3)[index]
    orthogonal_vector = jax.numpy.cross(vector, basis)
    return orthogonal_vector


def complete_base(partial_base):
    """Complete a pair of non-collinear vectors into an orthonormal basis.

    The first vector is preserved as the first basis vector. The second
    vector is orthogonalized with respect to the first one, and the third
    vector is obtained from their cross product.

    Parameters
    ----------
    partial_base : array_like, (3, 2)
        Two non-collinear vectors stored as columns.

    Returns
    -------
    orthonormal_base : jax.Array, (3, 3)
        Orthonormal basis whose first column is the normalized first input
        vector, whose second column is the orthogonalized and normalized
        second input vector, and whose third column is their cross product.

    Notes
    -----
    The input vectors must be non-zero and non-collinear.
    """
    u, v = jax.numpy.unstack(partial_base, axis=-1)
    w = jax.numpy.cross(u, v)
    v = jax.numpy.cross(w, u)
    orthogonal_base = jax.numpy.stack([u, v, w], axis=-1)
    orthonormal_base = (
        orthogonal_base
        / jax.numpy.linalg.vector_norm(
            orthogonal_base,
            axis=-2,
            keepdims=True,
        )
    )
    return orthonormal_base