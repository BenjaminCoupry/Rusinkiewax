import jax

def spherical_to_cartesian(spherical):
    """Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    spherical : array_like, (3, )
        Spherical coordinates ``(rho, theta, phi)``, where ``rho`` is
        the radial distance, ``theta`` is the polar angle measured from
        the positive z-axis, and ``phi`` is the azimuthal angle measured
        from the positive x-axis.

    Returns
    -------
    cartesian : jax.Array, (3, )
        cartesian coordinates ``(x, y, z)`` corresponding to the input
        spherical coordinates.
    """
    rho, theta, phi = jax.numpy.unstack(spherical, axis=-1)
    x = rho * jax.numpy.sin(theta) * jax.numpy.cos(phi)
    y = rho * jax.numpy.sin(theta) * jax.numpy.sin(phi)
    z = rho * jax.numpy.cos(theta)
    cartesian = jax.numpy.stack([x, y, z], axis=-1)
    return cartesian


def cartesian_to_spherical(cartesian):
    """Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cartesian : array_like, (3, )
        cartesian coordinates ``(x, y, z)``.

    Returns
    -------
    spherical : jax.Array, (3, )
        Spherical coordinates ``(rho, theta, phi)``, where ``rho`` is
        the radial distance, ``theta`` is the polar angle measured from
        the positive z-axis, and ``phi`` is the azimuthal angle measured
        from the positive x-axis.

    Notes
    -----
    The azimuthal angle ``phi`` is returned in the range
    ``[-pi, pi]`` due to the use of ``arctan2``. At the origin,
    spherical coordinates are undefined.
    """
    x, y, z = jax.numpy.unstack(cartesian, axis=-1)
    rho = jax.numpy.sqrt(
        jax.numpy.square(x)
        + jax.numpy.square(y)
        + jax.numpy.square(z)
    )
    phi = jax.numpy.arctan2(y, x)
    theta = jax.numpy.arccos(jax.numpy.clip(z / rho, -1.0, 1.0))
    spherical = jax.numpy.stack([rho, theta, phi], axis=-1)
    return spherical


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


def cartesian_to_rusinkiewicz(w_i, w_o, normal, tangent):
    """Convert incident and outgoing directions to Rusinkiewicz coordinates.

    Parameters
    ----------
    w_i : array_like, (3, )
        Incident direction in cartesian coordinates.
    w_o : array_like, (3, )
        Outgoing direction in cartesian coordinates.
    normal : array_like, (3, )
        Surface normal defining the first axis of the local reference frame.
    tangent : array_like, (3, )
        Tangent vector used together with ``normal`` to define the local
        reference frame. The two vectors must be non-collinear.

    Returns
    -------
    theta_h : jax.Array, (, )
        Polar angle of the half-vector.
    phi_h : jax.Array, (, )
        Azimuthal angle of the half-vector.
    theta_d : jax.Array, (, )
        Polar angle of the incident direction expressed in the half-vector
        local frame.
    phi_d : jax.Array, (, )
        Azimuthal angle of the incident direction expressed in the
        half-vector local frame.

    Notes
    -----
    The half-vector is defined as the normalized sum of ``w_i`` and
    ``w_o``. Therefore, the conversion is undefined when
    ``w_i = -w_o``.
    """
    local_base = jax.numpy.roll(
        complete_base(
            jax.numpy.stack([normal, tangent], axis=-1)
        ),
        -1,
        axis=-1,
    )
    half_vector = (
        (w_i + w_o)
        / jax.numpy.linalg.vector_norm(
            w_i + w_o,
            axis=-1,
            keepdims=True,
        )
    )
    _, theta_h, phi_h = jax.numpy.unstack(
        cartesian_to_spherical(local_base.T @ half_vector),
        axis=-1,
    )
    half_base = local_base @ jax.numpy.roll(
        spherical_local_base(
            jax.numpy.stack([1.0, theta_h, phi_h])
        ),
        -1,
        axis=-1,
    )
    _, theta_d, phi_d = jax.numpy.unstack(
        cartesian_to_spherical(half_base.T @ w_i),
        axis=-1,
    )
    return theta_h, phi_h, theta_d, phi_d


def rusinkiewicz_to_cartesian(
    theta_h,
    phi_h,
    theta_d,
    phi_d,
    normal,
    tangent,
):
    """Convert Rusinkiewicz coordinates to incident and outgoing directions.

    Parameters
    ----------
    theta_h : array_like, (, )
        Polar angle of the half-vector, in the range ``[0, pi / 2]``.
    phi_h : array_like, (, )
        Azimuthal angle of the half-vector, in the range
        ``[0, 2 * pi]``.
    theta_d : array_like, (, )
        Polar angle of the incident direction in the half-vector local
        frame, in the range ``[0, pi / 2]``.
    phi_d : array_like, (, )
        Azimuthal angle of the incident direction in the half-vector
        local frame, in the range ``[0, 2 * pi]``.
    normal : array_like, (3, )
        Surface normal defining the first axis of the local reference frame.
    tangent : array_like, (3, )
        Tangent vector used together with ``normal`` to define the local
        reference frame. The two vectors must be non-collinear.

    Returns
    -------
    w_i : jax.Array, (3, )
        Incident direction in cartesian coordinates.
    w_o : jax.Array, (3, )
        Outgoing direction in cartesian coordinates.

    Notes
    -----
    The outgoing direction is constructed by rotating the incident
    direction by ``pi`` in the azimuthal coordinate of the half-vector
    local frame.
    """
    local_base = jax.numpy.roll(
        complete_base(
            jax.numpy.stack([normal, tangent], axis=-1)
        ),
        -1,
        axis=-1,
    )
    half_base = local_base @ jax.numpy.roll(
        spherical_local_base(
            jax.numpy.stack([1.0, theta_h, phi_h])
        ),
        -1,
        axis=-1,
    )

    w_i = half_base @ spherical_to_cartesian(
        jax.numpy.stack([1.0, theta_d, phi_d], axis=-1)
    )
    w_o = half_base @ spherical_to_cartesian(
        jax.numpy.stack(
            [1.0, theta_d, phi_d + jax.numpy.pi],
            axis=-1,
        )
    )

    return w_i, w_o

    