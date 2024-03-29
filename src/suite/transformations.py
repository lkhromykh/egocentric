import numpy as np

cos = np.cos
sin = np.sin
_eps = 1e-6
_POLE_LIMIT = 1 - _eps
_RPOLE_LIMIT = np.pi / 2 * _POLE_LIMIT


def rv2rpy(rot_vec):
    # This transformation will require to alter RTDE pose: polyscope use different solution.
    theta = np.linalg.norm(rot_vec)
    if theta == 0:
        return rot_vec
    kx, ky, kz = rot_vec / theta
    cth = cos(theta)
    sth = sin(theta)
    vth = 1 - cth

    r11 = kx * kx * vth + cth
    r12 = kx * ky * vth - kz * sth
    # r13 = kx * kz * vth + ky * sth
    r21 = kx * ky * vth + kz * sth
    r22 = ky * ky * vth + cth
    # r23 = ky * kz * vth - kx * sth
    r31 = kx * kz * vth - ky * sth
    r32 = ky * kz * vth + kx * sth
    r33 = kz * kz * vth + cth

    beta = np.arctan2(-r31, np.sqrt(r11 * r11 + r21 * r21))
    if beta > _RPOLE_LIMIT:
        alpha = 0
        gamma = np.arctan2(r12, r22)
    elif beta < -_RPOLE_LIMIT:
        alpha = 0
        gamma = -np.arctan2(r12, r22)
    else:
        cb = cos(beta)
        alpha = np.arctan2(r21 / cb, r11 / cb)
        gamma = np.arctan2(r32 / cb, r33 / cb)
    return np.array([gamma, beta, alpha])


def rmat2euler(mat):
    if mat.shape == (9,):
        mat = mat.reshape((3, 3))

    if mat[0, 2] > _POLE_LIMIT:
        z = np.arctan2(mat[1, 0], mat[1, 1])
        y = np.pi / 2
        x = 0.0
        return np.array([x, y, z])

    if mat[0, 2] < -_POLE_LIMIT:
        z = np.arctan2(mat[1, 0], mat[1, 1])
        y = -np.pi / 2
        x = 0.0
        return np.array([x, y, z])

    z = -np.arctan2(mat[0, 1], mat[0, 0])
    y = np.arcsin(mat[0, 2])
    x = -np.arctan2(mat[1, 2], mat[2, 2])
    return np.array([x, y, z])


def rotm2axang1(mat):
    tr = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if np.abs(tr - 3) < _eps:
        return np.array([0, 0, 0.])
    if np.abs(tr + 1) < _eps:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            u = np.array([mat[0, 0] + 1, mat[0, 1], mat[0, 2]])
        elif mat[1, 1] > mat[2, 2]:
            u = np.array([mat[1, 0], mat[1, 1] + 1, mat[1, 2]])
        else:
            u = np.array([mat[2, 0], mat[2, 1], mat[2, 2] + 1])
        u /= np.linalg.norm(u)
        return np.pi * u

    theta = np.arccos((tr - 1) / 2)
    r = np.array([
        mat[2, 1] - mat[1, 2],
        mat[0, 2] - mat[2, 0],
        mat[1, 0] - mat[0, 1]
    ])
    r /= 2 * np.sin(theta)
    return theta * r


def rotm2axang2(mat):
    r = np.array([
        mat[2, 1] - mat[1, 2],
        mat[0, 2] - mat[2, 0],
        mat[1, 0] - mat[0, 1]
    ])
    s = np.linalg.norm(r)
    c = (mat[0, 0] + mat[1, 1] + mat[2, 2] - 1) / 2
    if s < _eps and np.abs(c - 1) < _eps:
        return np.array([0, 0, 0.])
    if s < _eps and np.abs(c + 1) < _eps:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            u = np.array([mat[0, 0] + 1, mat[0, 1], mat[0, 2]])
        elif mat[1, 1] > mat[2, 2]:
            u = np.array([mat[1, 0], mat[1, 1] + 1, mat[1, 2]])
        else:
            u = np.array([mat[2, 0], mat[2, 1], mat[2, 2] + 1])
        u /= np.linalg.norm(u)
        return np.pi * u
    theta = np.arctan2(s, c)
    if np.abs(np.sin(theta)) < _eps:
        return np.zeros(3)
    return theta * r / s
