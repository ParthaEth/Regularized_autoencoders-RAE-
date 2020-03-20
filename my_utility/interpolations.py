from __future__ import division
import numpy as np


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def slerpolate(x, y, C, num_pts):
    alphas = np.linspace(0, 1.0, num_pts)
    if C is not None:
        a = np.linalg.cholesky(C)
        a_inv = np.linalg.inv(a)
        x_inv = np.dot(a_inv, x)
        y_inv = np.dot(a_inv, y)
        res_inv = [slerp(alpha, x_inv, y_inv) for alpha in alphas]
        res = np.dot(a, np.array(res_inv).T)
        # res = np.transpose(res_inv)
        return res
    else:
        return np.array([slerp(alpha, x, y) for alpha in alphas]).T


if __name__ == "__main__":
    x = np.array([10, 0])
    y = np.array([0, 1])
    # C = [[1, 0], [0, 1]]
    a = [[1, 0], [0, 1]]
    C = np.matmul(a, a)
    zs = slerpolate(x, y, C, 11)

    zs_normal = slerpolate(x, y, None, 11)

    np.testing.assert_almost_equal(zs, zs_normal)

    # plt.figure()
    # plt.plot(*x, color='r', marker='*')
    # plt.plot(*y, color='g', marker='*')
    # plt.plot(*zs, color='b', marker='o', alpha=0.2)
    # plt.axis('equal')
    # plt.show()
    # plt.grid(True)
    # plt.close()