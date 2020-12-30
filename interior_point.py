import numpy as np


def find_step_length(x: np.ndarray, s: np.ndarray, d_x: np.ndarray, d_s: np.ndarray):
    n = x.shape[0]
    eta = 0.9
    alpha_pri_max, alpha_dual_max = np.finfo(np.float).max, np.finfo(np.float).max
    for i in range(n):
        if d_x[i] < 0:
            alpha_pri_max = min(alpha_pri_max, -x[i] / d_x[i])
        if d_s[i] < 0:
            alpha_dual_max = min(alpha_dual_max, -s[i] / d_s[i])
    alpha_pri, alpha_dual = min(1, eta * alpha_pri_max), min(1, eta * alpha_dual_max)
    return alpha_pri, alpha_dual


def interior_point(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                   x_0: np.ndarray, l_0: np.ndarray, s_0: np.ndarray):
    m, n = A.shape
    x, l, s = x_0, l_0, s_0
    obj = c.dot(x)
    while True:
        print("objective %f, residual %f" %
              (c.dot(x), np.linalg.norm(A.dot(x) - b, np.inf)))
        X, S = np.diag(x), np.diag(s)
        J = np.block([
            [np.zeros((n, n)), A.T, np.eye(n)],
            [A, np.zeros((m, m)), np.zeros((m, n))],
            [S, np.zeros((n, m)), X]
        ])
        r_b = A.dot(x) - b
        r_c = A.T.dot(l) + s - c
        F = -np.concatenate([r_c, r_b, x * s])
        d = np.linalg.solve(J, F)
        d_x, d_l, d_s = d[:n], d[n: n + m], d[n + m:]
        alpha_pri, alpha_dual = find_step_length(x, s, d_x, d_s)
        x, l, s = x + alpha_pri * d_x, l + alpha_dual * d_l, s + alpha_dual * d_s
        if np.abs(obj - c.dot(x)) < 1e-8:
            print("optimal point found, objective %f" % c.dot(x))
            return x
        else:
            obj = c.dot(x)


def find_starting_point(A: np.ndarray, b: np.ndarray, c: np.ndarray):
    AAT = np.linalg.inv(A.dot(A.T))
    x_ = A.T.dot(AAT).dot(b)
    l_ = AAT.dot(A).dot(c)
    s_ = c - A.T.dot(l_)
    x_ = x_ + max(-1.5 * min(x_), 0)
    s_ = s_ + max(-1.5 * min(s_), 0)
    x = x_ + 0.5 * x_.dot(s_) / sum(s_)
    s = s_ + 0.5 * x_.dot(s_) / sum(x_)
    return x, l_, s


if __name__ == "__main__":
    A = np.array([[1., 1., 1., 0.], [2., 0.5, 0., 1.]])
    b = np.array([5., 8.])
    c = np.array([-3., -2., 0., 0.])
    x_0, l_0, s_0 = find_starting_point(A, b, c)
    print(interior_point(A, b, c, x_0, l_0, s_0))