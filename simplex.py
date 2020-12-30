import numpy as np


def simplex(A: np.ndarray, b: np.ndarray, c: np.ndarray, B_index):
    m, n = A.shape
    B, c_B = A[:, B_index], c[B_index]
    N_index = [i for i in range(n) if i not in B_index]
    N, c_N = A[:, N_index], c[N_index]
    x_B = np.linalg.solve(B, b)
    while True:
        print("objective %f" % c_B.dot(x_B))
        l = np.linalg.solve(B.T, c_B)   # linear system 1
        s_N = c_N - N.T.dot(l)          # pricing
        q_list = [i for i in range(n - m) if s_N[i] < 0]
        if len(q_list) == 0:
            x = np.zeros(n)
            for i in range(m):
                x[B_index[i]] = x_B[i]
            print("optimal point found, objective %f" % c.dot(x))
            return x
        q_ = q_list[0]
        q = N_index[q_]                 # input base variable from N
        A_q = A[:, q]
        d = np.linalg.solve(B, A_q)     # linear system 2
        x_q_plus = np.finfo(np.float).max
        for i in range(m):
            if d[i] > 0:
                if x_B[i] / d[i] < x_q_plus:
                    x_q_plus = x_B[i] / d[i]
                    p_ = i
        if x_q_plus == np.finfo(np.float).max:
            print("problem is unbounded")
            return None
        p = B_index[p_]                 # output base variable from B
        x_B = x_B - d * x_q_plus
        x_B[p_], B_index[p_], B[:, p_], c_B[p_] = x_q_plus, q, A[:, q], c[q]
        N_index[q_], N[:, q_], c_N[q_] = p, A[:, p], c[p]


if __name__ == "__main__":
    A = np.array([[1., 1., 1., 0.], [2., 0.5, 0., 1.]])
    b = np.array([5., 8.])
    c = np.array([-3., -2., 0., 0.])
    B_index = [2, 3]
    print(simplex(A, b, c, B_index))
