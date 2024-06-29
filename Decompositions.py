import math
import numpy

class Helper:
    def __init__(self):
        pass
    def identity_matrix(self, n):
        return [[1 if i == j else 0 for i in range(n)] for j in range(n)]
    def transpose(self, A):
        temp = []
        for i in range(len(A[0])):
            temp.append([])
            for j in range(len(A)):
                temp[i].append(A[j][i])
        return temp
    def matrix_mult(self, A, B):
        temp = []
        for i in range(len(A)):
            temp.append([])
            for j in range(len(B[0])):
                temp[i].append(0)
                for k in range(len(A[0])):
                    temp[i][j] +=  A[i][k] * B[k][j]
        return temp
    def dot_product(self, v1, v2):
        ans = 0
        for i in range(len(v1)):
            ans += v1[i] * v2[i]
        return ans
    def vector_sub(self, v1, v2):
        temp = []
        for i in range(len(v1)):
            temp.append(v1[i] - v2[i])
        return temp
    def vector_mag(self, v1):
        ans = 0
        for i in range(len(v1)):
            ans += (v1[i] ** 2)
        return math.sqrt(ans)
    def vector_div(self, v1, num):
        for i in range(len(v1)):
            v1[i] /= num
        return v1
    def vector_mult(self, v1, num):
        v1 = v1.copy()
        for i in range(len(v1)):
            v1[i] *= num
        return v1
    def eigen(self, a):
        e_val, e_vec = numpy.linalg.eig(a)
        return e_val, e_vec
    def covariance(self, a, b):
        a_mean, b_mean = sum(a) / len(a), sum(b) / len(b)
        n = len(a)
        cov = 0
        for i in range(len(a)):
            cov += (a[i] - a_mean) * (b[i] - b_mean)
        return cov / (n-1)
    def covariance_matrix(self, a):
        col_vectors = self.transpose(a)
        cov_matrix = []
        for i in range(len(col_vectors)):
            cov_matrix.append([])
            for j in range(len(col_vectors)):
                cov_matrix[i].append(self.covariance(col_vectors[i], col_vectors[j]))
        return cov_matrix

class Decompositions:
    def __init__(self):
        self.help = Helper()
    def lu_decomposition(self, A):
        help = self.help
        U = A.copy()
        N = len(U)
        L = help.identity_matrix(N)
        for i in range(N):
            for j in range(i+1, N):
                L[j][i] = U[j][i] / U[i][i]
                for k in range(N):
                    U[j][k] -= U[i][k] * L[j][i]
        return [ L, U ]
    def qr_decomposition(self, A):
        help = self.help
        dup = A.copy()
        dup = help.transpose(dup)
        Q_t = [dup[0]]
        for i in range(1,len(dup)):
            q = dup[i]
            for j in range(i):
                proj_magnitude = help.dot_product(dup[i], Q_t[j]) / help.vector_mag(Q_t[j])
                proj_vector = help.vector_mult(Q_t[j], (proj_magnitude / help.vector_mag(Q_t[j])))
                q = help.vector_sub(q, proj_vector)
            Q_t.append(help.vector_div( q, help.vector_mag(q) ))
        R = help.matrix_mult(Q_t, A)
        Q = help.transpose(Q_t)
        return [ Q, R ]
    def svd_decomposition(self, A):
        help = self.help
        m1, m2 = A.copy(), A.copy()
        m1 = help.matrix_mult(help.transpose(m1), m1)
        m2 = help.matrix_mult(m2, help.transpose(m2))
        singular_values = set()
        temp, u = help.eigen(m1)
        singular_values.add(temp)
        temp, v = help.eigen(m2)
        singular_values.add(temp)
        return(u, v)



A_LU = [[2, 4, 3, 5], [-4, -7, -5, -8], [6, 8, 2, 9], [4, 9, -2, 14]]
A_QR = [[1,2,4],[0,0,5],[0,3,6]]
A_SVD = []
A_COV = [[1,2,3],[4,5,6],[7,8,9]]

D = Decompositions()
H = Helper()

ans = D.lu_decomposition(A_LU)

print("\nLU Decomposition")
print('-----' * len(ans[0]))
for matrix in ans:
    for row in matrix:
        print(" ", " ".join(f"{num:4}" for num in row))
    print('-----' * len(matrix))

ans = D.qr_decomposition(A_QR)

print("\n\nQR Decomposition")
print('-----' * len(ans[0]))
for matrix in ans:
    for row in matrix:
        print(" ", " ".join(f"{num:6.2f}" for num in row))
    print('-----' * len(matrix))

# ans = D.svd_decomposition(A_SVD)

# print("\n\nSVD Decomposition")
# print('-----' * len(ans[0]))
# for matrix in ans:
#     for row in matrix:
#         print(" ", " ".join(f"{num:6.2f}" for num in row))
#     print('-----' * len(matrix))

ans = H.covariance_matrix(A_COV)

print("\n\nCovariance Matrix")
print('-----' * len(ans[0]))
for row in ans:
    print(" ", " ".join(f"{num:6.2f}" for num in row))
print('-----' * len(ans))
