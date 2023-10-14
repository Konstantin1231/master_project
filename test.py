import numpy as np

def random_orthonormal_basis(n):
    # Step 1: Generate a random n x n matrix
    A = np.random.rand(n, n)

    # Step 2: Use QR decomposition to get an orthonormal basis
    Q, R = np.linalg.qr(A)

    return Q

# Specify the dimension
n = 3  # for R^3

# Generate the orthonormal basis
basis = random_orthonormal_basis(n)
print(basis)