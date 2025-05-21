import numpy as np


def solve_triangular_system(T, b, lower=False):
    """
    Решает систему Tx = b, где T - треугольная матрица 
    Args:
        T : Треугольная матрица (N x N).
        b : Вектор правой части (N x 1).
        lower : True, если T - нижняя треугольная, False - если верхняя.
    """
    n = T.shape[0]


    x = np.zeros(n, dtype=T.dtype) 

    if lower: # Прямая подстановка
        for i in range(n):
            sum_val = T[i, :i] @ x[:i]

            x[i] = (b[i] - sum_val) / T[i, i]
    else: # Обратная подстановка
        for i in range(n - 1, -1, -1):
            sum_val = T[i, i+1:] @ x[i+1:]


            x[i] = (b[i] - sum_val) / T[i, i]
    return x

def inverse_power_iteration_shifted(A, mu, max_iters=1000, tol=1e-9):
    """
    Находит собственный вектор, соответствующий собственному значению,
    ближайшему к сдвигу mu, для матрицы A.
    Args:
        A : Квадратная матрица N x N.
        mu : Сдвиг.
        max_iters : Максимальное количество итераций.
        tol :  для проверки сходимости.
    """
    n = A.shape[0]



    A_shifted = A - mu * np.eye(n, dtype=A.dtype)


    if np.iscomplexobj(A) or np.iscomplexobj(mu):
        x_k = np.random.rand(n) + 1j * np.random.rand(n)
    else:
        x_k = np.random.rand(n)
    x_k = x_k / np.linalg.norm(x_k) 


    is_lower = np.allclose(A_shifted, np.tril(A_shifted))
    is_upper = np.allclose(A_shifted, np.triu(A_shifted)) 


    for iteration in range(max_iters):

        # Решаем систему (A - mu*I) w_k = x_k
        if is_lower:
            w_k = solve_triangular_system(A_shifted, x_k, lower=True)
        elif is_upper:
            w_k = solve_triangular_system(A_shifted, x_k, lower=False)



        norm_w_k = np.linalg.norm(w_k)


        x_next = w_k / norm_w_k


        if np.linalg.norm(x_next - x_k) < tol or np.linalg.norm(x_next + x_k) < tol:
            return x_next
        x_k = x_next

    return x_k 

def find_eigenvectors_for_triangular(A, use_complex_shifts=False):
    """
    Находит собственные векторы для треугольной матрицы A,
    использует метод обратных степенных итераций со сдвигом.
    Args:
        A : Треугольная матрица N x N.
        use_complex_shifts : есть ли комплексные сдвиги 
    """
    n = A.shape[0]
    eigenvectors = {}
    eigenvalues = np.diag(A) 

    print(f"\nПоиск собственных векторов для матрицы A")


    for i in range(n):
        lambda_val = eigenvalues[i]
        print(f"\nПоиск собственного вектора для lambda_{i+1} = {lambda_val}")


        epsilon = 1e-8
        if use_complex_shifts and (np.iscomplexobj(lambda_val) or np.iscomplexobj(A)):
            mu = lambda_val - epsilon * (1 + 1j) #  комплексный сдвиг
        else:
            mu = lambda_val - epsilon #  действительный сдвиг

        print(f"  Сдвиг mu = {mu}")
        eigenvector = inverse_power_iteration_shifted(A, mu)

        if eigenvector is not None:
            eigenvectors[lambda_val] = eigenvector
            print(f"  Найденный собственный вектор (нормированный):\n  {np.round(eigenvector, 4) if not np.iscomplexobj(eigenvector) else eigenvector}")
            # Проверка: A*v - lambda*v 
            check_vector = A @ eigenvector - lambda_val * eigenvector
            norm_check = np.linalg.norm(check_vector)
            print(f"  Проверка ||A*v - lambda*v||_2: {norm_check:.2e}")



    return eigenvectors

if __name__ == "__main__":

    print("\nДействительная треугольная матрица ")
    A_real = np.array([[5., 2., 3.],
                       [0., 8., 1.],
                       [0., 0., 2.]], dtype=float)
    # Собственные значения: 5, 8, 2
    print("Матрица A_real:\n", A_real)
    eigenvectors_real = find_eigenvectors_for_triangular(A_real)

    #  Треугольная матрица с комплексными диагональными элементами 
    print("\n\n Комплексная треугольная матрица")
    A_complex = np.array([[1. + 2.j, 0.5 - 0.1j, 1.2 + 0.j],
                          [0. + 0.j, 3. - 1.j,  2.5 + 0.3j],
                          [0. + 0.j, 0. + 0.j,   -0.5 + 4.j]], dtype=complex)
    # Собственные значения: 1+2j, 3-1j, -0.5+4j
    print("Матрица A_complex:\n", A_complex)
    eigenvectors_complex = find_eigenvectors_for_triangular(A_complex, use_complex_shifts=True)

