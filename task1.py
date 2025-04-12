
import numpy as np


def lu_decomposition_pivot(A):
    """
    Выполняет PA = LU разложение матрицы A с выбором ведущего элемента по столбцу
    Args:
        A : Квадратная матрица  размера N x N
    Returns:
        tuple: Кортеж (P, L, U), где:
            P : Матрица перестановок (N x N).
            L : Нижняя треугольная матрица с единицами на диагонали (N x N).
            U : Верхняя треугольная матрица (N x N).
    """
    n = A.shape[0]
    U = A.astype(np.float64).copy()

    L = np.eye(n, dtype=np.float64)

    P = np.eye(n, dtype=np.float64)


    for k in range(n - 1):

        # выбор ведущего элемента
        # Ищем в k-м столбце ,начиная с k-й строки, элемент с максимальным  значением
        pivot_row_local = np.argmax(np.abs(U[k:, k]))
        pivot_row_global = k + pivot_row_local


        if pivot_row_global != k:

            U[[k, pivot_row_global], :] = U[[pivot_row_global, k], :]

            P[[k, pivot_row_global], :] = P[[pivot_row_global, k], :]


            L[[k, pivot_row_global], :k] = L[[pivot_row_global, k], :k]


        # Обнуление элементов под диагональю
        #  по всем строкам i ниже строки k (от k+1 до n-1)
        for i in range(k + 1, n):
            if U[k, k] == 0:
                continue


            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            U[i, k:] -= multiplier * U[k, k:]
            U[i, k] = 0


    return P, L, U



def forward_substitution(L, c):
    """
    Решает систему Ly = c, где L - нижняя треугольная матрица (с 1 на диагонали).
    Args:
        L : Нижняя треугольная матрица (N x N).
        c : Вектор правой части (N x 1).
    Returns:
       y : Вектор решения y (N x 1).
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)

    # Проходим по строкам от 0 до n-1
    for i in range(n):

        sum_val = L[i, :i] @ y[:i]
        y[i] = (c[i] - sum_val) / L[i, i]

    return y


def backward_substitution(U, y):
    """
    Решает систему Ux = y, где U - верхняя треугольная матрица.
    Args:
        U : Верхняя треугольная матрица (N x N)
        y : Вектор правой части (N x 1),  из forward_substitution
    Returns:
       x : Вектор решения x (N x 1).


    """
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)

    # Проходим по строкам в обратном порядке: от n-1 до 0

    for i in range(n - 1, -1, -1):

        sum_val = U[i, i+1:] @ x[i+1:]
        x[i] = (y[i] - sum_val) / U[i, i]

    return x



def solve_linear_system_lu(A, b):
    """
    Решает линейную систему Ax = b с использованием PA=LU разложения
    Args:
        A : Матрица коэффициентов (N x N)
        b : Вектор правой части (N x 1)

    Returns:
       x_hat  np.ndarray: Вектор решения x_hat (N x 1)
      (P, L, U) из разложения

    """
    P, L, U = lu_decomposition_pivot(A)
    #  c = Pb.
    c = P @ b
    y = forward_substitution(L, c)
    x_hat = backward_substitution(U, y)

    return x_hat, (P, L, U)




if __name__ == "__main__":

    while True:
        choice = input("Выберите режим: 'e' -  пример, 'm' - ручной ввод: ").lower().strip()
        if choice in ['e', 'm']:
            break


    A_matrix = None
    b_vector = None
    n = 0

    if choice == 'e':
        print("\nИспользуется встроенный пример ")
        A_matrix = np.array([[0.0, 1.0, 1.0],
                             [2.0, 1.0, -1.0],
                             [-1.0, 1.0, -2.0]], dtype=float)
        b_vector = np.array([2.0, 1.0, -5.0], dtype=float)
        n = A_matrix.shape[0]

    elif choice == 'm':
        print("\n--- Ввод данных вручную ---")

        while True:
            n_str = input("Введите размерность матрицы N : ")
            n = int(n_str)
            if n > 0:
                break



            print(f"\nВведите элементы матрицы A ({n}x{n}), по строкам")
            A_list = []
            for i in range(n):
                while True:
                    row_str = input(f"Строка {i+1} (ровно {n} чисел): ")
                    elements = row_str.split()
                    if len(elements) == n:
                        row_floats = [float(el) for el in elements]
                        A_list.append(row_floats)
                        break

            A_matrix = np.array(A_list, dtype=float)

            print(f"\nВведите элементы вектора b ({n} элементов)")
            while True:
                b_str = input(f"Вектор b (ровно {n} чисел): ")
                elements = b_str.split()
                if len(elements) == n:
                    b_vector = np.array([float(el) for el in elements], dtype=float)
                    break


    print("Матрица A:\n", np.round(A_matrix, 5))
    print("\nВектор b:\n", np.round(b_vector, 5))

    solution, matrices = solve_linear_system_lu(A_matrix, b_vector)

    if solution is not None:

        P_mat, L_mat, U_mat = matrices

        print("\n--- Результат разложения PA = LU ---")

        print(" P:\n", np.round(P_mat, 3))
        print("\n L:\n", np.round(L_mat, 3))
        print("\n U:\n", np.round(U_mat, 3))

        #Проверка разложения
        PA_check = P_mat @ A_matrix
        LU_check = L_mat @ U_mat
        print("\n--- Проверка разложения  ---")
        print("PA:\n", np.round(PA_check, 3))
        print("LU:\n", np.round(LU_check, 3))

        diff_norm = np.linalg.norm(PA_check - LU_check, ord=np.inf)
        print(f"||PA - LU||_inf: {diff_norm:.2e}")

        print("\n--- Решение системы Ax = b ---")
        print("Решение x_hat:\n", np.round(solution, 3))

        # Проверка решения
        Ax_check = A_matrix @ solution
        #  r = A*x_hat - b
        residual_vector = Ax_check - b_vector
        print("\n--- Проверка решения  ---")
        print("Вектор невязки A*x_hat - b:\n", np.round(residual_vector, 9))
        residual_norm = np.linalg.norm(residual_vector)
        print(f"||A*x_hat - b||_2: {residual_norm:.2e}")

        print("\n--- Сравнение с np.linalg.solve ---")
        numpy_solution = np.linalg.solve(A_matrix, b_vector)
        print("Решение, найденное np.linalg.solve:\n", np.round(numpy_solution, 3))
        solution_diff_norm = np.linalg.norm(solution - numpy_solution)
        print(f"||x_hat - x_numpy||_2: {solution_diff_norm:.2e}")


