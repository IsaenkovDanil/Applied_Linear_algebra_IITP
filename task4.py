import numpy as np
import math



def forward_substitution(L, c):
    """
    Решает систему Ly = c, где L - нижняя треугольная матрица
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sum_val = L[i, :i] @ y[:i]
        y[i] = (c[i] - sum_val) / L[i, i]
    return y

def backward_substitution(U, y):
    """
    Решает систему Ux = y, где U - верхняя треугольная матрица.
    """
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        sum_val = U[i, i+1:] @ x[i+1:]
        x[i] = (y[i] - sum_val) / U[i, i]
    return x



def cholesky_decomposition(A):
    """
    Выполняет разложение Холецкого A = LL^T для симметричной,
    положительно определенной матрицы A.
    Args:
        A : Квадратная матрица (N x N).

    Returns:
       L  : Нижняя треугольная матрица L (N x N), если A симметрична
         и положительно определена.
        None: Если матрица  не положительно определена.
    """
    n = A.shape[0]




    L = np.zeros_like(A, dtype=np.float64)


    for j in range(n): # Итерация по столбцам j от 0 до n-1
        # Вычисление  L[j, j]
        sum_sq = 0.0
        if j > 0:
            # Сумма квадратов элементов j-й строки L до диагонали
            sum_sq = np.dot(L[j, :j], L[j, :j])

        term_under_sqrt = A[j, j] - sum_sq

        # Проверка на положительную определенность
        if term_under_sqrt <= 1e-12:
            print(f"Ошибка: Матрица не является положительно определенной. ")
            return None

        L[j, j] = math.sqrt(term_under_sqrt)

        # Вычисление  L[i, j] ниже диагонали  i > j
        for i in range(j + 1, n):
            sum_prod = 0.0
            if j > 0:
                #  k от 0 до j-1
                sum_prod = np.dot(L[i, :j], L[j, :j])

            L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    return L




def solve_cholesky(A, b):
    """
    Решает линейную систему Ax = b для симметричной положительно
    определенной матрицы A, используя разложение Холецкого A = LL^T.

    Args:
        A : Матрица коэффициентов (N x N), должна быть СПО.
        b : Вектор правой части (N x 1).

    Returns:
       x : Вектор решения x (N x 1)
        None: Если матрица A не СПО
    """
    print("\n--- Решение системы Ax=b методом Холецкого ---")


    L = cholesky_decomposition(A)

    # Проверяем, успешно ли разложение
    if L is None:
        print("Разложение Холецкого не удалось.")
        return None

    print("Разложение Холецкого L найдено успешно")
    print(" L:\n", np.round(L, 3))

    n = A.shape[0]



    # прямая подстановка
    print("Решаем Ly = b")
    y = forward_substitution(L, b)

    # обратная подстановка
    print("Решаем L^T x = y")
    Lt = L.T
    x_hat = backward_substitution(Lt, y)

    print("Решение x найдено")
    return x_hat


if __name__ == "__main__":

    # 1. Симметричная положительно определенная матрица
    A_spd = np.array([[4., 12., -16.],
                      [12., 37., -43.],
                      [-16., -43., 98.]], dtype=float)
    b_spd = np.array([1., 2., 3.], dtype=float) # Пример вектора b

    # 2. Симметричная, но не положительно определенная
    A_not_pd = np.array([[1., 2., 3.],
                         [2., 1., 2.],
                         [3., 2., 1.]], dtype=float)
    b_not_pd = np.array([1., 1., 1.], dtype=float)



    matrices_to_test = {
        "СПО": (A_spd, b_spd),
        "Симм., не ПО": (A_not_pd, b_not_pd),
    }

    while True:
        mode = input("Выберите режим: 't' - тест примеров, 'm' - ручной ввод: ").lower().strip()
        if mode in ['t', 'm']:
            break


    if mode == 't':
        for name, (matrix, b_vec) in matrices_to_test.items():
            print(f"\n====== Тест: {name} ======")
            print(" A:\n", matrix)
            print(" b:\n", b_vec)
            x_solution = solve_cholesky(matrix, b_vec)

            if x_solution is not None:
                print("\nНайденное решение x_hat:\n", np.round(x_solution, 5))
                # Проверка
                residual = matrix @ x_solution - b_vec
                print("\nПроверка A*x_hat - b:\n", np.round(residual, 9))
                residual_norm = np.linalg.norm(residual)
                print(f"||A*x_hat - b||_2: {residual_norm:.2e}")


                 # Сравнение с numpy
                x_numpy = np.linalg.solve(matrix, b_vec)
                print("\nРешение np.linalg.solve:\n", np.round(x_numpy, 5))
                print("Разница с numpy:", np.linalg.norm(x_solution - x_numpy))


            else:
                 print("Алгоритм Холецкого не применим или матрица не CПО ")

    elif mode == 'm':
        print("\n--- Ввод данных вручную ---")
        A_manual = None
        b_manual = None
        n = 0

        while True:
            n_str = input("Введите размерность матрицы N : ")
            n = int(n_str)
            if n > 0: break

        print(f"\nВведите элементы матрицы A ({n}x{n}), по строкам ")
        A_list = []
        for i in range(n):
            while True:
                row_str = input(f"Строка {i+1} (ровно {n} чисел): ")

                elements = row_str.split()

                if len(elements) == n:

                    row_floats = [float(el) for el in elements]

                    A_list.append(row_floats)

                    break


        A_manual = np.array(A_list, dtype=float)

        print(f"\nВведите элементы вектора b ({n} элементов)")
        while True:
            b_str = input(f"Вектор b (ровно {n} чисел): ")
            elements = b_str.split()
            if len(elements) == n:
                b_manual = np.array([float(el) for el in elements], dtype=float)
                break




        if A_manual is not None and b_manual is not None:

            print("\n------ Расчет для введенных данных ------")
            print(" A:\n", A_manual)
            print(" b:\n", b_manual)

            x_solution_manual = solve_cholesky(A_manual, b_manual)

            if x_solution_manual is not None:

                print("\nНайденное решение x_hat:\n", np.round(x_solution_manual, 5))

                # Проверка
                residual_manual = A_manual @ x_solution_manual - b_manual
                print("\nПроверка A*x_hat - b:\n", np.round(residual_manual, 9))
                residual_norm_manual = np.linalg.norm(residual_manual)
                print(f"||A*x_hat - b||_2: {residual_norm_manual:.2e}")
                 # Сравнение с numpy

                x_numpy_manual = np.linalg.solve(A_manual, b_manual)
                print("\nРешение np.linalg.solve:\n", np.round(x_numpy_manual, 5))
                print("Разница с numpy:", np.linalg.norm(x_solution_manual - x_numpy_manual))

            else:
                print("\nАлгоритм Холецкого не применим или матрица не CПО")

