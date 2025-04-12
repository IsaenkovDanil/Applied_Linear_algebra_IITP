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


def invert_matrix_lu(A):
    """
    Находит обратную матрицу A^{-1} с использованием PA=LU разложения.
    Args:
        A : Квадратная матрица (N x N), для которой ищется обратная.
    Returns:
        A_inv : Обратная матрица A^{-1} (N x N), если A невырождена.
    """
    n = A.shape[0]


    P, L, U = lu_decomposition_pivot(A)
    I = np.eye(n, dtype=np.float64)
    A_inv = np.zeros_like(A, dtype=np.float64)

    # Решаем n систем Ax_i = e_i
    for i in range(n):
        e_i = I[:, i]

        # PA x_i = P e_i
        c_i = P @ e_i
        #  Ly_i = c_i (прямая подстановка)
        y_i = forward_substitution(L, c_i)
        #  Ux_i = y_i (обратная подстановка)
        x_i = backward_substitution(U, y_i)

        #  x_i - это i-й столбец обратной матрицы
        A_inv[:, i] = x_i

    return A_inv




if __name__ == "__main__":

    while True:
        choice = input("Выберите режим: 'e' -  пример, 'm' - ручной ввод: ").lower().strip()
        if choice in ['e', 'm']:
            break


    A_matrix = None
    n = 0

    if choice == 'e':
        print("\nИспользуется встроенный пример.")
        A_matrix = np.array([[1.0, 2.0, 0.0],
                             [2.0, 5.0, 1.0],
                             [0.0, 1.0, 3.0]], dtype=float)
        #  обратная :
        # [[14., -6.,  2.],
        #  [-6.,  3., -1.],
        #  [ 2., -1.,  1.]]
        n = A_matrix.shape[0]

    elif choice == 'm':
        print("\n--- Ввод данных вручную ---")
        # --- Ввод размерности ---
        while True:
            n_str = input("Введите размерность матрицы N : ")

            n = int(n_str)
            if n > 0:
                break



        # --- Ввод матрицы A ---
        print(f"\nВведите элементы матрицы A ({n}x{n}), по одной строке за раз.")
        print("Элементы в строке разделяйте пробелами.")
        A_list = []
        for i in range(n):
            while True:
                row_str = input(f"Строка {i+1} (ровно {n} чисел): ")
                elements = row_str.split()
                if len(elements) == n:
                    try:
                        row_floats = [float(el) for el in elements]
                        A_list.append(row_floats)
                        break
                    except ValueError:
                        print("Ошибка: Введите только числа, разделенные пробелами.")
                else:
                    print(f"Ошибка: Введите ровно {n} элементов.")
        A_matrix = np.array(A_list, dtype=float)




    print("\n--- Входная матрица A ---")
    print(np.round(A_matrix, 5))

    A_inverse = invert_matrix_lu(A_matrix)

    if A_inverse is not None:
        print("\nНайденная обратная матрица A_inv:\n", np.round(A_inverse, 5))

        print("\n--- Проверка: Вычисляем A * A_inv ---")
        check_identity = A_matrix @ A_inverse
        print("Результат A * A_inv:\n", np.round(check_identity, 5))


        # Оценка близости к единичной матрице
        identity_matrix = np.eye(n)
        difference = check_identity - identity_matrix

        diff_norm = np.linalg.norm(difference)
        print(f"\n ||A * A_inv - I||: {diff_norm:.2e}")



        print("\n--- Сравнение с np.linalg.inv ---")
        numpy_inverse = np.linalg.inv(A_matrix)
        print("Обратная матрица, найденная np.linalg.inv:\n", np.round(numpy_inverse, 5))
        inv_diff_norm = np.linalg.norm(A_inverse - numpy_inverse)
        print(f"|| A_inv - A_inv_nympy||: {inv_diff_norm:.2e}")



