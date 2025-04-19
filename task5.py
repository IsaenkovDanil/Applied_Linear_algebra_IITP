import numpy as np

def ldlt_tridiagonal(main_diag, sub_diag):
    """
    Выполняет LDL^T разложение для симметричной трехдиагональной матрицы A
    Args:
        main_diag : 1D массив главной диагонали A (a[0], ..., a[n-1]).
        sub_diag : 1D массив под-диагонали A (c[1], ..., c[n-1])
    Returns:
        (l_subdiag, d_diag) :
            l_subdiag : 1D массив под-диагонали L (l[1], ..., l[n-1]).
            d_diag : 1D массив диагонали D (d[0], ..., d[n-1]).
    """


    n = len(main_diag)


    l_subdiag = np.zeros(n - 1, dtype=np.float64)
    d_diag = np.zeros(n, dtype=np.float64)

    d_diag[0] = main_diag[0]
    if d_diag[0] <= 1e-12:
        print(f"Ошибка: Матрица не положительно определена ")
        return None, None

    # Шаги i = 1 до n-1
    for i in range(n - 1): #  от 0 до n-2
        #  l_{i+1} = c_{i+1} / d_i 
        l_subdiag[i] = sub_diag[i] / d_diag[i]

        #  d_{i+1} = a_{i+1} - l_{i+1}^2 * d_i 
        d_diag[i+1] = main_diag[i+1] - (l_subdiag[i]**2) * d_diag[i]

        if d_diag[i+1] <= 1e-12:
            print(f"Ошибка: Матрица не положительно определена ")
            return None, None

    return l_subdiag, d_diag


def solve_ldlt_tridiagonal(l_subdiag, d_diag, b):
    """
    Решает систему Ax=b,  A - симметричная трехдиагональная матрица,
    используя  LDL^T разложение
    Args:
        l_subdiag : Под-диагональ L (l[1], ..., l[n-1])
        d_diag : Диагональ D (d[0], ..., d[n-1])
        b : Вектор правой части 

    Returns:
       x : Вектор решения x 
    """
    n = len(d_diag)

    #  Ly = b
    y = np.zeros(n, dtype=np.float64)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - l_subdiag[i-1] * y[i-1] # l_subdiag[i-1] = l_i

    #  Dz = y
    z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        z[i] = y[i] / d_diag[i]

    #  L^T x = z
    x = np.zeros(n, dtype=np.float64)
    x[n-1] = z[n-1]
  
    for i in range(n - 2, -1, -1):
        x[i] = z[i] - l_subdiag[i] * x[i+1] # l_subdiag[i] = l_{i+1}

    return x





def construct_tridiagonal(main_diag, sub_diag):
    """Строит полную матрицу A из ее диагоналей"""
    n = len(main_diag)
    A = np.zeros((n, n), dtype=float)
    np.fill_diagonal(A, main_diag)
    if n > 1:
        np.fill_diagonal(A[1:, :-1], sub_diag) #под-диагональ
        np.fill_diagonal(A[:-1, 1:], sub_diag) #над-диагональ
    return A


if __name__ == "__main__":


    # --- Пример симметричной трехдиагональной положительно определенной матрицы ---
    n_example = 5
    # гл. диаг. a[i]
    a_i = np.array([2.0] * n_example)
    #  под-диаг. c[i]
    c_i = np.array([-1.0] * (n_example - 1))
    b_i = np.arange(1, n_example + 1, dtype=float)

    # --- Пример симметричной трехдиагональной, но не положительно определенной матрицы ---
    a1_i = np.array([1.0, 1.0, 1.0, 1.0])
    c1_i = np.array([1.0, 1.0, 1.0]) # d3 будет <= 0
    b1_i = np.array([1., 2., 3., 4.])

    matrices_to_test = {
        "СПО трехдиагональная": (a_i, c_i, b_i),
        "Не ПО трехдиагональная": (a1_i, c1_i, b1_i),
    }

    while True:
        mode = input("Выберите режим: 't' - тест примеров, 'm' - ручной ввод: ").lower().strip()
        if mode in ['t', 'm']:
            break


    if mode == 't':
        for name, (a_diag, c_diag, b_vec) in matrices_to_test.items():
            print(f"\n====== Тест: {name} ======")
            print("Главная диагональ A:", a_diag)
            print("Под-диагональ A:", c_diag)
            print("Вектор b:", b_vec)

            # Выполняем LDL^T разложение
            l_sub, d_diag = ldlt_tridiagonal(a_diag, c_diag)

            if l_sub is not None and d_diag is not None:
                print("LDL^T разложение :")
                print("  Под-диагональ L :", np.round(l_sub, 5))
                print("  Диагональ D :", np.round(d_diag, 5))

                x_solution = solve_ldlt_tridiagonal(l_sub, d_diag, b_vec)


                print("\nРешение x_hat:\n", np.round(x_solution, 5))

                # Проверка
                A_full = construct_tridiagonal(a_diag, c_diag)
                print("\n(Полная матрица A для проверки:\n", np.round(A_full, 3), "\n)")
                residual = A_full @ x_solution - b_vec
                print("Проверка A*x_hat - b:\n", np.round(residual, 9))
                residual_norm = np.linalg.norm(residual)
                print(f"Норма невязки ||A*x_hat - b||_2: {residual_norm:.2e}")

                # Сравнение с numpy 
                x_numpy = np.linalg.solve(A_full, b_vec)
                print("\nРешение np.linalg.solve :\n", np.round(x_numpy, 5))
                print("Разница с numpy:", np.linalg.norm(x_solution - x_numpy))

            else:
                 print("\n Разложение LDL^T не удалось (матрица не положительно определена).")

    elif mode == 'm':
        print("\n--- Ввод данных вручную ---")
        a_manual = None
        c_manual = None
        b_manual = None
        n = 0
            # Ввод размерности N
        while True:
            n_str = input("Введите размерность матрицы N : ")
            n = int(n_str)
            if n >= 1: break


        print(f"\nВведите элементы главной диагонали A")
        while True:
            main_str = input("Главная диагональ: ")
            elements = main_str.split()
            if len(elements) == n:
                a_manual = np.array([float(el) for el in elements], dtype=float)
                break


        if n > 1:
            print(f"\nВведите элементы под-диагонали A")
            while True:
                sub_str = input("Под-диагональ: ")
                elements = sub_str.split()
                if len(elements) == n - 1:
                    c_manual = np.array([float(el) for el in elements], dtype=float)
                    break
        else: #под-диагонали нет
            c_manual = np.array([], dtype=float)


            print(f"\nВведите вектор b")
            while True:
                 b_str = input(f"Вектор b: ")
                 elements = b_str.split()
                 if len(elements) == n:
                    b_manual = np.array([float(el) for el in elements], dtype=float)
                    break
                 

                 



        # --- Решение для введенных данных ---
        if a_manual is not None and c_manual is not None and b_manual is not None:
            print("\n------ Расчет для введенных данных ------")
            print("Главная диагональ A:", a_manual)
            print("Под-диагональ A:", c_manual)
            print("Вектор b:", b_manual)

            l_sub, d_diag = ldlt_tridiagonal(a_manual, c_manual)

            if l_sub is not None and d_diag is not None:
                print("LDL^T разложение :")
                print("  Под-диагональ L (l_i):", np.round(l_sub, 5))
                print("  Диагональ D (d_i):", np.round(d_diag, 5))

                x_solution = solve_ldlt_tridiagonal(l_sub, d_diag, b_manual)
                print("\nНайденное решение x_hat:\n", np.round(x_solution, 5))

                # Проверка
                A_full = construct_tridiagonal(a_manual, c_manual)
                print("\n(Полная матрица A для проверки:\n", np.round(A_full, 3), "\n)")
                residual = A_full @ x_solution - b_manual
                print("Проверка A*x_hat - b:\n", np.round(residual, 9))
                residual_norm = np.linalg.norm(residual)
                print(f"Норма невязки ||A*x_hat - b||_2: {residual_norm:.2e}")

                # Сравнение с numpy

                x_numpy = np.linalg.solve(A_full, b_manual)
                print("\nРешение np.linalg.solve :\n", np.round(x_numpy, 5))
                print("Разница с numpy:", np.linalg.norm(x_solution - x_numpy))

            else:
                 print("\nРазложение LDL^T не удалось (матрица не положительно определена).")

