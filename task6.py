import numpy as np
import math

def backward_substitution(U, y):
    n = U.shape[0]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        sum_val = U[i, i+1:] @ x[i+1:]
        x[i] = (y[i] - sum_val) / U[i, i]
    return x

def householder_qr_transform(A, b):
    """
    Выполняет QR-разложение A = QR, преобразуя b в c = Q^T b.
    Args:
        A : Прямоугольная матрица m x n .
        b : Вектор правой части .
    Returns:
         (R, c) где:
            R : Верхняя трапециевидная матрица.
            c :  Q^T b .
    """
    m, n = A.shape


    R = A.astype(np.float64).copy()
    c = b.astype(np.float64).copy()

    print(f"\Начало QR для матрицы {m}x{n}")

    for k in range(n):

        x = R[k:m, k]
        sub_dim = m - k

        norm_x = np.linalg.norm(x)

        if norm_x < 1e-12:
            print(f"    Норма подстолбца близка к нулю, пропускаем отражение")
            continue #переход к следующему столбцу

        v = x.copy()
        v[0] = v[0] + np.copysign(norm_x, x[0])

        v_dot_v = np.dot(v, v)
        if v_dot_v < 1e-12:
            print(f"    v^T v близко к нулю, пропускаем отражение")
            continue

        beta = 2.0 / v_dot_v

        Hk = np.eye(m)
        vvT = np.outer(v, v)
        H_sub = np.eye(sub_dim) - beta * vvT
        #  H_sub в нижний правый угол Hk
        Hk[k:m, k:m] = H_sub

        R = Hk @ R
        c = Hk @ c


    print("Выполнено QR разложение и преобразование b")
    return R, c


def solve_least_squares_qr(A, b):
    """
    Решает МНК min ||Ax - b||_2
    """
    m, n = A.shape
    print(f"\n--- Решение МНК для матрицы {m}x{n} ---")


    R, c = householder_qr_transform(A, b)


    print(" R:\n", np.round(R, 5))
    print("c = Q^T b:\n", np.round(c, 5))

    R_hat = R[0:n, 0:n] #  n x n часть
    c1 = c[0:n]         # первые n элементов вектора c
    print("R_hat :\n", np.round(R_hat, 5))
    print("c1 :\n", np.round(c1, 5))




    print("\nРешаем систему R_hat x = c1 обратной подстановкой")
    x_hat = backward_substitution(R_hat, c1)
    print("МНК-решение x найдено.")
    return x_hat

if __name__ == "__main__":

    A_example = np.array([[1., 1.],
                          [1., 2.],
                          [1., 3.],
                          [1., 4.]], dtype=float)
    b_example = np.array([1., 2., 2., 3.], dtype=float)

    # --- Выбор режима ---
    while True:
        mode = input("Выберите режим: 'e' -  пример, 'm' - ручной ввод: ").lower().strip()
        if mode in ['e', 'm']:
            break
        print("Неверный ввод.")

    A_input = None
    b_input = None

    if mode == 'e':
        print("\n--- Запуск примера ---")
        A_input = A_example
        b_input = b_example

    elif mode == 'm':
        print("\n--- Ввод данных вручную ---")

        while True:

            m = int(input("Введите количество строк m: "))
            n = int(input("Введите количество столбцов n: "))
            if m >= n and m > 0 and n > 0:
                break

        print(f"\nВведите элементы матрицы A ({m}x{n}), по строкам ")
        A_list = []
        for i in range(m): # Цикл по строкам
            while True:
                row_str = input(f"Строка {i+1} (ровно {n} чисел): ")
                elements = row_str.split()
                if len(elements) == n:
                    row_floats = [float(el) for el in elements]
                    A_list.append(row_floats)
                    break


        A_input = np.array(A_list, dtype=float)

        print(f"\nВведите элементы вектора b ({m} элементов)")
        while True:
            b_str = input(f"Вектор b ( {m} чисел): ")
            elements = b_str.split()
            if len(elements) == m:
                b_input = np.array([float(el) for el in elements], dtype=float)
                break





    if A_input is not None and b_input is not None:
        print(f"\n====== Расчет для {'примера' if mode == 'e' else 'введенных данных'} ======")
        print("Матрица A:\n", A_input)
        print("Вектор b:\n", b_input)

        x_solution = solve_least_squares_qr(A_input, b_input)

        if x_solution is not None:
            print("\n--- Результаты  ---")
            print("МНК Решение x:\n", np.round(x_solution, 5))
            residual = b_input - A_input @ x_solution
            print("Невязка r = b - A*x:\n", np.round(residual, 5))
            norm_residual = np.linalg.norm(residual)
            print("Норма невязки ||r||_2:", norm_residual)

            print("\n--- Сравнение с np.linalg.lstsq ---")

            x_np, res_np, rank_np, s_np = np.linalg.lstsq(A_input, b_input, rcond=None)
            print("Решение np.linalg.lstsq:\n", np.round(x_np, 5))
            print("Разница решений :", np.linalg.norm(x_solution - x_np))
            norm_residual_np = np.sqrt(res_np[0]) if len(res_np)>0 else np.linalg.norm(b_input - A_input @ x_np)
            print("Норма невязки :", norm_residual_np )
            print("Разница норм невязки:", abs(norm_residual - norm_residual_np))



