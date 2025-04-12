import numpy as np

def ldlt_decomposition(A):
    """
    Выполняет LDL^T разложение для симметричной матрицы A
    Args:
        A : Квадратная симметричная матрица (N x N).
    Returns:
         (L, D_diag):
            L : Нижняя треугольная матрица с 1 на диагонали (N x N).
            D_diag : 1D массив с диагональными элементами матрицы D.

    """
    n = A.shape[0]




    L = np.eye(n, dtype=np.float64)
    D_diag = np.zeros(n, dtype=np.float64)

    for j in range(n):
        # Вычисляем сумму для d_j
        sum_d = 0.0
        if j > 0:
                sum_d = np.sum((L[j, :j]**2) * D_diag[:j])
        #  d_j
        D_diag[j] = A[j, j] - sum_d



        # Вычисляем элементы L[i, j] для i > j
        for i in range(j + 1, n):
            # Вычисляем сумму для L[i, j]
            sum_l = 0.0
            if j > 0:
                sum_l = np.sum(L[i, :j] * L[j, :j] * D_diag[:j])


            #  L[i, j]
            if abs(D_diag[j]) < 1e-12: # Проверка перед делением
                # Если D[j] ноль
                numerator = A[i, j] - sum_l
                if abs(numerator) > 1e-12:
                    print(f"Ошибка d_j=0")
                    return None, None
                else:
                    L[i, j] = 0.0 # Случай 0/0
            else:
                L[i, j] = (A[i, j] - sum_l) / D_diag[j]

    return L, D_diag



def is_positive_definite(A):
    """
    Проверяет, является ли симметричная матрица A
    положительно определенной, используя LDL^T разложение.
    Args:
        A : Квадратная матрица (N x N).
    Returns:
        bool: True, если A симметрична и положительно определена, False в противном случае.
    """




    print("\n--- Проверка на положительную определенность ---")

    # Шаг 1 и 2: Пытаемся выполнить LDL^T разложение (включает проверку на симметрию)
    L, D_diag = ldlt_decomposition(A)


    all_positive = np.all(D_diag > 1e-12)

    if all_positive:
        print("Матрица является положительно определенной.")
        return True
    else:
        print("Матрица не является положительно определенной.")
        return False

if __name__ == "__main__":

    #Симметричная положительно определенная
    A_pos_def = np.array([[4., 12., -16.],
                          [12., 37., -43.],
                          [-16., -43., 98.]], dtype=float)

    #  Симметричная, не положительно определенная
    A_not_pos_def = np.array([[1., 2., 3.],
                              [2., 1., 2.],
                              [3., 2., 1.]], dtype=float)




    matrices_to_test = {
        "Положительно определенная": A_pos_def,
        "Не положительно определенная": A_not_pos_def,
    }

    while True:
        mode = input("Выберите режим: 't' - тест примеров, 'm' - ручной ввод: ").lower().strip()
        if mode in ['t', 'm']:
            break

    if mode == 't':


        for name, matrix in matrices_to_test.items():
            print(f"\n====== Тестирование матрицы: {name} ======")
            print("Матрица A:\n", matrix)
            is_positive_definite(matrix)
            L_res, D_diag_res = ldlt_decomposition(matrix)

            if L_res is not None and D_diag_res is not None:

                print("\nРезультат LDL^T разложения:")
                print("Матрица L:\n", np.round(L_res, 3))
                D_res = np.diag(D_diag_res)
                print("Матрица D:\n", np.round(D_res, 3))
                # Проверка A == L @ D @ L.T
                A_reconstructed = L_res @ D_res @ L_res.T
                print("Проверка L @ D @ L.T:\n", np.round(A_reconstructed,3))
                print(" ||A - LDL^T||:", np.linalg.norm(matrix - A_reconstructed))


    elif mode == 'm':
        print("\n--- Ввод матрицы A вручную ---")

        while True:
            n_str = input("Введите размерность N: ")
            n = int(n_str)

        print(f"Введите элементы матрицы A ({n}x{n}), по строкам")
        A_list = []
        for i in range(n):
            while True:
                row_str = input(f"Строка {i+1}: ")
                elements = row_str.split()
                if len(elements) == n:
                    A_list.append([float(el) for el in elements])
                    break
        A_manual = np.array(A_list, dtype=float)

        print("\nВведенная матрица A:\n", A_manual)
        is_positive_definite(A_manual)
        L_res, D_diag_res = ldlt_decomposition(A_manual)
        if L_res is not None and D_diag_res is not None:
                print("\nРезультат LDL^T разложения:")
                print("Матрица L:\n", np.round(L_res, 3))
                D_res = np.diag(D_diag_res)
                print("Матрица D:\n", np.round(D_res, 3))
                A_reconstructed = L_res @ D_res @ L_res.T
                print("Проверка L @ D @ L.T:\n", np.round(A_reconstructed,3))
                print("Разница ||A - LDL^T||:", np.linalg.norm(A_manual - A_reconstructed))

