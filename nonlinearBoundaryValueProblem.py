import numpy as np
import constants.variables as const
import tridiagonalMatrixAlgorithm as tma
import utils.logger as logger
import utils.util as util
import utils.equation as equation

# метод Ньютона
def solve(inputA, inputC, n):
    H = np.zeros((n-1, 1)) # вектор-функция H
    dH = np.zeros((n-1, n-1)) # вектор-функция производных значений H
    k = np.ones((n, 1))  # вектор-решение метода прогонки, является y(k)
    v = np.zeros((n-1, 1))  # вектор-решение шага 2 метода Ньютона
    A = util.init_A(n)  # матрица А
    h = util.init_h(n)  # h
    y = np.zeros((n+1, 1)) # результат решения методом Ньютона, игрики

    logger.log(text="Начинаем итерации методом Ньютона", force=True)
    while np.max(abs(k)) > const.STOP_VALUE:

        # Шаг 1
        for i in range(0, n-1, 1):
            H[i] = pow(h,2) * equation.g(inputA, inputC, h, i, v[i])
            dH[i][i] = pow(h,2) * equation.dg(inputA, v[i])
        logger.log(text='H', value=H)
        logger.log(text='dH', value=dH)

        L = (A + dH ) # * k # метод Ньютона, левая часть
        R = -1 * (np.dot(A, v) + H)  # метод Ньютона, правая часть
        logger.log(text='L', value=L)
        logger.log(text='R', value=R)

        # вычисление вектора-решения методом прогонки
        k = tma.solve(L, R)
        logger.log(text='k', value=k, force=True)

        #if np.isinf(np.max(abs(k))) or np.isnan(np.max(abs(k))):
        #    break

        # Шаг 2
        v = v + k
        logger.log(text='v', value=v, force=True)
    logger.log(text="Итерации методом Ньютона завершены", force=True)

    # заполняем y
    for i in range(1, n, 1):
        y[i] = v[i-1]
    logger.log(text='y', value=y)

    return y


