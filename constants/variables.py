import numpy as np

A = 0 # интервал от
B = 1 # интервал до

N_ARR = np.array([10, 100, 1000]) # список n
#N_ARR = np.array([10]) # список n
STOP_VALUE = pow(10,-6) # значение, до которого проводятся итерации метода Ньютона

LOG = False # поставить в True, если нужно выводить в консоль инфу по созданным объектам, иначе - False