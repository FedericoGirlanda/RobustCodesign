import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from cart_pole.model.parameters import Cartpole

sys = Cartpole("short")

N = 100
x_c_dot = np.linspace(-0.7,0.7,N)
f = np.linspace(-6,6,N)

res = np.zeros((N**2, 3))
k = 0
for i in range(N):
    f_i = f[i]
    for j in range(N):
        x_c_dot_j = x_c_dot[j]
        res[k] = np.array([f_i,x_c_dot_j,sys.amplitude(f_i,x_c_dot_j)])
        k += 1

print("minus V max: ",np.max(res.T[2]))
print("plus V max: ",np.max(-res.T[2]))