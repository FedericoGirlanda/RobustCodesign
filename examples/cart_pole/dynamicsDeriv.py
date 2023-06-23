from sympy import *
import numpy as np

fl,Rm,Lm,Kt,eta_m,Km,Jm,Mc,Mw,Kg,eta_g,r_mp,Beq,M,Jeq,Mp,Lp,lp,Jp,Bp,g = symbols("self.fl self.Rm self.Lm self.Kt self.eta_m self.Km self.Jm self.Mc self.Mw self.Kg self.eta_g self.r_mp self.Beq self.M self.Jeq self.Mp self.Lp self.lp self.Jp self.Bp self.g")
x0,x1,x2,x3, u, w = symbols("x[0] x[1] x[2] x[3] u[0] w")
x = [x0+w,x1+w,x2+w,x3+w]
u = [u]

D = 4 * M * r_mp ** 2 + Mp * r_mp ** 2 + 4 * Jm * Kg ** 2 \
            + 3 * Mp * r_mp ** 2 * sin(x[1]) ** 2

c_acc = - 3 * r_mp ** 2 * Bp * cos(x[1]) * x[3] / (lp * D) \
        - 4 * Mp * lp * r_mp ** 2 * sin(x[1]) * x[3] ** 2 / D \
        - 4 * r_mp ** 2 * Beq * x[2] / D \
        + 3 * Mp * r_mp ** 2 * g * cos(x[1]) * sin(x[1]) / D \
        + 4 * r_mp**2 * u[0] / (D * eta_g * eta_m)

p_acc = - 3 * (M * r_mp**2 + Mp * r_mp**2 + Jm * Kg**2) * Bp * x[3] \
        / (Mp * lp**2 * D) - 3 * Mp * r_mp**2 * cos(x[1]) * sin(x[1]) * x[3]**2 \
        / D - 3 * r_mp**2 * Beq * cos(x[1]) * x[2] / (lp * D) \
        + 3 * (M * r_mp**2 + Mp * r_mp**2 + Jm * Kg**2) * g * sin(x[1]) \
        / (lp * D) + 3 * r_mp**2 * cos(x[1]) * u[0] / (lp * D * eta_g * eta_m)
f = np.array([x[2], x[3], c_acc, p_acc])

df0dx0 = diff(f[0],x0)
df1dx0 = diff(f[1],x0)
df0dx1 = diff(f[0],x1)
df1dx1 = diff(f[1],x1)
df0dx2 = diff(f[0],x2)
df1dx2 = diff(f[1],x2)
df0dx3 = diff(f[0],x3)
df1dx3 = diff(f[1],x3)
df0du  = diff(f[0],u[0])
df1du  = diff(f[1],u[0])
df0dw  = diff(f[0],w)
df1dw  = diff(f[1],w)
df2dx0 = diff(f[2],x0)
df3dx0 = diff(f[3],x0)
df2dx1 = diff(f[2],x1)
df3dx1 = diff(f[3],x1)
df2dx2 = diff(f[2],x2)
df3dx2 = diff(f[3],x2)
df2dx3 = diff(f[2],x3)
df3dx3 = diff(f[3],x3)
df2du  = diff(f[2],u[0])
df3du  = diff(f[3],u[0])
df2dw  = diff(f[2],w)
df3dw  = diff(f[3],w)

print("df0dx0= ", df0dx0)
print("df1dx0= ", df1dx0)
print("df0dx1= ", df0dx1)
print("df1dx1= ", df1dx1)
print("df0dx2= ", df0dx2)
print("df1dx2= ", df1dx2)
print("df0dx3= ", df0dx3)
print("df1dx3= ", df1dx3)
print("df0du = ", df0du)
print("df1du = ", df1du)
print("df0dw = ", df0dw)
print("df1dw = ", df1dw)
print("df2dx0= ", df2dx0)
print("df3dx0= ", df3dx0)
print("df2dx1= ", df2dx1)
print("df3dx1= ", df3dx1)
print("df2dx2= ", df2dx2)
print("df3dx2= ", df3dx2)
print("df2dx3= ", df2dx3)
print("df3dx3= ", df3dx3)
print("df2du = ", df2du)
print("df3du = ", df3du)
print("df2dw = ", df2dw)
print("df3dw = ", df3dw)