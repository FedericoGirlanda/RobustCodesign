import numpy as np
import os
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

# Read the saved data
save_dir = "results/simple_pendulum/Design3.1Opt /gridSearch/"
Id = "Controller"
data_path = os.path.join(save_dir, "data"+Id+".csv" )
data_readed = np.array(pd.read_csv(data_path))
vol_storage = -data_readed[:,3]

# 3d scatter heatmap
cube_l = 10
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
i = 0
j = 0
k = 0
for l in range(len(data_readed[:,0])):
    ax.scatter(data_readed[l,0], data_readed[l,1], data_readed[l,2], c = -data_readed[l,3], cmap=cm.Oranges, s=35, vmin=vol_storage.min(), vmax=vol_storage.max())
    i += 1
    if i%cube_l == 0:
        i = 0
        j += 1
        if j%cube_l == 0:
            j = 0
            k += 1
#ax.set_title("3D Heatmap")
ax.set_xlabel('q11')
ax.set_ylabel('q22')
ax.set_zlabel('r')
color_map = cm.ScalarMappable(cmap=cm.Oranges)
color_map.set_array(vol_storage)
plt.colorbar(color_map, ax = ax, location = "left")
data_path = os.path.join(save_dir, "3dheatmap"+Id+".png" )
fig.savefig(data_path)

# 2d heatmap with a fixed parameter
fixed_par = "r"
fixed_value = 1
fig, ax = plt.subplots()
if fixed_par == "q11":
    fixed_idx = 0
    plotted_idx = [1,2]
    ax.set_title(f'q22VSr, q11 = {np.round(fixed_value,4)}')
    ax.set_xlabel('q22')
    ax.set_ylabel('r')
    file_name = f"q22VSr_q11is{np.round(fixed_value,2)}"
elif fixed_par == "q22":
    fixed_idx = 1
    plotted_idx = [0,2]
    ax.set_title(f'q11VSr, q22 = {np.round(fixed_value,4)}')
    ax.set_xlabel('q11')
    ax.set_ylabel('r')
    file_name = f"q11VSr_q22is{np.round(fixed_value,2)}"
elif fixed_par =="r":
    fixed_idx = 2
    plotted_idx = [0,1]
    ax.set_title(f'q11VSq22, r = {np.round(fixed_value,4)}')
    ax.set_xlabel('q11')
    ax.set_ylabel('q22')
    file_name = f"q11VSq22_ris{np.round(fixed_value,2)}"

i = 0
j = 0
X = np.zeros((cube_l, cube_l))
Y = np.zeros((cube_l, cube_l))
V = np.zeros((cube_l,cube_l))
for l in range(cube_l*cube_l*cube_l):
    if np.round(data_readed[l,fixed_idx],4) == np.round(fixed_value,4):
        X[i,j] = data_readed[l,plotted_idx[0]]
        Y[i,j] = data_readed[l,plotted_idx[1]]
        V[i,j] = -data_readed[l,3]
        i += 1
        if i%cube_l == 0:
            i = 0
            j += 1
plt.pcolormesh(X, Y, V, cmap='RdBu', vmin = np.abs(vol_storage).min(), 
                                    vmax = np.abs(vol_storage).max())
plt.colorbar()
data_path = os.path.join(save_dir, file_name+Id+".png" )
fig.savefig(data_path)

# displaying plots
plt.show()