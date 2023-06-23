import time
import os
from datetime import timedelta
import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from pathlib import Path

from TrajOpt_TrajStab_CMAES import roaVolComputation
from cart_pole.trajectory_optimization.dirtrel.dirtrelTrajOpt import RobustDirtranTrajectoryOptimization
from cart_pole.model.parameters import Cartpole

save = True
funnel_volume = True

# Trajectory Optimization
sys = Cartpole("short")
options = {"N": 101,
           "x0": [0, np.pi, 0, 0],
           "xG": [0, 0, 0, 0],
           "hBounds": [0.01, 0.05],
           "fl": 8,
           "cart_pos_lim": 0.35,
           "QN": np.diag([100, 100, 100, 100]),
           "R": 10,
           "Q": np.diag([10, 10, 1, 1]),
           "time_penalization": 0,
           "Rl": .1,
           "QNl": np.diag([100, 100, 100, 100]),
           "Ql": np.diag([10, 10, 1, 1]),
           "D": .2*.2, 
           "E1": np.zeros((4,4)),
           "tf0": 5,
           "urdf": "data/cart_pole/urdfs/cartpole.urdf"}
    
trajopt = RobustDirtranTrajectoryOptimization(sys, options)

# calculation with timer
print(f'Starting Day Time: {time.strftime("%H:%M:%S", time.localtime())}')
t_start = time.time()
[timesteps, x_trj, u_trj] = trajopt.ComputeTrajectory()
print(f'Calculation Time (mins): {int((time.time() - t_start)/60)}')
print(f'Calculation Time (sec): {int((time.time() - t_start))}')

timesteps = np.array(timesteps)[:, np.newaxis]
print(f'timesteps shape: {timesteps.shape}')
u_trj = np.array(u_trj)[:, np.newaxis]
print(f'input size: {u_trj.shape}')
x_trj = x_trj.T
print(f'state size: {x_trj.shape}')

# plot results
fig, ax = plt.subplots(5, 1, figsize=(18, 6), sharex="all")
ax[0].plot(timesteps, x_trj[:, 0] * 1000, label="x")
ax[0].set_ylabel("cart pos. [mm]")
ax[0].legend(loc="best")
ax[1].plot(timesteps, x_trj[:, 1], label="theta")
ax[1].set_ylabel("pend. pos. [rad]")
ax[1].legend(loc="best")
ax[2].plot(timesteps, x_trj[:, 2] * 1000, label="x_dot")
ax[2].set_ylabel("cart vel. [mm/s]")
ax[2].legend(loc="best")
ax[3].plot(timesteps, x_trj[:, 3], label="theta_dot")
ax[3].set_ylabel("pend. vel. [rad/s]")
ax[3].legend(loc="best")
ax[4].plot(timesteps, u_trj, label="u")
ax[4].set_xlabel("time [s]")
ax[4].set_ylabel("Force [N]")
ax[4].legend(loc="best")
plt.show()

if save:
    TIME = timesteps
    CART_POS = x_trj[:, 0][:, np.newaxis]
    PEND_POS = x_trj[:, 1][:, np.newaxis]
    CART_VEL = x_trj[:, 2][:, np.newaxis]
    PEND_VEL = x_trj[:, 3][:, np.newaxis]
    FORCE = u_trj
    print(f'TIME shape: {TIME.shape}')
    print(f'CART_POS shape: {CART_POS.shape}')
    print(f'Force shape: {FORCE.shape}')

    WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])
    print("Workspace is set to:", WORK_DIR)
    csv_path = "data/cart_pole/dirtrel/trajectory.csv"
    csv_data = np.hstack((TIME, CART_POS, PEND_POS, CART_VEL, PEND_VEL, FORCE))
    np.savetxt(csv_path, csv_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="")

if funnel_volume:
    funnel_path = "data/cart_pole/RoA/Probfunnel_DIRTREL.csv"
    print("DIRTREL funnel volume: ", roaVolComputation(sys, csv_path,funnel_path, options))
