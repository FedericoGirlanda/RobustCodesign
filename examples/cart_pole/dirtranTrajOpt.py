import time
import os
from datetime import timedelta
import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole.trajectory_optimization.dirtran.dirtranTrajOpt import DirtranTrajectoryOptimization
from cart_pole.model.parameters import Cartpole
from cart_pole.model.parameters import generateUrdf

with_simulation = True
save_dir = "data/cart_pole/dirtran/trajectory.csv"

# Trajectory Optimization
sys = Cartpole("short")
old_Mp = sys.Mp
sys.Mp = 0.227
sys.Jp = sys.Jp + (sys.Mp-old_Mp)*(sys.lp**2)
sys.fl = 6
urdf_path = generateUrdf(sys.Mp, sys.lp, sys.Jp)
options = {"N": 201,
           "x0": [0, np.pi, 0, 0],
           "xG": [0, 0, 0, 0],
           "hBounds": [0.01, 0.06],
           "fl": sys.fl,
           "cart_pos_lim": 0.3,
           "QN": np.diag([100, 100, 100, 100]),
           "R": 10,
           "Q":  np.diag([10, 10, 1, 1]),
           "time_penalization": 0,
           "tf0": 8,
           "urdf": urdf_path }
trajopt = DirtranTrajectoryOptimization(sys, options)

# calculation with timer
print(f'Starting Day Time: {time.strftime("%H:%M:%S", time.localtime())}')
t_start = time.time()
[timesteps, x_trj, u_trj] = trajopt.ComputeTrajectory()
print(f'Calculation Time(sec): {(time.time() - t_start)}')

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

# save results
TIME = timesteps
CART_POS = x_trj[:, 0][:, np.newaxis]
PEND_POS = x_trj[:, 1][:, np.newaxis]
CART_VEL = x_trj[:, 2][:, np.newaxis]
PEND_VEL = x_trj[:, 3][:, np.newaxis]
FORCE = u_trj
print(f'TIME shape: {TIME.shape}')
print(f'CART_POS shape: {CART_POS.shape}')
print(f'Force shape: {FORCE.shape}')

# traj_data = np.hstack((TIME, CART_POS, PEND_POS, CART_VEL, PEND_VEL, FORCE))
# np.savetxt(save_dir, traj_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="") 
# print("Trajectory saved in:", save_dir)

if with_simulation: 
    from cart_pole.utilities.process_data import prepare_trajectory
    from cart_pole.simulation.simulator import StepSimulator

    print("Simulating...")
          
    trajectory = np.loadtxt(save_dir, skiprows=1, delimiter=",")
    X = np.array([trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]])
    U = np.array([trajectory.T[5]])
    T = np.array([trajectory.T[0]]).T 
    traj_dict = prepare_trajectory(save_dir)

    # Simulation of tvlqr stabilization
    traj_x1 = traj_dict["des_cart_pos_list"]
    traj_x2 = traj_dict["des_pend_pos_list"]
    traj_x3 = traj_dict["des_cart_vel_list"]
    traj_x4 = traj_dict["des_pend_vel_list"]
    controller_options = {"T_nom": traj_dict["des_time_list"],
                            "U_nom": traj_dict["des_force_list"],
                            "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                            "Q": options["Q"],
                            "R": np.array([options["R"]]),
                            "xG": options["xG"]}
    cartpole = {"urdf": urdf_path,
                "sys": sys,
                "x_lim": options["cart_pos_lim"]}
    dt_sim = 0.003 #traj_dict["des_time_list"][1]-traj_dict["des_time_list"][0]
    sim = StepSimulator(cartpole, controller_options)
    sim.init_simulation(dt_sim = dt_sim,init_knot = 30, final_knot = 70)
    T_sim, X_sim, U_sim = sim.simulate()
    print(len(T_sim))
    assert False

    # Plot the results
    fig_test, ax_test = plt.subplots(2,2, figsize = (8, 8))
    fig_test.suptitle(f"Dynamics trajectory stabilization: simulated(blue) vs desired(orange)")
    ax_test[0][0].plot(T_sim, X_sim[0])
    ax_test[0][1].plot(T_sim, X_sim[1])
    ax_test[1][0].plot(T_sim, X_sim[2])
    ax_test[1][1].plot(T_sim, X_sim[3])
    ax_test[0][0].plot(T, X[0], linestyle = "--")
    ax_test[0][1].plot(T, X[1], linestyle = "--")
    ax_test[1][0].plot(T, X[2], linestyle = "--")
    ax_test[1][1].plot(T, X[3], linestyle = "--")
    ax_test[0][0].hlines(np.vstack((np.ones((len(T),1)),-np.ones((len(T),1))))*0.3,T[0], T[-1])
    ax_test[0][0].set_xlabel("x0(x_cart)")
    ax_test[0][1].set_xlabel("x1(theta)")
    ax_test[1][0].set_xlabel("x2(x_cart_dot)")
    ax_test[1][1].set_xlabel("x3(theta_dot)")

    fig_test, ax_test = plt.subplots(1,1, figsize = (8, 8))
    ax_test.plot(T_sim,U_sim.T)
    ax_test.plot(T,U.T, linestyle = "--")
    ax_test.hlines(np.vstack((np.ones((len(T),1)),-np.ones((len(T),1))))*sys.fl,T[0], T[-1])
    ax_test.set_xlabel("u(force)")

plt.show()