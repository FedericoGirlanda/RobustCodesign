import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

from cart_pole.model.parameters import Cartpole
from cart_pole.controllers.tvlqr.tvlqr import TVLQRController
from cart_pole.utilities.process_data import prepare_trajectory
from cart_pole.simulation.simulator import DrakeStepSimulator, StepSimulator

from pydrake.all import PiecewisePolynomial
from generateUrdf import generateUrdf

def overSamplingTraj(T,X,U, dt_o, csv_path, sys = None):
    dt = T[1] - T[0]
    # if dt%dt_o != 0:
    #     print("Choose another dt_o")
    #     assert False
    osf = int(dt/dt_o) # knot point per interval
    n_os = int(len(T)*osf) # new n of knot points
    
    # Drake interpolation
    T_o = np.linspace(T[0],T[-1],n_os)
    X_o = np.zeros((n_os,4))
    U_o = np.zeros((n_os,1))
    U_p = PiecewisePolynomial.FirstOrderHold(T, U)
    X_p = PiecewisePolynomial.CubicShapePreserving(T, X, zero_end_point_derivatives=True)
    for j in range(n_os):
        U_o[j] = U_p.value(T_o[j])
        X_o[j] = X_p.value(T_o[j]).T[0]

    csv_data = np.vstack((T_o[:,0], X_o[:,0], X_o[:,1], X_o[:,2], X_o[:,3], U_o[:,0])).T
    np.savetxt(csv_path, csv_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="")

    return T_o,X_o,U_o

# System, trajectory and controller init
sys = Cartpole("short")
sys.fl = 6
generateUrdf(sys.Mp,sys.lp)
urdf_path = "data/cart_pole/urdfs/cartpole_CMAES.urdf"
xG = np.array([0,0,0,0])
dirtranTraj_path = "data/cart_pole/dirtran/trajectory.csv"    
trajectory = np.loadtxt(dirtranTraj_path, skiprows=1, delimiter=",")
X = np.array([trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]])
U = np.array([trajectory.T[5]])
T = np.array([trajectory.T[0]]).T 
traj_dict = prepare_trajectory(dirtranTraj_path)

# Simulation of tvlqr stabilization
traj_x1 = traj_dict["des_cart_pos_list"]
traj_x2 = traj_dict["des_pend_pos_list"]
traj_x3 = traj_dict["des_cart_vel_list"]
traj_x4 = traj_dict["des_pend_vel_list"]
controller_options = {"T_nom": traj_dict["des_time_list"],
                        "U_nom": traj_dict["des_force_list"],
                        "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                        "Q": np.diag([10,10,1,1]),
                        "R": np.array([.1]),
                        "xG": xG}
cartpole = {"urdf": urdf_path,
            "sys": sys,
            "x_lim": 0.35}
dt_sim = 0.01
sim = StepSimulator(cartpole, controller_options, dt_sim)
sim.init_simulation()
T_sim, X_sim, U_sim = sim.simulate()

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