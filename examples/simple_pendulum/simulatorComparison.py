import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.simulation.simulation import Simulator, DrakeSimulator
from simple_pendulum.controllers.tvlqr.tvlqr import TVLQRController
from simple_pendulum.utilities.process_data import prepare_trajectory
from simple_pendulum.trajectory_optimization.dirtrel.dirtrelTrajOpt import pendulum, tvlqr_controller

def usingFelixSimulator(pendulum_par, trajectory_path, Q, R, t0 = 0, x0 = [0.0,0.0],u0 = None):
    data_dict = prepare_trajectory(trajectory_path)
    trajectory = np.loadtxt(trajectory_path, skiprows=1, delimiter=",")
    T = trajectory.T[0].T
    X = [trajectory.T[1].T, trajectory.T[2].T]
    U = trajectory.T[3].T
    if u0 == None:
        u0 = U[0]
    pendulum_plant = PendulumPlant(mass=pendulum_par["m"],
                         length=pendulum_par["l"],
                         damping=pendulum_par["b"],
                         gravity=pendulum_par["g"],
                         coulomb_fric=pendulum_par["cf"],
                         inertia=None,
                         torque_limit=pendulum_par["tl"])
    sim = Simulator(plant=pendulum_plant)
    controller = TVLQRController(data_dict=data_dict, mass=pendulum_par["m"], length=pendulum_par["l"],
                                damping=pendulum_par["b"], gravity=pendulum_par["g"],
                                torque_limit=pendulum_par["tl"])
    controller.set_costs(Q, R)
    controller.set_goal(np.array(X).T[-1])

    dt_controller = (T[1]-T[0])
    T_sim = np.zeros((len(T),1))
    X_sim = np.zeros((len(T),2))
    X_sim[0] = x0
    U_sim = np.zeros((len(T),1))
    U_sim[0] = u0
    for i in range(len(T)-1):
        tf = T[i+1]
        t0 = T[i] 
        x0 = np.array(X).T[i]
        ti, xi, ui= sim.simulate(t0,x0,tf,dt_controller,controller,integrator="euler")
        T_sim[i+1] = tf
        X_sim[i+1] = xi[-1]
        U_sim[i+1] = ui[-1]

    # tf = T[-1]
    # dt = T[1]-T[0] 
    # T_sim,X_sim,U_sim = sim.simulate(t0,x0,tf,dt,controller)
    #T_sim = T_sim - dt_controller

    return T_sim, X_sim, U_sim

def usingDrakeSimulator(pendulum_par, trajectory_path, Q, R, x0 = [0.0, 0.0],init_knot = 0):
    trajectory = np.loadtxt(trajectory_path, skiprows=1, delimiter=",")
    X = [trajectory.T[1].T, trajectory.T[2].T]
    U = trajectory.T[3].T
    T = trajectory.T[0]
    T = np.reshape(T, (T.shape[0], -1))
    U = np.reshape(U,(U.shape[0], -1)).T
    dt = (T[1]-T[0])[0]
    drake_pendulum_plant = pendulum(pendulum_par["m"], pendulum_par["l"], pendulum_par["b"], pendulum_par["g"], pendulum_par["tl"])
    drake_controller = tvlqr_controller(drake_pendulum_plant, X, U, T, Q, R, Q)
    drake_sim = DrakeSimulator(drake_controller, drake_pendulum_plant, dt, dt, pendulum_par)
    T_sim, X_sim, U_sim = drake_sim.simulate(x0 = x0, init_knot=init_knot)

    return T_sim, X_sim.T, U_sim.T

if __name__ == "__main__": 
    # Pendulum and controller params
    traj_path = "data/simple_pendulum/dirtran/trajectory.csv"
    mpar = {"l": 0.5,
            "m": 0.67,
            "b": 0.35, 
            "g": 9.81,
            "cf": 0.0,
            "tl": 3}
    x0 = [0.0, 0.0]
    goal = [np.pi, 0.0]
    Q = np.diag([10,1])
    R = [0.1]

    # traj_path = "results/simple_pendulum/Design3.1Opt /CMA-ES/DesignOpt/Vdirtrel/trajectoryOptimal_CMAES.csv"
    # mpar = {"m":  0.5000177138814754,
    #         "l":  0.31735095140935576,
    #         "b": 0.35, 
    #         "g": 9.81,
    #         "cf": 0.0,
    #         "tl": 3}
    # Q = np.diag([9.71414241, 1.0114207])
    # R = [0.9425422759070431]

    # Nominal traj
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    X = [trajectory.T[1].T, trajectory.T[2].T]
    U = trajectory.T[3].T
    T = trajectory.T[0].T
    plt.plot(T, np.array(X).T, linestyle = "--", color = "black")
    plt.plot(T,np.array(U).T, linestyle = "--", color = "black")

    ##################
    # Felix Simulation
    ##################

    T_sim, X_sim, U_sim = usingFelixSimulator(mpar,traj_path,Q,R)

    # plot the resulting simulation
    plt.plot(T_sim, X_sim, color = "blue")
    plt.plot(T_sim,U_sim, color = "blue")

    ##################
    # Drake Simulation
    ##################

    # T_sim, X_sim, U_sim = usingDrakeSimulator(mpar, traj_path, Q, R)

    # # plot the resulting simulation
    # plt.plot(T_sim, X_sim, color = "orange")
    # plt.plot(T_sim,U_sim, color = "orange")
    plt.show()