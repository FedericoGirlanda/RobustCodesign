import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("WebAgg")
from simple_pendulum.utilities.process_data import prepare_trajectory

from pydrake.all import FiniteHorizonLinearQuadraticRegulatorOptions, \
                        DiagramBuilder, Saturation, LogVectorOutput,\
                        FiniteHorizonLinearQuadraticRegulator, Simulator, PiecewisePolynomial
from pydrake.examples.pendulum import PendulumPlant

from simple_pendulum.trajectory_optimization.dirtran.dirtranTrajOpt import DirtranTrajectoryOptimization, DrakeDirtranTrajectoryOptimization

from examples.simple_pendulum.rtc_CMAES import roaVolComputation

with_simulation = True
save_dir = "data/simple_pendulum/dirtran/trajectory.csv"

# pendulum parameters
mpar = {"l": 0.4, 
        "m": 0.7,
        "b": 0.1,
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}

# robust direct transcription parameters
options = {"N": 51,
        "R": .1,
        "Q": np.diag([10,1]),
        "QN": np.eye(2)*100,
        "x0": [0.0,0.0],
        "xG": [np.pi, 0.0],
        "tf0": 3,
        "speed_limit": 7,
        "theta_limit": 2*np.pi,
        "time_penalization": 0.1,
        "hBounds": [0.01,0.1]}  

import time
start = time.time()
dirtran = DirtranTrajectoryOptimization(mpar, options)
T,X,U = dirtran.ComputeTrajectory()
print("Duration(mins): ", int((time.time()-start)/60))
print("Duration(secs): ", (time.time()-start))

if not with_simulation:
    plt.plot(T,X[0], label = "theta", color = "blue")
    plt.plot(T,X[1], label = "theta_dot", color = "orange")
    plt.plot(T,U[0], label = "u", color = "purple")
    plt.legend()
    plt.show()
else:
    plt.plot(T,X[0], linestyle= "dashed", label = "theta", color = "blue")
    plt.plot(T,X[1], linestyle= "dashed", label = "theta_dot", color = "orange")
    plt.plot(T,U[0], linestyle= "dashed", label = "u", color = "purple")
    plt.legend()

    from simulatorComparison import usingFelixSimulator
    
    traj_data = np.vstack((T, X[0], X[1], U[0])).T
    traj_path = "data/simple_pendulum/dirtran/trajectory.csv"
    np.savetxt(traj_path, traj_data, delimiter=',',
            header="time,pos,vel,torque", comments="")
    t_sim,x_sim,u_sim = usingFelixSimulator(mpar,traj_path,options["Q"],[options["R"]])
    plt.plot(t_sim,x_sim)
    plt.plot(t_sim,u_sim, color = "purple")

    funnel_path = "data/simple_pendulum/funnels/SosfunnelDIRTRAN.csv"
    start = time.time()
    print("Volume of DIRTRAN funnel: ",roaVolComputation(mpar,options,traj_path, funnel_path = funnel_path, verbose=True))
    print("Duration(secs): ", (time.time()-start))

    plt.show()