import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("WebAgg")

from TrajOpt_TrajStab_CMAES import roaVolComputation
from simple_pendulum.trajectory_optimization.dirtrel.dirtrelTrajOpt import RobustDirtranTrajectoryOptimization
from simple_pendulum.controllers.tvlqr.roa.plot import plotFirstLastEllipses, plotFunnel, plotRhoEvolution,\
                                                       TVrhoVerification, funnel2DComparison, rhoComparison

funnel_comparison = True
with_simulation = True

# pendulum parameters
mpar = {"l": 0.4, 
        "m": 0.67,
        "b": 0.2,
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}

# robust direct transcription parameters
options = {"N": 51,
        "R": .1,
        "Rl": .1,
        "Q": np.diag([10,1]),
        "Ql": np.diag([10,1]),
        "QN": np.eye(2)*100,
        "QNl": np.eye(2)*100,
        "D": .2*.2, 
        "E1": np.zeros((2,2)),
        "x0": [0.0,0.0],
        "xG": [np.pi, 0.0],
        "tf0": 3,
        "speed_limit": 7,
        "theta_limit": 2*np.pi,
        "time_penalization": 0.1, 
        "hBounds": [0.01, 0.2]}

import time
start = time.time()
dirtrel = RobustDirtranTrajectoryOptimization(mpar, options)
T, X, U = dirtrel.ComputeTrajectory()
print("Duration(mins): ", int((time.time()-start)/60))
print("Duration(secs): ", (time.time()-start))

# S = np.zeros((2,2,options["N"])) # TODO: why E is always singular and not >0?
# S[:,:,0] = options["E1"]
# for k in range(options["N"]-1):
#     E_kplus1 = ExtractValue(dirtrel.E[:,:,k+1])
#     det_E = np.linalg.det(E_kplus1)
#     print(E_kplus1)
#     print(det_E)
    #S[:,:,k+1] = np.linalg.inv(E_kplus1)
#print(S)
      
if not with_simulation:
    plt.plot(T,X[0], label = "theta", color = "blue")
    plt.plot(T,X[1], label = "theta_dot", color = "orange")
    plt.plot(T,U, label = "u", color = "purple")
    plt.legend()
    plt.show()
else:
    plt.plot(T,X[0], linestyle= "dashed", label = "theta", color = "blue")
    plt.plot(T,X[1], linestyle= "dashed", label = "theta_dot", color = "orange")
    plt.plot(T,U, linestyle= "dashed", label = "u", color = "purple")
    plt.legend()

    from simulatorComparison import usingFelixSimulator
    
    traj_data = np.vstack((T, X[0], X[1], U)).T
    traj_path = "data/simple_pendulum/dirtrel/trajectory.csv"
    np.savetxt(traj_path, traj_data, delimiter=',',
            header="time,pos,vel,torque", comments="")
    t_sim,x_sim,u_sim = usingFelixSimulator(mpar,traj_path,options["Q"],[options["R"]])
    plt.plot(t_sim,x_sim)
    plt.plot(t_sim,u_sim, color = "purple")

    if funnel_comparison:
        funnel_path_dirtrel = "data/simple_pendulum/funnels/SosfunnelDIRTREL.csv"
        funnel_path_dirtran = "data/simple_pendulum/funnels/SosfunnelDIRTRAN.csv"
        traj_path_dirtrel = "data/simple_pendulum/dirtrel/trajectory.csv"
        traj_path_dirtran = "data/simple_pendulum/dirtran/trajectory.csv"

        print("Volume of DIRTREL funnel: ",roaVolComputation(mpar,options,traj_path_dirtrel, funnel_path = funnel_path_dirtrel, time_out = False))
        print("Volume of DIRTRAN funnel: ",roaVolComputation(mpar,options,traj_path_dirtran, funnel_path = funnel_path_dirtran, time_out = False))
        funnel2DComparison(funnel_path_dirtran, funnel_path_dirtrel, traj_path_dirtran, traj_path_dirtrel, "")
        plt.show()
    else:
        plt.show()