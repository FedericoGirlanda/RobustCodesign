import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("WebAgg")
import time

from simple_pendulum.trajectory_optimization.dirtran.dirtranTrajOpt import DirtranTrajectoryOptimization

save_dir = "data/simple_pendulum/dirtran/trajectory.csv"

# Pendulum parameters
mpar = {"l": 0.4, 
        "m": 0.7,
        "b": 0.1,
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}

# Direct transcription parameters
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

# Direct transcription execution
start = time.time()
dirtran = DirtranTrajectoryOptimization(mpar, options)
T,X,U = dirtran.ComputeTrajectory()
print("Duration(mins): ", int((time.time()-start)/60))
print("Duration(secs): ", (time.time()-start))

# Trajectory visualization
ticksSize = 10
fontSize = 10
labels = [r"$\theta$", r"$\dot \theta$", r"$u$"]
fig, axs = plt.subplots(3,1, figsize=(18, 9))
axs[0].plot(T,X[0], label = labels[0], color = "blue")
axs[0].legend(loc = "upper right", fontsize = fontSize)
axs[0].set_ylabel(labels[0]+" [rad]", fontsize = fontSize)
axs[0].set_xlabel("time [s]", fontsize = fontSize)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[1].plot(T,X[1], label = labels[1], color = "orange")
axs[1].legend(loc = "upper right", fontsize = fontSize)
axs[1].set_ylabel(labels[1]+" [rad/s]", fontsize = fontSize)
axs[1].set_xlabel("time [s]", fontsize = fontSize)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[2].plot(T,U[0], label = labels[2], color = "purple")
axs[2].legend(loc = "upper right", fontsize = fontSize)
axs[2].set_ylabel(labels[2]+" [Nm]", fontsize = fontSize)
axs[2].set_xlabel("time [s]", fontsize = fontSize)
axs[2].grid(True)
axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
plt.show()