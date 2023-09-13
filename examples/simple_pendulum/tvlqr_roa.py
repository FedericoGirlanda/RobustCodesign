import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter

from simple_pendulum.trajectory_optimization.direct_collocation.direct_collocation import DirectCollocationCalculator
from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.simulation.simulation import Simulator
from simple_pendulum.controllers.tvlqr.tvlqr import TVLQRController
from simple_pendulum.controllers.lqr.lqr_controller import LQRController
from simple_pendulum.utilities.process_data import prepare_trajectory, saveFunnel
from simple_pendulum.controllers.tvlqr.roa.probabilistic import TVprobRhoComputation
from simple_pendulum.controllers.tvlqr.roa.sos import TVsosRhoComputation
from simple_pendulum.controllers.lqr.roa.sos import SOSequalityConstrained
from simple_pendulum.controllers.tvlqr.roa.plot import plotFunnel, TVrhoVerification,\
                                                       plotFunnel3d, rhoComparison
from simple_pendulum.controllers.tvlqr.roa.utils import funnelVolume_convexHull

# Pendulum parameters
mass = 0.57288
length = 0.5
damping = 0.15
gravity = 9.81
coulomb_fric = 0.0
torque_limit = 2

# Swing-up parameters
x0 = [0.0, 0.0]
goal = [np.pi, 0.0]

# Direct collocation parameters, N is chosen also to be the number of knot points
N = 60
max_dt = 0.05

# Number of simulations for the simulation-based estimation method
nSimulations = 100

# Plots parameters
fontSize = 40
ticksSize = 40

#######################################################
# Compute the nominal trajectory via direct collocation
#######################################################

dircal = DirectCollocationCalculator()
dircal.init_pendulum(mass=mass,
                     length=length,
                     damping=damping,
                     gravity=gravity,
                     torque_limit=torque_limit)
x_trajectory, dircol, result = dircal.compute_trajectory(N=N,
                                                         max_dt=max_dt,
                                                         start_state=x0,
                                                         goal_state=goal)
T, X, XD, U = dircal.extract_trajectory(x_trajectory, dircol, result, N=N)

# save trajectory
log_dir = "data/simple_pendulum/dircol"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

traj_data = np.vstack((T, X, XD, U)).T
traj_path = os.path.join(log_dir, "trajectory.csv" )
np.savetxt(traj_path, traj_data, delimiter=',',
           header="time,pos,vel,torque", comments="")

##################################
# RoA computation and Funnels plot
##################################

# load trajectory
data_dict = prepare_trajectory(traj_path)
trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
time = trajectory.T[0].T
dt = time[1]-time[0]
x0_traj = [trajectory.T[1].T, trajectory.T[2].T]

pendulum = PendulumPlant(mass=mass,
                         length=length,
                         damping=damping,
                         gravity=gravity,
                         coulomb_fric=coulomb_fric,
                         inertia=None,
                         torque_limit=torque_limit)

sim = Simulator(plant=pendulum)

controller = TVLQRController(data_dict=data_dict, mass=mass, length=length,
                             damping=damping, gravity=gravity,
                             torque_limit=torque_limit)

# Taking the finals values of S and rho from the invariant case, SOS method has been chosen
(rhof, Sf) = SOSequalityConstrained(pendulum,LQRController(mass=mass,
                                                            length=length,
                                                            damping=damping,
                                                            gravity=gravity,
                                                            torque_limit=torque_limit))
controller.set_goal(goal)
S = controller.tvlqr.S

# Application of the algorithm for time-varying RoA estimation
parser = argparse.ArgumentParser(description=''' Time-varying RoA estimation: 
                    torque-limited Simple Pendulum''', formatter_class=RawTextHelpFormatter)
method = parser.add_mutually_exclusive_group(required=True)    
method.add_argument("-prob", action='store_true',
                            help="Probabilistic method")
method.add_argument("-sos", action='store_true',
                            help="SOS method") 
method.add_argument("-compare", action='store_true',
                            help="Compare the two methods")   
args, unknown = parser.parse_known_args() 
if args.prob:
    funnel_path = f"data/simple_pendulum/funnels/Probfunnel{max_dt}-{N}.csv"
    if not os.path.exists(funnel_path):
        (rho, ctg) = TVprobRhoComputation(pendulum, controller, x0_traj, time, N, nSimulations, rhof)
        print("The final rho is: "+str(rho))
        saveFunnel(rho, S, time, max_dt, N, "Prob")
    ax, nominal = plotFunnel3d(funnel_path, traj_path) # Plotting the 3d funnel
    ax.legend(handles = [nominal], labels = ["Nominal trajectory"], loc = 'upper right', fontsize = fontSize)
    ax.set_title("3d Prob Funnel", fontsize = fontSize)
if args.sos:
    funnel_path = f"data/simple_pendulum/funnels/Sosfunnel{max_dt}-{N}.csv"
    if not os.path.exists(funnel_path):
        (rho, S) = TVsosRhoComputation(pendulum, controller, time, N, rhof)
        print("The final rho is: "+str(rho))
        saveFunnel(rho, S, time, max_dt, N, "Sos")
    ax, nominal = plotFunnel3d(funnel_path, traj_path) # Plotting the 3d funnel
    ax.legend(handles = [nominal], labels = ["Nominal trajectory"], loc = 'upper right', fontsize = fontSize)
    ax.set_title("3d SOS Funnel", fontsize = fontSize)
if args.compare:
    funnelSos_path = f"data/simple_pendulum/funnels/Sosfunnel{max_dt}-{N}.csv"
    funnelProb_path = f"data/simple_pendulum/funnels/Probfunnel{max_dt}-{N}.csv"
    if not os.path.exists(funnelProb_path):
        (rho_prob, ctg) = TVprobRhoComputation(pendulum, controller, x0_traj, time, N, nSimulations, rhof)
        saveFunnel(rho, S, time, max_dt, N, "Prob")
    if not os.path.exists(funnelSos_path):   
        (rho_sos, S) = TVsosRhoComputation(pendulum, controller, time, N, rhof)
        saveFunnel(rho, S, time, max_dt, N, "Sos")
    ax_comp = plotFunnel(funnelSos_path, traj_path, fontSize= fontSize, ticksSize= ticksSize, noTraj = True) # Comparison plots
    plotFunnel(funnelProb_path, traj_path, ax=ax_comp, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
    rhoComparison(funnelSos_path, funnelProb_path)

plt.show()