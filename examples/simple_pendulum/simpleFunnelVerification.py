import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import random

from simple_pendulum.controllers.tvlqr.roa.plot import get_ellipse_patch, sample_from_ellipsoid, quad_form, getEllipseFromCsv, TVrhoVerification, getEllipseContour
from simple_pendulum.utilities.process_data import prepare_trajectory
from simulatorComparison import usingDrakeSimulator, usingFelixSimulator

from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.controllers.tvlqr.tvlqr import TVLQRController

realExpPlot = True #False
algorithm = "RTC" #RTCD
verificationSamples = 0#1000
verifiedKnot = 8#18
ticksSize = 30
fontSize = 30

if algorithm == "RTC":
    optimized_traj_path = "results/simple_pendulum/optCMAES_167329/trajectoryOptimal_CMAES.csv" 
    optimized_funnel_path = "results/simple_pendulum/optCMAES_167329/SosfunnelOptimal_CMAES.csv"
    optimal_pars_path = "results/simple_pendulum/optCMAES_167329/CooptData_CMAES.csv"
    optimal_label = "RTC"

    # Load optimal params
    controller_data = np.loadtxt(optimal_pars_path, skiprows=1, delimiter=",")
    max_idx = np.where(-controller_data.T[3] == max(-controller_data.T[3]))[0][0]
    m_opt = 0.7
    l_opt = 0.4
    Q_opt = np.diag([controller_data[max_idx,0],controller_data[max_idx,1]])
    R_opt = [controller_data[max_idx,2]]
elif algorithm == "RTCD":
    optimized_traj_path = "results/simple_pendulum/optDesignCMAES_167329/trajectoryOptimal_CMAES.csv" 
    optimized_funnel_path = "results/simple_pendulum/optDesignCMAES_167329/SosfunnelOptimal_CMAES.csv"
    optimal_pars_path = "results/simple_pendulum/optDesignCMAES_167329/fullCoopt_CMAES.csv"
    optimal_label = "RTCD"

    # Load optimal params
    controller_data = np.loadtxt(optimal_pars_path, skiprows=1, delimiter=",")
    max_idx = np.where(-controller_data.T[5] == max(-controller_data.T[5]))[0][0]
    m_opt = controller_data[max_idx,0]
    l_opt = controller_data[max_idx,1]
    Q_opt = np.diag([controller_data[max_idx,2],controller_data[max_idx,3]]) 
    R_opt = [controller_data[max_idx,4]]
else:
    print("Erroneous algorithm label.")
    assert False

print("The optimal m is: ", m_opt)
print("The optimal l is: ", l_opt)
print("The optimal Q is: ", Q_opt)
print("The optimal R is: ", R_opt)

# pendulum parameters
opt_mpar = {"l": l_opt,
        "m": m_opt,
        "b": 0.1, 
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}

# loading the trajectories
optimized_data_dict = prepare_trajectory(optimized_traj_path)
pendulum_par = {"l": l_opt,
        "m": m_opt,
        "b": 0.1, 
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}
pendulum_plant = PendulumPlant(mass=pendulum_par["m"],
                         length=pendulum_par["l"],
                         damping=pendulum_par["b"],
                         gravity=pendulum_par["g"],
                         coulomb_fric=pendulum_par["cf"],
                         inertia=None,
                         torque_limit=pendulum_par["tl"])
controller = TVLQRController(data_dict=optimized_data_dict, mass=pendulum_par["m"], length=pendulum_par["l"],
                            damping=pendulum_par["b"], gravity=pendulum_par["g"],
                            torque_limit=pendulum_par["tl"])
controller.set_costs(Q_opt, R_opt)
controller.set_goal([np.pi,0])
(ax_ellipse, ax_traj) = TVrhoVerification(pendulum_plant,controller,optimized_funnel_path, optimized_traj_path,verificationSamples,verifiedKnot, fontSize=fontSize, ticksSize=ticksSize)

if realExpPlot:
    trajDirtrandist_path = "results/simple_pendulum/realExperiments/167329dirtrandist/data_measured.csv"
    trajRTCdist_path = "results/simple_pendulum/realExperiments/167329rtcdist/data_measured.csv"
    trajRTCDdist_path = "results/simple_pendulum/realExperiments/167329rtcddist/data_measured.csv"

    trajDirtrandist = np.loadtxt(trajDirtrandist_path, skiprows=1, delimiter=",")
    dirtrandist_time_list = trajDirtrandist.T[0].T  
    dirtrandist_pos_list = trajDirtrandist.T[1].T  
    dirtrandist_vel_list = trajDirtrandist.T[2].T  
    dirtrandist_tau_list = trajDirtrandist.T[3].T   

    trajOptRTCdist = np.loadtxt(trajRTCdist_path, skiprows=1, delimiter=",")
    rtcdist_time_list = trajOptRTCdist.T[0].T  
    rtcdist_pos_list  = trajOptRTCdist.T[1].T  
    rtcdist_vel_list  = trajOptRTCdist.T[2].T  
    rtcdist_tau_list  = trajOptRTCdist.T[3].T

    trajOptRTCDdist = np.loadtxt(trajRTCDdist_path, skiprows=1, delimiter=",")
    rtcddist_time_list = trajOptRTCDdist.T[0].T  
    rtcddist_pos_list  = trajOptRTCDdist.T[1].T  
    rtcddist_vel_list  = trajOptRTCDdist.T[2].T  
    rtcddist_tau_list  = trajOptRTCDdist.T[3].T

    # plotting the real traj in the funnel
    simKnot = 0 #verifiedKnot*7 - 8
    dirtran, =ax_traj.plot(dirtrandist_time_list[simKnot:], dirtrandist_pos_list[simKnot:], dirtrandist_vel_list[simKnot:], label = "DIRTRAN", color = "C0")
    opt1, = ax_traj.plot(rtcdist_time_list[simKnot:], rtcdist_pos_list[simKnot:], rtcdist_vel_list[simKnot:], label = "RTC", color = "C1")
    opt2, = ax_traj.plot(rtcddist_time_list[simKnot:], rtcddist_pos_list[simKnot:], rtcddist_vel_list[simKnot:], label = "RTCD", color = "C2")
    ax_traj.legend(handles = [dirtran, opt1, opt2], fontsize = fontSize)

plt.show()