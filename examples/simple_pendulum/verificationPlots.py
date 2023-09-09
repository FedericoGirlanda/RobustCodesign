import matplotlib as mpl
mpl.use("WebAgg")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as plt
import numpy as np

from simple_pendulum.controllers.tvlqr.roa.plot import TVrhoVerification, plotFunnel3d
from simple_pendulum.utilities.process_data import prepare_trajectory

from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.controllers.tvlqr.tvlqr import TVLQRController

algorithms =["RTCD"] #,"RTCD"]
verificationSamples = 100
verifiedKnot = 8
ticksSize = 40
fontSize = 40

# #########################
# # Simulatied verification
# #########################

for algorithm in algorithms:
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

#     print("Verifying "+algorithm+" results...")
#     print("The optimal m is: ", m_opt)
#     print("The optimal l is: ", l_opt)
#     print("The optimal Q is: ", Q_opt)
#     print("The optimal R is: ", R_opt)

#     # verification
#     optimized_data_dict = prepare_trajectory(optimized_traj_path)
#     pendulum_par = {"l": l_opt,
#             "m": m_opt,
#             "b": 0.1, 
#             "g": 9.81,
#             "cf": 0.0,
#             "tl": 2.5}
#     pendulum_plant = PendulumPlant(mass=pendulum_par["m"],
#                             length=pendulum_par["l"],
#                             damping=pendulum_par["b"],
#                             gravity=pendulum_par["g"],
#                             coulomb_fric=pendulum_par["cf"],
#                             inertia=None,
#                             torque_limit=pendulum_par["tl"])
#     controller = TVLQRController(data_dict=optimized_data_dict, mass=pendulum_par["m"], length=pendulum_par["l"],
#                                 damping=pendulum_par["b"], gravity=pendulum_par["g"],
#                                 torque_limit=pendulum_par["tl"])
#     controller.set_costs(Q_opt, R_opt)
#     controller.set_goal([np.pi,0])
#     (ax_ellipse, ax_traj) = TVrhoVerification(pendulum_plant,controller,optimized_funnel_path, optimized_traj_path,verificationSamples,verifiedKnot, fontSize=fontSize, ticksSize=ticksSize)

################################
# Real experimental verification
################################

# load trajectories from csv file
trajDirtrandist_path = "results/simple_pendulum/realExperiments/167329dirtrandist/data_measured.csv"
trajRTCdist_path = "results/simple_pendulum/realExperiments/167329rtcdist/data_measured.csv"
trajRTCDdist_path = "results/simple_pendulum/realExperiments/167329rtcddist/data_measured.csv"
trajDirtran_path = "results/simple_pendulum/realExperiments/167329dirtran/data_measured.csv"
trajRTC_path = "results/simple_pendulum/realExperiments/167329rtc/data_measured.csv"
trajRTCD_path = "results/simple_pendulum/realExperiments/167329rtcd/data_measured.csv"

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
trajDirtran = np.loadtxt(trajDirtran_path, skiprows=1, delimiter=",")
dirtran_time_list = trajDirtran.T[0].T  
dirtran_pos_list = trajDirtran.T[1].T  
dirtran_vel_list = trajDirtran.T[2].T  
dirtran_tau_list = trajDirtran.T[3].T   
trajOptRTC = np.loadtxt(trajRTC_path, skiprows=1, delimiter=",")
rtc_time_list = trajOptRTC.T[0].T  
rtc_pos_list  = trajOptRTC.T[1].T  
rtc_vel_list  = trajOptRTC.T[2].T  
rtc_tau_list  = trajOptRTC.T[3].T
trajOptRTCD = np.loadtxt(trajRTCD_path, skiprows=1, delimiter=",")
rtcd_time_list = trajOptRTCD.T[0].T  
rtcd_pos_list  = trajOptRTCD.T[1].T  
rtcd_vel_list  = trajOptRTCD.T[2].T  
rtcd_tau_list  = trajOptRTCD.T[3].T

# Plotting the real traj in the funnel
ax_traj, nominal = plotFunnel3d(optimized_funnel_path, optimized_traj_path,fontSize=fontSize,ticksSize=ticksSize)
dirtran, =ax_traj.plot(dirtrandist_time_list[0:], dirtrandist_pos_list[0:], dirtrandist_vel_list[0:], label = "DIRTRAN", color = "C3", linewidth = "0.7")
#opt1, = ax_traj.plot(rtcdist_time_list[0:], rtcdist_pos_list[0:], rtcdist_vel_list[0:], label = "RTC", color = "C1", linewidth = "0.3")
opt2, = ax_traj.plot(rtcddist_time_list[0:], rtcddist_pos_list[0:], rtcddist_vel_list[0:], label = "RTC-D", color = "C2", linewidth = "0.7")
#ax_traj.legend(handles = [dirtran, opt1, opt2], fontsize = fontSize)
ax_traj.legend(handles = [nominal,dirtran, opt2], fontsize = fontSize)

# # Plot the not disturbed case results
# fig, axs = plt.subplots(3,1, figsize=(4, 5))
# axs[0].plot(dirtran_time_list, dirtran_pos_list, label = "DIRTRAN", color = "C0", linewidth = "0.3")
# axs[0].plot(rtc_time_list, rtc_pos_list, label = "RTC", color = "C1", linewidth = "0.3")
# axs[0].plot(rtcd_time_list, rtcd_pos_list, label = "RTCD", color = "C2", linewidth = "0.3")
# axs[0].legend(loc = "center right", fontsize = fontSize)
# axs[0].set_ylabel(r'$\theta$'+" [rad]", fontsize = fontSize)
# axs[0].grid(True)
# axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[1].plot(dirtran_time_list, dirtran_vel_list, color = "C0", label = "DIRTRAN", linewidth = "0.3")
# axs[1].plot(rtc_time_list, rtc_vel_list, color = "C1", label = "RTC", linewidth = "0.3")
# axs[1].plot(rtcd_time_list, rtcd_vel_list, color = "C2", label = "RTCD", linewidth = "0.3")
# axs[1].legend(loc = "upper right", fontsize = fontSize)
# axs[1].set_ylabel(r'$\dot \theta$'+" [rad/s]", fontsize = fontSize)
# axs[1].grid(True)
# axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[2].plot(dirtran_time_list, dirtran_tau_list, label = "DIRTRAN", color = "C0", linewidth = "0.3")
# axs[2].plot(rtc_time_list, rtc_tau_list, label = "RTC", color = "C1", linewidth = "0.3")
# axs[2].plot(rtcd_time_list, rtcd_tau_list, label = "RTCD", color = "C2", linewidth = "0.3")
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].set_xlabel("time [s]", fontsize=fontSize)
# axs[2].set_ylabel("u [Nm]", fontsize=fontSize)
# axs[2].grid(True)

# fig, axs = plt.subplots(3,1, figsize=(4, 5))
# axs[0].plot(dirtrandist_time_list, dirtrandist_pos_list, label = "DIRTRAN", color = "C0", linewidth = "0.3")
# axs[0].plot(rtcdist_time_list, rtcdist_pos_list, label = "RTC", color = "C1", linewidth = "0.3")
# axs[0].plot(rtcddist_time_list, rtcddist_pos_list, label = "RTCD", color = "C2", linewidth = "0.3")
# axs[0].legend(loc = "center right", fontsize = fontSize)
# axs[0].set_ylabel(r'$\theta$'+" [rad]", fontsize = fontSize)
# axs[0].grid(True)
# axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[1].plot(dirtrandist_time_list, dirtrandist_vel_list, color = "C0", label = "DIRTRAN", linewidth = "0.3")
# axs[1].plot(rtcdist_time_list, rtcdist_vel_list, color = "C1", label = "RTC", linewidth = "0.3")
# axs[1].plot(rtcddist_time_list, rtcddist_vel_list, color = "C2", label = "RTCD", linewidth = "0.3")
# axs[1].legend(loc = "upper right", fontsize = fontSize)
# axs[1].set_ylabel(r'$\dot \theta$'+" [rad/s]", fontsize = fontSize)
# axs[1].grid(True)
# axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[0].hlines(1.5,0.5,0.7,colors=["black"], linewidth = 0.4)
# axs[1].hlines(0,0.5,0.7,colors=["black"], linewidth = 0.4)
# axs[0].vlines(np.linspace(0.5,0.7,2),-1.5,4.5,colors=["black"], linewidth = 0.4)
# axs[1].vlines(np.linspace(0.5,0.7,2),-6.5,6.5,colors=["black"], linewidth = 0.4)
# axs[2].plot(dirtrandist_time_list, dirtrandist_tau_list, label = "DIRTRAN", color = "C0", linewidth = "0.3")
# axs[2].plot(rtcdist_time_list, rtcdist_tau_list, label = "RTC", color = "C1", linewidth = "0.3")
# axs[2].plot(rtcddist_time_list, rtcddist_tau_list, label = "RTCD", color = "C2", linewidth = "0.3")
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[2].hlines(0,0.5,0.7,colors=["black"], linewidth = 0.4)
# axs[2].vlines(np.linspace(0.5,0.7,2),-3,3,colors=["black"], linewidth = 0.4)
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].set_xlabel("time [s]", fontsize=fontSize)
# axs[2].set_ylabel("u [Nm]", fontsize=fontSize)
# axs[2].grid(True)

plt.show()