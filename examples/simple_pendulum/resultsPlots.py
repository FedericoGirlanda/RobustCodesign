import numpy as np
import matplotlib as mlp
mlp.use("WebAgg")
import matplotlib.pyplot as plt
import time

from simple_pendulum.controllers.tvlqr.roa.utils import funnelVolume_convexHull, funnelVolume
from simple_pendulum.controllers.tvlqr.roa.plot import plotFunnel, rhoComparison, funnel2DComparison
from simple_pendulum.utilities.process_data import prepare_trajectory, saveFunnel

oldOpt = False
algorithm = "RTC" #RTCD

traj_path1 = "data/simple_pendulum/dirtran/trajectory.csv"
funnel_path1 = "data/simple_pendulum/funnels/SosfunnelDIRTRAN.csv"
label1 = "DIRTRAN"

if algorithm == "RTC" and not oldOpt:
    traj_path2 = "results/simple_pendulum/optCMAES_167329/trajectoryOptimal_CMAES.csv" 
    funnel_path2 = "results/simple_pendulum/optCMAES_167329/SosfunnelOptimal_CMAES.csv"
    label2 = "RTC"
elif algorithm == "RTCD" and not oldOpt:
    traj_path2 = "results/simple_pendulum/optDesignCMAES_167329/trajectoryOptimal_CMAES.csv"
    funnel_path2 = "results/simple_pendulum/optDesignCMAES_167329/SosfunnelOptimal_CMAES.csv" 
    label2 = "RTCD"
elif algorithm == "RTC" and oldOpt:
    traj_path1 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /DIRTRAN/trajectory_dirtran.csv"
    funnel_path1 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /DIRTRAN/SosfunnelDIRTRAN.csv"
    traj_path2 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/VOpt/DIRTRAN/trajectoryOptimal_CMAES.csv" 
    funnel_path2 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/VOpt/DIRTRAN/SosfunnelOptimal_CMAES.csv"
    label2 = "RTC"
elif algorithm == "RTCD" and oldOpt:
    traj_path1 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /DIRTRAN/trajectory_dirtran.csv"
    funnel_path1 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /DIRTRAN/SosfunnelDIRTRAN.csv"
    traj_path2 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/DesignOpt/Vdirtran/trajectoryOptimal_CMAES.csv" 
    funnel_path2 = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/DesignOpt/Vdirtran/SosfunnelOptimal_CMAES.csv"
    label2 = "RTCD"
else:
    print("Erroneous algorithm label.")
    assert False

#########################
# Trajectories comparison
#########################

# load trajectories
trajectory1 = np.loadtxt(traj_path1, skiprows=1, delimiter=",")
T1 = trajectory1.T[0].T
X1 = [trajectory1.T[1].T, trajectory1.T[2].T]
U1 = trajectory1.T[3].T

trajectory2 = np.loadtxt(traj_path2, skiprows=1, delimiter=",")
T2 = trajectory2.T[0].T
X2 = [trajectory2.T[1].T, trajectory2.T[2].T]
U2 = trajectory2.T[3].T

# comparison plots
ticksSize = 30
fontSize = 30
fig, axs = plt.subplots(2,1, figsize=(12, 10))
axs[0].plot(T1,X1[0], label = label1)
axs[0].plot(T2,X2[0], label = label2)
axs[0].legend(loc = "lower right", fontsize = fontSize)
axs[0].set_ylabel(r'$\theta$'+" [rad]", fontsize = fontSize)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[1].plot(T1,U1, label = label1)
axs[1].plot(T2,U2, label = label2)
axs[1].legend(loc = "upper right", fontsize = fontSize)
axs[1].set_ylabel(r'$u$'+" [Nm]", fontsize = fontSize)
axs[1].set_xlabel("time [s]", fontsize = fontSize)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)

# ####################
# # Funnels comparison
# ####################

# # Volume computation
# start = time.time()
# vol1 = funnelVolume_convexHull(funnel_path1, traj_path1)
# vol2 = funnelVolume_convexHull(funnel_path2, traj_path2)
# print("The convex hull volume of the "+ label1 +" funnel is", vol1)
# print("The convex hull volume of the "+ label2 +" funnel is", vol2)
# print(f"The calculation of the convex volume took: {int(time.time()-start)} seconds")

vol1 = 14.83
vol2 = 47.2 #57.84

# Funnel plots
ticksSize = 30
fontSize = 30
volumes = (np.round(vol1,2),np.round(vol2,2))
funnel2DComparison(funnel_path1,funnel_path2,traj_path1,traj_path2,volumes = volumes, fontSize = fontSize, ticksSize = ticksSize)

# ###################
# # Optimal cost plot
# ###################

# if oldOpt:
#     fullcoopt_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/DesignOpt/Vdirtran/fullCoopt_CMAES.csv" 
#     inner_coopt_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/DesignOpt/Vdirtran/CooptData_CMAES.csv"
#     coopt_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/VOpt/DIRTRAN/CooptData_CMAES.csv" 
#     evolution_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/VOpt/DIRTRAN/outcmaes/evolution.csv"
#     fullcoopt_data = np.loadtxt(fullcoopt_path, delimiter=",",skiprows=1)
#     coopt_data = np.loadtxt(coopt_path, delimiter=",",skiprows=1) 
#     inner_coopt_data = np.loadtxt(inner_coopt_path, delimiter=",",skiprows=1)
#     evolution_data = np.loadtxt(evolution_path, delimiter=",")
# else:
#     fullcoopt_path = "results/simple_pendulum/optDesignCMAES_167329/fullCoopt_CMAES.csv" 
#     inner_coopt_path = "results/simple_pendulum/optDesignCMAES_167329/CooptData_CMAES.csv"
#     coopt_path = "results/simple_pendulum/optCMAES_167329/CooptData_CMAES.csv"  
#     evolution_path = "results/simple_pendulum/optCMAES_167329/outcmaes/evolution.csv"
#     fullcoopt_data = np.loadtxt(fullcoopt_path, delimiter=",")
#     coopt_data = np.loadtxt(coopt_path, delimiter=",")
#     inner_coopt_data = np.loadtxt(inner_coopt_path, delimiter=",")
#     evolution_data = np.loadtxt(evolution_path, delimiter=",")
# V = np.sort(-np.array(coopt_data.T[-1]))
# V_full = np.sort(-np.array(fullcoopt_data.T[-1]))
# m_var_full = []
# l_var_full = []
# r_var_full = []
# ml_var_full = []
# for i in range(len(V_full)):
#     idx = np.where(-np.array(fullcoopt_data.T[-1]) == V_full[i])
#     m_var_full = np.append(m_var_full, fullcoopt_data.T[0][idx])
#     l_var_full = np.append(l_var_full, fullcoopt_data.T[1][idx])
#     r_var_full = np.append(r_var_full, fullcoopt_data.T[4][idx])
#     ml_var_full = np.append(ml_var_full, fullcoopt_data.T[0][idx][0]*fullcoopt_data.T[1][idx][0])
# r_var = []
# q11_var = []
# q22_var = []
# for i in range(len(V)):
#     idx = np.where(-np.array(coopt_data.T[-1]) == V[i])
#     r_var = np.append(r_var, coopt_data.T[2][idx])
#     q11_var = np.append(q11_var, coopt_data.T[0][idx])
#     q22_var = np.append(q22_var, coopt_data.T[1][idx])

# ticksSize = 30
# fontSize = 30
# cost = []
# V_inner = np.sort(-np.array(inner_coopt_data.T[-1]))
# for i in range(len(V_full)):
#     if V_full[i] > 0.1:
#         for j in range(int(len(V_inner)/len(V_full))):
#             cost = np.append(cost, V_full[i])
# cost = np.append([15.75], cost)
# eval_full = np.array([i for i in range(len(cost))])
# costRTC = []
# iterations = np.array(evolution_data.T[-1])
# V_ev = np.sort(-np.array(evolution_data.T[3]))
# for i in range(len(V_ev)):
#     if i == 0:
#         j_prev = 0
#     else:
#         j_prev = int(iterations[i-1])
#     for j in range(j_prev, int(iterations[i])):
#         costRTC = np.append(costRTC, V_ev[i])
# costRTC = np.append([cost[0]], costRTC)
# eval = np.array([i for i in range(len(costRTC))])
# plt.figure(figsize=(12,9))
# plt.plot(eval_full,cost, label = "RTCD", color = "C1")
# plt.plot(eval,costRTC, label = "RTC", color = "C0")
# plt.grid(True)
# plt.xlabel("Evaluations", fontsize=fontSize)
# plt.ylabel("ROA Volume", fontsize=fontSize)
# plt.xticks(fontsize=ticksSize)
# plt.yticks(fontsize=ticksSize)
# plt.rc('legend', fontsize=fontSize)
# plt.legend(loc = "upper left")

# plt.figure(figsize=(11,9))
# plt.scatter(V_full, np.array(ml_var_full)*9.81, label = r"$mgl_{RTCD}\ [Nm]$", color = "C1", marker = ".",s=150)
# plt.hlines(2.5,0,V_full.max(),colors=["C1"], linestyles="--")
# plt.text(60,2.52,r"$\tau_{lim}$",color = "C1", fontsize=fontSize)
# plt.scatter(V, r_var, label = r"$r_{RTC}$", color = "C0", marker = ".", s=150)
# plt.grid(True)
# plt.xlabel("ROA Volume", fontsize=fontSize)
# plt.ylabel("Optimization Parameters", fontsize=fontSize)
# plt.xticks(fontsize=ticksSize)
# plt.yticks(fontsize=ticksSize)
# plt.xlim(0,V_full.max())
# plt.ylim(-1,8)
# plt.rc('legend', fontsize=fontSize)
# plt.legend(loc = "upper left") #loc = (2,1))

# # 3d scatter heatmap
# from matplotlib import cm
# cube_l = int(np.cbrt(len(V)))
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# i = 0
# j = 0
# k = 0
# for l in range(cube_l*cube_l*cube_l):
#     ax.scatter(q11_var[l], q22_var[l], r_var[l], c = V[l], cmap=cm.Oranges, s=35, vmin=np.array(V).min(), vmax=np.array(V).max())
#     i += 1
#     if i%cube_l == 0:
#         i = 0
#         j += 1
#         if j%cube_l == 0:
#             j = 0
#             k += 1
# #ax.set_title("3D Heatmap")
# ax.set_xlabel('q11')
# ax.set_ylabel('q22')
# ax.set_zlabel('r')
# color_map = cm.ScalarMappable(cmap=cm.Oranges)
# color_map.set_array(np.array(V))
# plt.colorbar(color_map, ax = ax, location = "left")

########################
# Real experiments plots
########################

trajDirtran_path = "results/simple_pendulum/realExperiments/167329dirtran/data_measured.csv"
trajRTC_path = "results/simple_pendulum/realExperiments/167329rtc/data_measured.csv"
trajRTCD_path = "results/simple_pendulum/realExperiments/167329rtcd/data_measured.csv"
trajDirtrandist_path = "results/simple_pendulum/realExperiments/167329dirtrandist/data_measured.csv"
trajRTCdist_path = "results/simple_pendulum/realExperiments/167329rtcdist/data_measured.csv"
trajRTCDdist_path = "results/simple_pendulum/realExperiments/167329rtcddist/data_measured.csv"

# load trajectories from csv file
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

# Plot the not disturbed case results
ticksSize = 30
fontSize = 30

# fig, axs = plt.subplots(3,1, figsize=(14, 22))
# axs[0].plot(dirtran_time_list, dirtran_pos_list, label = "DIRTRAN", color = "C0")
# axs[0].plot(rtc_time_list, rtc_pos_list, label = "RTC", color = "C1")
# axs[0].plot(rtcd_time_list, rtcd_pos_list, label = "RTCD", color = "C2")
# axs[0].legend(loc = "center right", fontsize = fontSize)
# axs[0].set_ylabel(r'$\theta$'+" [rad]", fontsize = fontSize)
# axs[0].grid(True)
# axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[1].plot(dirtran_time_list, dirtran_vel_list, color = "C0", label = "DIRTRAN")
# axs[1].plot(rtc_time_list, rtc_vel_list, color = "C1", label = "RTC")
# axs[1].plot(rtcd_time_list, rtcd_vel_list, color = "C2", label = "RTCD")
# axs[1].legend(loc = "upper right", fontsize = fontSize)
# axs[1].set_ylabel(r'$\dot \theta$'+" [rad/s]", fontsize = fontSize)
# axs[1].grid(True)
# axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[2].plot(dirtran_time_list, dirtran_tau_list, label = "DIRTRAN", color = "C0")
# axs[2].plot(rtc_time_list, rtc_tau_list, label = "RTC", color = "C1")
# axs[2].plot(rtcd_time_list, rtcd_tau_list, label = "RTCD", color = "C2")
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].set_xlabel("time [s]", fontsize=fontSize)
# axs[2].set_ylabel("u [Nm]", fontsize=fontSize)
# axs[2].grid(True)

# fig, axs = plt.subplots(3,1, figsize=(14, 22))
# axs[0].plot(dirtrandist_time_list, dirtrandist_pos_list, label = "DIRTRAN", color = "C0")
# axs[0].plot(rtcdist_time_list, rtcdist_pos_list, label = "RTC", color = "C1")
# axs[0].plot(rtcddist_time_list, rtcddist_pos_list, label = "RTCD", color = "C2")
# axs[0].legend(loc = "center right", fontsize = fontSize)
# axs[0].set_ylabel(r'$\theta$'+" [rad]", fontsize = fontSize)
# axs[0].grid(True)
# axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[1].plot(dirtrandist_time_list, dirtrandist_vel_list, color = "C0", label = "DIRTRAN")
# axs[1].plot(rtcdist_time_list, rtcdist_vel_list, color = "C1", label = "RTC")
# axs[1].plot(rtcddist_time_list, rtcddist_vel_list, color = "C2", label = "RTCD")
# axs[1].legend(loc = "upper right", fontsize = fontSize)
# axs[1].set_ylabel(r'$\dot \theta$'+" [rad/s]", fontsize = fontSize)
# axs[1].grid(True)
# axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[0].hlines(1.5,0.5,0.7,colors=["black"])
# axs[1].hlines(0,0.5,0.7,colors=["black"])
# axs[0].vlines(np.linspace(0.5,0.7,2),-1.5,4.5,colors=["black"])
# axs[1].vlines(np.linspace(0.5,0.7,2),-6.5,6.5,colors=["black"])
# axs[2].plot(dirtrandist_time_list, dirtrandist_tau_list, label = "DIRTRAN", color = "C0")
# axs[2].plot(rtcdist_time_list, rtcdist_tau_list, label = "RTC", color = "C1")
# axs[2].plot(rtcddist_time_list, rtcddist_tau_list, label = "RTCD", color = "C2")
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[2].hlines(0,0.5,0.7,colors=["black"])
# axs[2].vlines(np.linspace(0.5,0.7,2),-3,3,colors=["black"])
# axs[2].legend(loc = "upper right", fontsize = fontSize)
# axs[2].set_xlabel("time [s]", fontsize=fontSize)
# axs[2].set_ylabel("u [Nm]", fontsize=fontSize)
# axs[2].grid(True)

# TODO: state-space plots

plt.show()