import numpy as np
import matplotlib as mlp
mlp.use("WebAgg")
import matplotlib.pyplot as plt
import time

from cart_pole.controllers.tvlqr.RoAest.plot import plotFunnel, plotRhoEvolution, plotFunnel3d
from cart_pole.controllers.tvlqr.RoAest.utils import funnelVolume_convexHull
from cart_pole.controllers.tvlqr.RoAest.utils import getEllipseFromCsv
from cart_pole.controllers.lqr.RoAest.plots import plot_ellipse, get_ellipse_patch
from cart_pole.controllers.lqr.RoAest.utils import sample_from_ellipsoid

algorithm = "RTCD" #RTCD
indexes = (0,1) # Meaningful values (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)

traj_path1 = "data/cart_pole/dirtran/trajectory.csv"
funnel_path1 = "data/cart_pole/RoA/Probfunnel_DIRTRAN.csv"
label1 = "DIRTRAN"

if algorithm == "RTC":
    traj_path2 = "results/cart_pole/optCMAES_167332/trajectoryOptimal_CMAES.csv" 
    funnel_path2 = "results/cart_pole/optCMAES_167332/RoA_CMAES.csv"
    label2 = "RTC"
elif algorithm == "RTCD":
    traj_path2 = "results/cart_pole/optDesignCMAES_167332/trajectoryOptimal_CMAES.csv"
    funnel_path2 = "results/cart_pole/optDesignCMAES_167332/RoA_CMAES.csv" 
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
X1 = [trajectory1.T[1].T, trajectory1.T[2].T,trajectory1.T[3].T, trajectory1.T[4].T]
U1 = trajectory1.T[3].T

trajectory2 = np.loadtxt(traj_path2, skiprows=1, delimiter=",")
T2 = trajectory2.T[0].T
X2 = [trajectory2.T[1].T, trajectory2.T[2].T,trajectory2.T[3].T, trajectory2.T[4].T]
U2 = trajectory2.T[3].T

# comparison plots
ticksSize = 30
fontSize = 30
labels = [r"$x_{cart}$ [m]",r"$\theta$ [rad]",r"$\dot x_{cart}$ [m/s]",r"$\dot \theta$ [rad/s]"]
fig_test, ax_test = plt.subplots(2,2, figsize = (17, 10))
#fig_test.suptitle(f"Dynamics trajectory stabilization: simulated(blue) vs desired(orange)")
ax_test[0][0].plot(T1, X1[0])
ax_test[0][1].plot(T1, X1[1], label = "DIRTRAN")
ax_test[1][0].plot(T1, X1[2])
ax_test[1][1].plot(T1, X1[3])
ax_test[0][0].plot(T2, X2[0])
ax_test[0][1].plot(T2, X2[1], label = algorithm)
ax_test[1][0].plot(T2, X2[2])
ax_test[1][1].plot(T2, X2[3])
ax_test[0][0].hlines(np.vstack((np.ones((len(T1),1)),-np.ones((len(T1),1))))*0.3,T1[0], T1[-1])
ax_test[0][0].set_ylabel(labels[0], fontsize = fontSize)
ax_test[0][1].set_ylabel(labels[1], fontsize = fontSize)
ax_test[1][0].set_ylabel(labels[2], fontsize = fontSize)
ax_test[1][1].set_ylabel(labels[3], fontsize = fontSize)
ax_test[1][0].set_xlabel("time [s]", fontsize = fontSize)
ax_test[1][1].set_xlabel("time [s]", fontsize = fontSize)
ax_test[0][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax_test[0][1].tick_params(axis='both', which='major', labelsize=ticksSize)
ax_test[1][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax_test[1][1].tick_params(axis='both', which='major', labelsize=ticksSize)
ax_test[0][1].legend(loc = "upper right", fontsize = fontSize)

fig_test, ax_test = plt.subplots(1,1, figsize = (10, 10))
ax_test.plot(T1,U1)
ax_test.plot(T2,U2)
ax_test.hlines(np.vstack((np.ones((len(T1),1)),-np.ones((len(T1),1))))*5,T1[0], T1[-1])
ax_test.set_ylabel("u [N]", fontsize = fontSize)
ax_test.set_xlabel("time [s]", fontsize = fontSize)
ax_test.tick_params(axis='both', which='major', labelsize=ticksSize)

##########################
# Last Ellipses comparison
##########################
from matplotlib.ticker import FormatStrFormatter

traj_pathRtc = "results/cart_pole/optCMAES_167332/trajectoryOptimal_CMAES.csv" 
funnel_pathRtc = "results/cart_pole/optCMAES_167332/RoA_CMAES.csv"
traj_pathRtcd = "results/cart_pole/optDesignCMAES_167332/trajectoryOptimal_CMAES.csv"
funnel_pathRtcd = "results/cart_pole/optDesignCMAES_167332/RoA_CMAES.csv" 

fig1 = plt.figure(figsize = (12,12)) 
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(T2, X2[indexes[0]], X2[indexes[1]])
plotFunnel3d(funnel_pathRtcd, traj_pathRtcd, indexes, ax1)
# plotFunnel(funnel_pathRtcd, traj_pathRtcd, indexes)
# plotRhoEvolution(funnel_pathRtcd, traj_pathRtcd, indexes)
plt.show()
assert False

# start = time.time()
# vol1 = funnelVolume_convexHull(funnel_path1, traj_path1)
# vol2 = funnelVolume_convexHull(funnel_pathRtc, traj_pathRtc)
# vol3 = funnelVolume_convexHull(funnel_pathRtcd, traj_pathRtcd)
# print("The convex hull volume of the "+ label1 +" funnel is", vol1)
# print("The convex hull volume of the "+ label2 +" funnel is", vol2)
# print(f"The calculation of the convex volume took: {int(time.time()-start)} seconds")

(rho1, S1) = getEllipseFromCsv(funnel_path1,-1)
(rho2, S2) = getEllipseFromCsv(funnel_pathRtc,-1)
(rho3, S3) = getEllipseFromCsv(funnel_pathRtcd,-1)

ticksSize = 30
fontSize = 50
xG = [0,0,0,0]
p1 = get_ellipse_patch(indexes[0], indexes[1], xG,rho1,S1, alpha_val = 0.5,linec = "red", facec = "red") 
p2 = get_ellipse_patch(indexes[0], indexes[1], xG,rho2,S2, alpha_val = 0.5,linec = "green", facec = "green")  
p3 = get_ellipse_patch(indexes[0], indexes[1], xG,rho3,S3, alpha_val = 0.25,linec = "green", facec = "green")
p1.set(label = f"V = {np.round(vol1,2)}")
p2.set(label = f"V = {np.round(vol2,2)}")
p3.set(label = f"V = {np.round(vol3,2)}")
fig, ax = plt.subplots(figsize = (20,20))
ax.add_patch(p3)
ax.add_patch(p2)
ax.add_patch(p1)
ax.scatter(xG[indexes[0]],xG[indexes[1]], color = "black", marker="o")
ax.set_ylabel(labels[indexes[1]], fontsize = fontSize)
ax.set_xlabel(labels[indexes[0]], fontsize = fontSize)
ax.set_xlim(-0.7,0.7)
ax.set_ylim(-0.4,0.4)
ax.tick_params(axis='both', which='major', labelsize=ticksSize)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.legend(loc = "upper left", fontsize = fontSize-10)

########################
# Optimal Cost Evolution
#######################

ticksSize = 30
fontSize = 30
fullcoopt_path = "results/cart_pole/optDesignCMAES_167332/fullCooptData_CMAES.csv" 
inner_coopt_path = "results/cart_pole/optDesignCMAES_167332/CooptData_CMAES.csv"
coopt_path = "results/cart_pole/optCMAES_167332/CooptData_CMAES.csv"    
evolution_path = "results/cart_pole/optCMAES_167332/outcmaes/evolution.csv"
fullcoopt_data = np.loadtxt(fullcoopt_path, delimiter=",")
coopt_data = np.loadtxt(coopt_path, delimiter=",")
inner_coopt_data = np.loadtxt(inner_coopt_path, delimiter=",")
evolution_data = np.loadtxt(evolution_path, delimiter=",")
V_full = np.sort(-np.array(fullcoopt_data.T[-1]))
V = np.sort(-np.array(coopt_data.T[-1]))
cost = []
V_inner = np.sort(-np.array(inner_coopt_data.T[-1]))
for i in range(len(V_full)):
    if V_full[i] > 0.1:
        for j in range(int(len(V_inner)/len(V_full))):
            cost = np.append(cost, V_full[i])
cost = np.append([7], cost)
eval_full = np.array([i for i in range(len(cost))])
costRTC = []
iterations = np.array(evolution_data.T[-1])
V_ev = np.sort(-np.array(evolution_data.T[3]))
for i in range(len(V_ev)):
    if i == 0:
        j_prev = 0
    else:
        j_prev = int(iterations[i-1])
    for j in range(j_prev, int(iterations[i])):
        costRTC = np.append(costRTC, V_ev[i])
costRTC = np.append([cost[0]], costRTC)
eval = np.array([i for i in range(len(costRTC))])
plt.figure(figsize=(11,9))
plt.plot(eval_full,cost, label = "RTCD", color = "C1")
plt.plot(eval,costRTC, label = "RTC", color = "C0")
plt.grid(True)
plt.xlabel("Evaluations", fontsize=fontSize)
plt.ylabel("ROA Volume", fontsize=fontSize)
plt.xticks(fontsize=ticksSize)
plt.yticks(fontsize=ticksSize)
plt.rc('legend', fontsize=fontSize)
plt.legend(loc = "upper left")

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
# plt.figure(figsize=(11,9))
# plt.scatter(V_full, np.array(ml_var_full)*9.81, label = r"$mgl_{RTCD}\ [Nm]$", color = "C1", marker = ".",s=150)
# plt.hlines(2.5,0,70,colors=["C1"], linestyles="--")
# plt.text(60,2.52,r"$\tau_{lim}$",color = "C1", fontsize=fontSize)
# plt.scatter(V, r_var, label = r"$r_{RTC}$", color = "C0", marker = ".", s=150)
# plt.grid(True)
# plt.xlabel("ROA Volume", fontsize=fontSize)
# plt.ylabel("RTCD Optimization Parameters", fontsize=fontSize)
# plt.xticks(fontsize=ticksSize)
# plt.yticks(fontsize=ticksSize)
# plt.xlim(0,70)
# plt.ylim(-1,8)
# plt.rc('legend', fontsize=fontSize)
# plt.legend(loc = "upper left") 



########################
# Real Trajectories plot
########################

trajDirtran_path = "results/cart_pole/realExperiments/167332dirtran.csv"
trajRTC_path = "results/cart_pole/realExperiments/167332rtc.csv"
trajDirtrandist_path = "results/cart_pole/realExperiments/167332dirtrandist.csv"
trajRTCdist_path = "results/cart_pole/realExperiments/167332rtcdist.csv"

# load trajectories from csv file
trajDirtran = np.loadtxt(trajDirtran_path, skiprows=1, delimiter=",")
dirtran_time_list = trajDirtran.T[0].T  
dirtran_x0_list = trajDirtran.T[indexes[0]+1].T  
dirtran_x1_list = trajDirtran.T[indexes[1]+1].T  
dirtran_force_list = trajDirtran.T[5].T   

trajOptRTC = np.loadtxt(trajRTC_path, skiprows=1, delimiter=",")
rtc_time_list = trajOptRTC.T[0].T  
rtc_x0_list  = trajOptRTC.T[indexes[0]+1].T  
rtc_x1_list  = trajOptRTC.T[indexes[1]+1].T  
rtc_force_list  = trajOptRTC.T[5].T


trajDirtrandist = np.loadtxt(trajDirtrandist_path, skiprows=1, delimiter=",")
dirtrandist_time_list = trajDirtrandist.T[0].T  
dirtrandist_x0_list = trajDirtrandist.T[indexes[0]+1].T  
dirtrandist_x1_list = trajDirtrandist.T[indexes[1]+1].T  
dirtrandist_force_list = trajDirtrandist.T[5].T   

trajOptRTCdist = np.loadtxt(trajRTCdist_path, skiprows=1, delimiter=",")
rtcdist_time_list = trajOptRTCdist.T[0].T  
rtcdist_x0_list  = trajOptRTCdist.T[indexes[0]+1].T  
rtcdist_x1_list  = trajOptRTCdist.T[indexes[1]+1].T  
rtcdist_force_list  = trajOptRTCdist.T[5].T


# Plot the not disturbed case results
ticksSize = 40
fontSize = 40
fig, axs = plt.subplots(3,1, figsize=(17, 25))
axs[0].plot(dirtran_time_list, dirtran_x0_list, label = "DIRTRAN", color = "C0")
axs[0].plot(rtc_time_list, rtc_x0_list, label = "RTC", color = "C1")
axs[0].legend(loc = "upper right", fontsize = fontSize)
axs[0].set_ylabel(labels[indexes[0]], fontsize = fontSize)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[1].plot(dirtran_time_list, dirtran_x1_list, label = "DIRTRAN", color = "C0")
axs[1].plot(rtc_time_list, rtc_x1_list, label = "RTC", color = "C1")
axs[1].legend(loc = "upper right", fontsize = fontSize)
axs[1].set_ylabel(labels[indexes[1]], fontsize = fontSize)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[2].plot(dirtran_time_list, dirtran_force_list, label = "DIRTRAN", color = "C0")
axs[2].plot(rtc_time_list, rtc_force_list, label = "RTC", color = "C1")
axs[2].set_ylabel("force [N]", fontsize = fontSize)
axs[2].set_xlabel("time [s]", fontsize = fontSize)
axs[2].grid(True)
axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[2].legend(loc = "upper right", fontsize = fontSize)

fig, axs = plt.subplots(3,1, figsize=(17, 25))
axs[0].plot(dirtrandist_time_list, dirtrandist_x0_list, label = "DIRTRAN", color = "C0")
axs[0].plot(rtcdist_time_list, rtcdist_x0_list, label = "RTC", color = "C1")
axs[0].legend(loc = "upper right", fontsize = fontSize)
axs[0].set_ylabel(labels[indexes[0]], fontsize = fontSize)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
axs[1].plot(dirtrandist_time_list, dirtrandist_x1_list, label = "DIRTRAN", color = "C0")
axs[1].plot(rtcdist_time_list, rtcdist_x1_list, label = "RTC", color = "C1")
axs[1].legend(loc = "upper right", fontsize = fontSize)
axs[1].set_ylabel(labels[indexes[1]], fontsize = fontSize)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)
tdist_i = dirtrandist_time_list[int(len(dirtrandist_time_list)/2)-8]
tdist_f = dirtrandist_time_list[int(len(dirtrandist_time_list)/2)-5]
axs[0].hlines(0.25,tdist_i,tdist_f,colors=["black"])
axs[1].hlines(0,tdist_i,tdist_f,colors=["black"])
axs[0].vlines(np.linspace(tdist_i,tdist_f,2),-0.5,1,colors=["black"])
axs[1].vlines(np.linspace(tdist_i,tdist_f,2),-6,6,colors=["black"])
axs[2].plot(dirtrandist_time_list, dirtrandist_force_list, label = "DIRTRAN", color = "C0")
axs[2].plot(rtcdist_time_list, rtcdist_force_list, label = "RTC", color = "C1")
axs[2].set_ylabel("force [N]", fontsize = fontSize)
axs[2].set_xlabel("time [s]", fontsize = fontSize)
axs[2].grid(True)
axs[2].tick_params(axis='both', which='major', labelsize=ticksSize)
tdist_i = dirtrandist_time_list[int(len(dirtrandist_time_list)/2)-8]
tdist_f = dirtrandist_time_list[int(len(dirtrandist_time_list)/2)-5]
axs[2].hlines(0.25,tdist_i,tdist_f,colors=["black"])
axs[2].hlines(0,tdist_i,tdist_f,colors=["black"])
axs[2].hlines([-6,6],tdist_i,tdist_f,colors=["black"], linestyles="--")
axs[2].vlines(np.linspace(tdist_i,tdist_f,2),-0.5,1,colors=["black"])
axs[2].vlines(np.linspace(tdist_i,tdist_f,2),-7,7,colors=["black"])
axs[2].legend(loc = "upper right", fontsize = fontSize)

# TODO: state-space plots

plt.show()