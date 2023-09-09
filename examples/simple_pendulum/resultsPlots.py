import numpy as np
import matplotlib as mlp
import tikzplotlib
import matplotlib.pyplot as plt
import time

from simple_pendulum.controllers.tvlqr.roa.utils import funnelVolume_convexHull, funnelVolume
from simple_pendulum.controllers.tvlqr.roa.plot import plotFunnel, rhoComparison, funnel2DComparison
from simple_pendulum.utilities.process_data import prepare_trajectory, saveFunnel

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

oldOpt = False
algorithm = "RTCD" #"RTC" #both

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
elif algorithm == "both" and not oldOpt:
    traj_path2 = "results/simple_pendulum/optCMAES_167329/trajectoryOptimal_CMAES.csv" 
    funnel_path2 = "results/simple_pendulum/optCMAES_167329/SosfunnelOptimal_CMAES.csv"
    label2 = "RTC"
    traj_path3 = "results/simple_pendulum/optDesignCMAES_167329/trajectoryOptimal_CMAES.csv"
    funnel_path3 = "results/simple_pendulum/optDesignCMAES_167329/SosfunnelOptimal_CMAES.csv" 
    label3 = "RTCD"
else:
    print("Erroneous algorithm label.")
    assert False

#########################
# Trajectories comparison
#########################

# # load trajectories
# trajectory1 = np.loadtxt(traj_path1, skiprows=1, delimiter=",")
# T1 = trajectory1.T[0].T
# X1 = [trajectory1.T[1].T, trajectory1.T[2].T]
# U1 = trajectory1.T[3].T

# trajectory2 = np.loadtxt(traj_path2, skiprows=1, delimiter=",")
# T2 = trajectory2.T[0].T
# X2 = [trajectory2.T[1].T, trajectory2.T[2].T]
# U2 = trajectory2.T[3].T

# # comparison plots
# ticksSize = 30
# fontSize = 30
# fig, axs = plt.subplots(2,1, figsize=(18, 9))
# axs[0].plot(T1,X1[0], label = label1)
# axs[0].plot(T2,X2[0], label = label2)
# axs[0].legend(loc = "lower right", fontsize = fontSize)
# axs[0].set_ylabel(r'$\theta$'+" [rad]", fontsize = fontSize)
# axs[0].grid(True)
# axs[0].tick_params(axis='both', which='major', labelsize=ticksSize)
# axs[1].plot(T1,U1, label = label1)
# axs[1].plot(T2,U2, label = label2)
# axs[1].legend(loc = "upper right", fontsize = fontSize)
# axs[1].set_ylabel(r'$u$'+" [Nm]", fontsize = fontSize)
# axs[1].set_xlabel("time [s]", fontsize = fontSize)
# axs[1].grid(True)
# axs[1].tick_params(axis='both', which='major', labelsize=ticksSize)

# if algorithm == "both":
#     trajectory3 = np.loadtxt(traj_path3, skiprows=1, delimiter=",")
#     T3 = trajectory3.T[0].T
#     X3 = [trajectory3.T[1].T, trajectory3.T[2].T]
#     U3 = trajectory3.T[3].T
#     axs[0].plot(T3,X3[0], label = label3, color = "black")
#     axs[0].legend(loc = "lower right", fontsize = fontSize)
#     axs[1].plot(T3,U3, label = label3, color = "black")
#     axs[1].legend(loc = "upper right", fontsize = fontSize)

####################
# Funnels comparison
####################

# Volume computation
vol1 = funnelVolume_convexHull(funnel_path1, traj_path1)
vol2 = funnelVolume_convexHull(funnel_path2, traj_path2)
print("The convex hull volume of the "+ label1 +" funnel is", vol1)
print("The convex hull volume of the "+ label2 +" funnel is", vol2)

# Funnel plots
ticksSize = 40
fontSize = 40
volumes = (np.round(vol1,2),np.round(vol2,2))
# funnel2DComparison(funnel_path1,funnel_path2,traj_path1,traj_path2,volumes = volumes, fontSize = fontSize, ticksSize = ticksSize)
# if algorithm == "both":    
#     vol3 = funnelVolume_convexHull(funnel_path3, traj_path3)
#     print("The convex hull volume of the "+ label3 +" funnel is", vol3)  
#     volumes = (np.round(vol1,2),np.round(vol3,2))
#     funnel2DComparison(funnel_path1,funnel_path3,traj_path1,traj_path3,volumes = volumes, fontSize = fontSize, ticksSize = ticksSize)  

#     ax0 = plotFunnel(funnel_path1, traj_path1, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
#     plotFunnel(funnel_path2, traj_path2, ax=ax0, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
#     ax1 = plotFunnel(funnel_path1, traj_path1, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
#     plotFunnel(funnel_path3, traj_path3, ax=ax1, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
#     text0 = f"V = {np.round(vol1,2)} \n"+r"$V_{RTC}$"+f"= {np.round(vol2,2)}"
#     text1 = f"V = {np.round(vol1,2)} \n"+r"$V_{RTCD}$"+f"= {np.round(vol3,2)}"
#     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     ax0.text(0.05, 0.95, text0, transform=ax0.transAxes, fontsize=fontSize,
#     verticalalignment='top', bbox=props)
#     ax1.text(0.05, 0.95, text1, transform=ax1.transAxes, fontsize=fontSize,
#     verticalalignment='top', bbox=props)

ax0 = plotFunnel(funnel_path2, traj_path2, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
plotFunnel(funnel_path1, traj_path1, ax=ax0, fontSize= fontSize, ticksSize= ticksSize, noTraj = True)
#text0 = r"$V_{DIRTRAN}$"+f" = {np.round(vol1,2)} \n"+r"$V_{RTCD}$"+f"= {np.round(vol2,2)}"
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# ax0.text(0.05, 0.95, text0, transform=ax0.transAxes, fontsize=fontSize,
# verticalalignment='top', bbox=props)
g_patch = mlp.patches.Patch(color='green', label='RTC-D')
r_patch = mlp.patches.Patch(color='red', label='DIRTRAN')
leg = ax0.legend(handles=[r_patch, g_patch], fontsize=fontSize,loc = "upper right")
# tikzplotlib_fix_ncols(leg)
# tikzplotlib.save("results/simple_pendulum/resultsPlots.tex")
plt.show()
assert False

###################
# Optimal cost plot
###################

if oldOpt:
    fullcoopt_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/DesignOpt/Vdirtran/fullCoopt_CMAES.csv" 
    inner_coopt_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/DesignOpt/Vdirtran/CooptData_CMAES.csv"
    coopt_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/VOpt/DIRTRAN/CooptData_CMAES.csv" 
    evolution_path = "results/simple_pendulum/optimizationTrials/Design3.1Opt /CMA-ES/VOpt/DIRTRAN/outcmaes/evolution.csv"
    fullcoopt_data = np.loadtxt(fullcoopt_path, delimiter=",",skiprows=1)
    coopt_data = np.loadtxt(coopt_path, delimiter=",",skiprows=1) 
    inner_coopt_data = np.loadtxt(inner_coopt_path, delimiter=",",skiprows=1)
    evolution_data = np.loadtxt(evolution_path, delimiter=",")
else:
    fullcoopt_path = "results/simple_pendulum/optDesignCMAES_167329/fullCoopt_CMAES.csv" 
    inner_coopt_path = "results/simple_pendulum/optDesignCMAES_167329/CooptData_CMAES.csv"
    coopt_path = "results/simple_pendulum/optCMAES_167329/CooptData_CMAES.csv"  
    evolution_path = "results/simple_pendulum/optCMAES_167329/outcmaes/evolution.csv"
    fullcoopt_data = np.loadtxt(fullcoopt_path, delimiter=",")
    coopt_data = np.loadtxt(coopt_path, delimiter=",")
    inner_coopt_data = np.loadtxt(inner_coopt_path, delimiter=",")
    evolution_data = np.loadtxt(evolution_path, delimiter=",")
V = np.sort(-np.array(coopt_data.T[-1]))
V_full = np.sort(-np.array(fullcoopt_data.T[-1]))
m_var_full = []
l_var_full = []
r_var_full = []
ml_var_full = []
for i in range(len(V_full)):
    idx = np.where(-np.array(fullcoopt_data.T[-1]) == V_full[i])
    m_var_full = np.append(m_var_full, fullcoopt_data.T[0][idx])
    l_var_full = np.append(l_var_full, fullcoopt_data.T[1][idx])
    r_var_full = np.append(r_var_full, fullcoopt_data.T[4][idx])
    ml_var_full = np.append(ml_var_full, fullcoopt_data.T[0][idx][0]*fullcoopt_data.T[1][idx][0])
r_var = []
q11_var = []
q22_var = []
for i in range(len(V)):
    idx = np.where(-np.array(coopt_data.T[-1]) == V[i])
    r_var = np.append(r_var, coopt_data.T[2][idx])
    q11_var = np.append(q11_var, coopt_data.T[0][idx])
    q22_var = np.append(q22_var, coopt_data.T[1][idx])

ticksSize = 30
fontSize = 30
cost = []
V_inner = np.sort(-np.array(inner_coopt_data.T[-1]))
for i in range(len(V_full)):
    if V_full[i] > 0.1:
        for j in range(int(len(V_inner)/len(V_full))):
            cost = np.append(cost, V_full[i])
cost = np.append([15.75], cost)
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
plt.figure(figsize=(12,9))
plt.plot(eval_full,cost, label = "RTCD", color = "C1")
plt.plot(eval,costRTC, label = "RTC", color = "C0")
plt.grid(True)
plt.xlabel("Evaluations", fontsize=fontSize)
plt.ylabel("ROA Volume", fontsize=fontSize)
plt.xticks(fontsize=ticksSize)
plt.yticks(fontsize=ticksSize)
plt.rc('legend', fontsize=fontSize)
plt.legend(loc = "upper left")

plt.figure(figsize=(11,9))
plt.scatter(V_full, np.array(ml_var_full)*9.81, label = r"$mgl_{RTCD}\ [Nm]$", color = "C1", marker = ".",s=150)
plt.hlines(2.5,0,V_full.max(),colors=["C1"], linestyles="--")
plt.text(60,2.52,r"$\tau_{lim}$",color = "C1", fontsize=fontSize)
plt.scatter(V, r_var, label = r"$r_{RTC}$", color = "C0", marker = ".", s=150)
plt.grid(True)
plt.xlabel("ROA Volume", fontsize=fontSize)
plt.ylabel("Optimization Parameters", fontsize=fontSize)
plt.xticks(fontsize=ticksSize)
plt.yticks(fontsize=ticksSize)
plt.xlim(0,V_full.max())
plt.ylim(-1,8)
plt.rc('legend', fontsize=fontSize)
plt.legend(loc = "upper left") #loc = (2,1))

plt.show()