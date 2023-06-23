import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import random

from simple_pendulum.controllers.tvlqr.roa.plot import get_ellipse_patch, sample_from_ellipsoid, quad_form, getEllipseFromCsv, getEllipseContour
from simple_pendulum.utilities.process_data import prepare_trajectory
from simulatorComparison import usingDrakeSimulator, usingFelixSimulator
from simple_pendulum.trajectory_optimization.dirtrel.dirtrelTrajOpt import pendulum, lqr_controller, tvlqr_controller
from simple_pendulum.simulation.simulation import DrakeSimulator

# results to verify
traj_path = "results/simple_pendulum/Design3.1Opt /DIRTREL/trajectory_dirtrel.csv"
funnel_path = "results/simple_pendulum/Design3.1Opt /DIRTREL/SosfunnelDIRTREL.csv"
label = "DIRTREL"

optimized_traj_path = "results/simple_pendulum/Design3.1Opt /CMA-ES/DesignOpt/Vdirtrel/trajectoryOptimal_CMAES.csv" 
optimized_funnel_path = "results/simple_pendulum/Design3.1Opt /CMA-ES/DesignOpt/Vdirtrel/SosfunnelOptimal_CMAES.csv"
optimal_pars_path = "results/simple_pendulum/Design3.1Opt /CMA-ES/DesignOpt/Vdirtrel/fullCoopt_CMAES.csv"
optimal_label = "CMA-ES"

initial_knot = 0

# Load optimal params
controller_data = np.loadtxt(optimal_pars_path, skiprows=1, delimiter=",")
max_idx = np.where(-controller_data.T[5] == max(-controller_data.T[5]))[0][0]
m_opt = controller_data[max_idx,0]
l_opt = controller_data[max_idx,1]
Q_opt = np.diag([controller_data[max_idx,2],controller_data[max_idx,3]])
R_opt = [controller_data[max_idx,4]]
print("The optimal m is: ", m_opt)
print("The optimal l is: ", l_opt)
print("The optimal Q is: ", Q_opt)
print("The optimal R is: ", R_opt)
opt_mpar = {"l": l_opt,
        "m": m_opt,
        "b": 0.4, 
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}
# initial parameters
Q = np.diag([10,1])
R = [0.1]
mpar = {"l": 0.5,
        "m": 0.67,
        "b": 0.4, 
        "g": 9.81,
        "cf": 0.0,
        "tl": 2.5}

# loading the trajectories
data_dict = prepare_trajectory(traj_path)
trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
x_t = np.array([trajectory.T[1], trajectory.T[2]])
u_t = np.array([trajectory.T[3]])
time = np.array([trajectory.T[0]]).T

optimized_data_dict = prepare_trajectory(optimized_traj_path)
optimized_trajectory = np.loadtxt(optimized_traj_path, skiprows=1, delimiter=",")
optimized_x_t = np.array([optimized_trajectory.T[1], optimized_trajectory.T[2]])
optimized_u_t = np.array([optimized_trajectory.T[3]])
optimized_time = np.array([optimized_trajectory.T[0]]).T

# load funnel data
funnel_data = np.loadtxt(funnel_path, skiprows=1, delimiter=",")
rho = funnel_data[0].T

optimized_funnel_data = np.loadtxt(optimized_funnel_path, skiprows=1, delimiter=",")
optimized_rho = optimized_funnel_data[0].T

# figure initialization
fig = plt.figure(figsize=(12,6))
fig.suptitle("Verification of the optimized RoA guarantee certificate")
gs = fig.add_gridspec(1, 3)
ax = fig.add_subplot(gs[0,0])
ax.set_xlabel("x")
ax.set_ylabel(r"$\dot{x}$")

# plot of the verified ellipse
rho0, S0 = getEllipseFromCsv(funnel_path, initial_knot)
p = get_ellipse_patch(x_t[0][initial_knot],x_t[1][initial_knot], rho0,S0,linec= "black", linest="dashed", label = label)
ax.add_patch(p)
rho0_opt, S0_opt = getEllipseFromCsv(optimized_funnel_path, initial_knot)
optimized_p = get_ellipse_patch(optimized_x_t[0][initial_knot],optimized_x_t[1][initial_knot], rho0_opt,S0_opt,linec= "black", label = optimal_label)
ax.add_patch(optimized_p)
ax.scatter(optimized_x_t[0][initial_knot],optimized_x_t[1][initial_knot], marker = ".", color = "k")
ax.grid(True)
ax.legend()
ax.set_title(f"Verified ellipse, knot {initial_knot}")

contour = getEllipseContour(S0,rho0, optimized_x_t[:,initial_knot])
theta_limit = max(contour[:,0])-optimized_x_t[0][initial_knot]+0.5
w_limit = max(contour[:,1])-optimized_x_t[1][initial_knot]+1

# plot the funnels rho
ax1 = fig.add_subplot(gs[0,1]) 
ax1.plot(time, rho, color = "k", linestyle="dashed")
ax1.plot(optimized_time, optimized_rho, color = "k")
ax1.set_title(f"Initial Controller")
ax1.set_xlabel("time")
ax1.set_ylabel(r"ctg VS $\rho$")
ax1.set_ylim([0,15])

# plot the funnels rho
ax2 = fig.add_subplot(gs[0,2]) 
#ax2.plot(time, rho, color = "k", linestyle="dashed")
ax2.plot(optimized_time, optimized_rho, color = "k")
ax2.set_title(f"Final Controller")
ax2.set_xlabel("time")
ax2.set_ylabel(r"ctg VS $\rho$")
ax2.set_ylim([0,15])

# simulate and verify
nSimulations = 100 #000
first_green = True
first_red = True
first_orange = True
first_yellow = True
for j in range(1,nSimulations+1):   
    xBar0 = sample_from_ellipsoid(S0_opt,rho0_opt) # sample new initial state inside the optimized RoA
    x_i = xBar0 + np.array(optimized_x_t)[:,initial_knot]

    t_sim, x_sim, u_sim = usingFelixSimulator(mpar,traj_path, Q, R,x0=x_i, t0 =time[initial_knot][0]) #simulating
    optimized_t_sim, optimized_x_sim, optimized_u_sim = usingFelixSimulator(opt_mpar,optimized_traj_path, Q_opt, R_opt,x0=x_i, t0 =optimized_time[initial_knot][0]) #simulating
    optimized_x_sim = np.array(optimized_x_sim).T
    x_sim = np.array(x_sim).T

    # plotting the checked initial states and resulting trajectories, the color depends on the result
    optimized_outRoa = False
    outRoa = False
    optimized_stabilized = True
    stabilized = True
    init_innerEllipse = True
    ctg_t = [quad_form(S0, x_i - np.array(optimized_x_t)[:,initial_knot])]
    optimized_ctg_t = [quad_form(S0_opt, x_i - np.array(optimized_x_t)[:,initial_knot])]
    if ctg_t[0]>rho0:
        init_innerEllipse = False
    for i in range(1,len(optimized_time)-initial_knot):
        rhoi_opt, Si_opt = getEllipseFromCsv(optimized_funnel_path, initial_knot+i)
        optimized_ctg_i = quad_form(Si_opt, np.array(optimized_x_sim)[:,i-1] - np.array(optimized_x_t)[:,initial_knot+i])
        optimized_ctg_t = np.append(optimized_ctg_t,optimized_ctg_i)  
        if optimized_ctg_i > rhoi_opt:
            optimized_outRoa = True
            if i == (len(optimized_t_sim)-1):
                optimized_stabilized = False
        rhoi, Si = getEllipseFromCsv(optimized_funnel_path, initial_knot+i)
        ctg_i = quad_form(Si, np.array(optimized_x_sim)[:,i-1] - np.array(x_t)[:,initial_knot+i])
        ctg_t = np.append(ctg_t,ctg_i)  
        if ctg_i > rhoi:
            outRoa = True
            if i == (len(optimized_t_sim)-1):
                stabilized = False                

    if init_innerEllipse:
        if not (stabilized or optimized_stabilized):
            ax.scatter(x_i[0],x_i[1], 4,color ="red")
            # if first_red:
            #     ax1.plot(t_sim,ctg_t,color="red", label = "not stabilized")
            #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="red", label = "not stabilized")
            # else:
            #     ax1.plot(t_sim,ctg_t,color="red")
            #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="red")
            first_red = False
        else:
            if optimized_outRoa:
                ax.scatter(x_i[0],x_i[1], 4,color ="orange")
                # if first_orange:
                #     ax1.plot(t_sim,ctg_t,color="orange", label = "out of optimal RoA")
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="orange", label = "out of optimal RoA")
                # else:
                #     ax1.plot(t_sim,ctg_t,color="orange")
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="orange")
                first_orange = False
            if outRoa and (not optimized_outRoa):
                ax.scatter(x_i[0],x_i[1], 4,color ="yellow")
                # if first_yellow:
                #     ax1.plot(t_sim,ctg_t,color="yellow", label = "out of initial RoA")
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="green")
                # else:
                #     ax1.plot(t_sim,ctg_t,color="yellow")
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="green")
                first_yellow = False
            if (not outRoa):
                ax.scatter(x_i[0],x_i[1], 4,color ="green")
                # if first_green:
                #     ax1.plot(t_sim,ctg_t,color="green", label = "inside of initial RoA")
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="green", label = "inside of optimal RoA")
                # else:
                #     ax1.plot(t_sim,ctg_t,color="green")
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="green")
                first_green = False
    else:
        if not (stabilized or optimized_stabilized):
            ax.scatter(x_i[0],x_i[1], 4,color ="red")
            # if first_red:
            #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="red", label = "not stabilized")
            # else:
            #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="red")
            first_red = False
        else:
            if optimized_outRoa:
                ax.scatter(x_i[0],x_i[1], 4,color ="orange")
                # if first_orange:
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="orange", label = "out of optimal RoA")
                # else:
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="orange")
                first_orange = False
            else:
                ax.scatter(x_i[0],x_i[1], 4,color ="green")
                # if first_green:
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="green", label = "inside of optimal RoA")
                # else:
                #     ax2.plot(optimized_t_sim,optimized_ctg_t,color="green")
                first_green = False
#ax1.legend(loc = "upper left")
#ax2.legend(loc = "upper left")

plt.show()
assert False

# figure initialization
fig, ax = plt.subplots()
fig.suptitle("Verification of the optimized RoA guarantee certificate")
ax.set_xlabel("x")
ax.set_ylabel(r"$\dot{x}$")

# plot of the verified ellipse
p = get_ellipse_patch(x_t[0][initial_knot],x_t[1][initial_knot],rho[initial_knot],S_t.value(time[initial_knot]),linec= "black", linest="dashed", label = "DIRTREL")
ax.add_patch(p)
optimized_p = get_ellipse_patch(optimized_x_t[0][initial_knot],optimized_x_t[1][initial_knot],optimized_rho[initial_knot],optimized_S_t.value(optimized_time[initial_knot]),linec= "black", label = "CMA-ES")
ax.add_patch(optimized_p)
ax.scatter(x_t[0][initial_knot],x_t[1][initial_knot], marker = ".", color = "k")
ax.scatter(optimized_x_t[0][initial_knot],optimized_x_t[1][initial_knot], marker = "x", color = "k")
ax.grid(True)
ax.set_title(f"Verified ellipse, knot {initial_knot}")

# simulate and verify
nSimulations = 1000
first_green = True
first_red = True
first_orange = True
first_yellow = True
for j in range(1,nSimulations+1):   
    random_x_val = round(random.uniform(optimized_x_t[0][initial_knot]-2,optimized_x_t[0][initial_knot]+2), 2)
    random_xDot_val = round(random.uniform(optimized_x_t[1][initial_knot]-6,optimized_x_t[1][initial_knot]+6), 2)
    x_i = np.array([random_x_val, random_xDot_val])

    t_sim, x_sim, u_sim = sim.simulate(x0 = x_i, init_knot=initial_knot) # simulating  
    optimized_t_sim, optimized_x_sim, optimized_u_sim = optimized_sim.simulate(x0 = x_i, init_knot=initial_knot)

    # plotting the checked initial states and resulting trajectories, the color depends on the result
    optimized_outRoa = False
    outRoa = False
    optimized_stabilized = True
    stabilized = True
    optimized_ctg_t = []
    ctg_t = []
    for i in range(len(optimized_t_sim)):
        optimized_ctg_i = quad_form(optimized_S_t.value(optimized_time[initial_knot+i]), np.array(optimized_x_sim)[:,i] - np.array(optimized_x_t)[:,initial_knot+i])
        optimized_ctg_t = np.append(ctg_t,optimized_ctg_i)  
        if optimized_ctg_i > optimized_rho[initial_knot+i]:
            optimized_outRoa = True
            if i == (len(optimized_t_sim)-1):
                optimized_stabilized = False
        ctg_i = quad_form(S_t.value(time[initial_knot+i]), np.array(x_sim)[:,i] - np.array(x_t)[:,initial_knot+i])
        ctg_t = np.append(ctg_t,ctg_i)  
        if ctg_i > rho[initial_knot+i]:
            outRoa = True
            if i == (len(optimized_t_sim)-1):
                stabilized = False

    if not (stabilized or optimized_stabilized):
        if first_red:
            ax.scatter(x_i[0],x_i[1], 4,color ="red", label = "not stabilized")
        else:
            ax.scatter(x_i[0],x_i[1], 4,color ="red")
        first_red = False
    else:
        if optimized_outRoa:
            if first_orange:
                ax.scatter(x_i[0],x_i[1], 4,color ="orange", label = "out of optimal RoA")
            else:
                ax.scatter(x_i[0],x_i[1], 4,color ="orange")
            first_orange = False
        if outRoa and (not optimized_outRoa):
            if first_yellow:
                ax.scatter(x_i[0],x_i[1], 4,color ="yellow", label = "out of initial RoA")
            else:
                ax.scatter(x_i[0],x_i[1], 4,color ="yellow")
            first_yellow = False
        if (not outRoa):
            if first_green:
                ax.scatter(x_i[0],x_i[1], 4,color ="green", label = "inside of initial RoA")
            else:
                ax.scatter(x_i[0],x_i[1], 4,color ="green")
            first_green = False
ax.legend()
plt.show()