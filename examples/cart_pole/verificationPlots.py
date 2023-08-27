import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import pandas

from cart_pole.model.parameters import Cartpole
from cart_pole.utilities.process_data import prepare_trajectory
from cart_pole.simulation.simulator import StepSimulator
from cart_pole.controllers.tvlqr.RoAest.utils import getEllipseFromCsv
from cart_pole.model.parameters import generateUrdf

nVerifications = 100

# Initial simulation environment
traj_path1 = "data/cart_pole/dirtran/trajectory.csv"
funnel_path1 = "data/cart_pole/RoA/Probfunnel_DIRTRAN.csv"
label1 = "DIRTRAN"
sys = Cartpole("short")
sys.fl = 6
old_Mp = sys.Mp
old_lp = sys.lp
sys.Mp = 0.227
l_ratio = (sys.lp/old_lp)**2
sys.Jp = sys.Jp*l_ratio + (sys.Mp-old_Mp)*(sys.lp**2)
urdf_path = generateUrdf(sys.Mp, sys.lp, sys.Jp)
xG = np.array([0,0,0,0])  
trajectory = np.loadtxt(traj_path1, skiprows=1, delimiter=",")
X1 = np.array([trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]])
U1 = np.array([trajectory.T[5]])
T1 = np.array([trajectory.T[0]]).T 
traj_dict = prepare_trajectory(traj_path1)
traj_x1 = traj_dict["des_cart_pos_list"]
traj_x2 = traj_dict["des_pend_pos_list"]
traj_x3 = traj_dict["des_cart_vel_list"]
traj_x4 = traj_dict["des_pend_vel_list"]
controller_options = {"T_nom": traj_dict["des_time_list"],
                        "U_nom": traj_dict["des_force_list"],
                        "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                        "Q": np.diag([10,10,.1,.1]),
                        "R": np.array([10]),
                        "xG": xG}
cartpole = {"urdf": urdf_path,
            "sys": sys,
            "x_lim": 0.3}
dt_sim = 0.01
sim1 = StepSimulator(cartpole, controller_options, dt_sim)

# Optimized simulation environment
traj_path2 = "results/cart_pole/optCMAES_167332/trajectoryOptimal_CMAES.csv" 
funnel_path2 = "results/cart_pole/optCMAES_167332/RoA_CMAES.csv"
label2 = "RTC"
sys2 = sys
urdf_path2 = urdf_path
Q_2 = np.diag([1.71,1.01,.1,.1])
R_2 = np.array([5.01])
trajectory2 = np.loadtxt(traj_path2, skiprows=1, delimiter=",")
X2 = np.array([trajectory2.T[1], trajectory2.T[2], trajectory2.T[3], trajectory2.T[4]])
U2 = np.array([trajectory2.T[5]])
T2 = np.array([trajectory2.T[0]]).T 
traj_dict = prepare_trajectory(traj_path2)
traj_x1 = traj_dict["des_cart_pos_list"]
traj_x2 = traj_dict["des_pend_pos_list"]
traj_x3 = traj_dict["des_cart_vel_list"]
traj_x4 = traj_dict["des_pend_vel_list"]
controller_options = {"T_nom": traj_dict["des_time_list"],
                        "U_nom": traj_dict["des_force_list"],
                        "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                        "Q": Q_2,
                        "R": R_2,
                        "xG": xG}
cartpole = {"urdf": urdf_path2,
            "sys": sys2,
            "x_lim": 0.3}
sim2 = StepSimulator(cartpole, controller_options, dt_sim)

traj_path3 = "results/cart_pole/optDesignCMAES_167332/trajectoryOptimal_CMAES.csv"
funnel_path3 = "results/cart_pole/optDesignCMAES_167332/RoA_CMAES.csv" 
label3 = "RTCD"
sys3 = Cartpole("short")
sys3.fl = 6
old_Mp = sys3.Mp
old_lp = sys3.lp
sys3.Mp = 0.129
sys3.lp = 0.185
l_ratio = (sys3.lp/old_lp)**2
sys3.Jp = sys3.Jp*l_ratio + (sys3.Mp-old_Mp)*(sys3.lp**2)
urdf_path3 = generateUrdf(sys3.Mp, sys3.lp, sys3.Jp)
Q_3 = np.diag([1.31,1.023,.1,.1])
R_3 = np.array([5.71])
trajectory3 = np.loadtxt(traj_path3, skiprows=1, delimiter=",")
X3 = np.array([trajectory3.T[1], trajectory3.T[2], trajectory3.T[3], trajectory3.T[4]])
U3 = np.array([trajectory3.T[5]])
T3 = np.array([trajectory3.T[0]]).T 
traj_dict = prepare_trajectory(traj_path3)
traj_x1 = traj_dict["des_cart_pos_list"]
traj_x2 = traj_dict["des_pend_pos_list"]
traj_x3 = traj_dict["des_cart_vel_list"]
traj_x4 = traj_dict["des_pend_vel_list"]
controller_options = {"T_nom": traj_dict["des_time_list"],
                        "U_nom": traj_dict["des_force_list"],
                        "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                        "Q": Q_3,
                        "R": R_3,
                        "xG": xG}
cartpole = {"urdf": urdf_path3,
            "sys": sys3,
            "x_lim": 0.3}
sim3 = StepSimulator(cartpole, controller_options, dt_sim)

# Simulate grid sampled init states
ver_knot = int(len(traj_x1)/2)
x0_center = X1.T[ver_knot] #np.array([0,np.pi,0,0])
l_grid = 0.4
(rho1, S1) = getEllipseFromCsv(funnel_path1,-1)
(rho2, S2) = getEllipseFromCsv(funnel_path2,-1)
(rho3, S3) = getEllipseFromCsv(funnel_path3,-1)
succ1 = 0
succ2 = 0
succ3 = 0
fig1, ax1 = plt.subplots(2,2, figsize = (11, 9))
fig2, ax2 = plt.subplots(2,2, figsize = (11, 9))
fig3, ax3 = plt.subplots(2,2, figsize = (11, 9))
fontSize = 30
ticksSize = 30
ax1[0][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax1[0][1].tick_params(axis='both', which='major', labelsize=ticksSize)
ax1[1][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax1[1][1].tick_params(axis='both', which='major', labelsize=ticksSize)
#ax1[0][1].legend(loc = "upper right", fontsize = fontSize)
ax2[0][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax2[0][1].tick_params(axis='both', which='major', labelsize=ticksSize)
ax2[1][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax2[1][1].tick_params(axis='both', which='major', labelsize=ticksSize)
#ax2[0][1].legend(loc = "upper right", fontsize = fontSize)
ax3[0][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax3[0][1].tick_params(axis='both', which='major', labelsize=ticksSize)
ax3[1][0].tick_params(axis='both', which='major', labelsize=ticksSize)
ax3[1][1].tick_params(axis='both', which='major', labelsize=ticksSize)
labels = [r"$x_{cart}$ [m]",r"$\theta$ [rad]",r"$\dot x_{cart}$ [m/s]",r"$\dot \theta$ [rad/s]"]
for i in range(nVerifications):
    x0 = np.array([np.random.uniform(x0_center[0]-l_grid,x0_center[0]+l_grid),
                   np.random.uniform(x0_center[1]-l_grid,x0_center[1]+l_grid),
                   np.random.uniform(x0_center[2]-l_grid,x0_center[2]+l_grid),
                   np.random.uniform(x0_center[3]-l_grid,x0_center[3]+l_grid)])
    sim1.init_simulation(x0=x0,init_knot=ver_knot)
    T1_sim, X1_sim, U1_sim = sim1.simulate()
    sim2.init_simulation(x0=x0,init_knot=ver_knot)
    T2_sim, X2_sim, U2_sim = sim2.simulate()
    sim3.init_simulation(x0=x0,init_knot=ver_knot)
    T3_sim, X3_sim, U3_sim = sim3.simulate()

    # Plot the results
    ax1[0][0].plot(T1_sim, X1_sim[0])
    ax1[0][1].plot(T1_sim, X1_sim[1])
    ax1[1][0].plot(T1_sim, X1_sim[2])
    ax1[1][1].plot(T1_sim, X1_sim[3])
    ax1[0][0].hlines(np.vstack((np.ones((len(T1_sim),1)),-np.ones((len(T1_sim),1))))*0.3,T1_sim[0], T1_sim[-1])
    ax1[0][0].set_xlabel(labels[0], fontsize = fontSize)
    ax1[0][1].set_xlabel(labels[1], fontsize = fontSize)
    ax1[1][0].set_xlabel(labels[2], fontsize = fontSize)
    ax1[1][1].set_xlabel(labels[3], fontsize = fontSize)

    ax2[0][0].plot(T2_sim, X2_sim[0])
    ax2[0][1].plot(T2_sim, X2_sim[1])
    ax2[1][0].plot(T2_sim, X2_sim[2])
    ax2[1][1].plot(T2_sim, X2_sim[3])
    ax2[0][0].hlines(np.vstack((np.ones((len(T2_sim),1)),-np.ones((len(T2_sim),1))))*0.3,T2_sim[0], T2_sim[-1])
    ax2[0][0].set_xlabel(labels[0], fontsize = fontSize)
    ax2[0][1].set_xlabel(labels[1], fontsize = fontSize)
    ax2[1][0].set_xlabel(labels[2], fontsize = fontSize)
    ax2[1][1].set_xlabel(labels[3], fontsize = fontSize)

    ax3[0][0].plot(T3_sim, X3_sim[0])
    ax3[0][1].plot(T3_sim, X3_sim[1])
    ax3[1][0].plot(T3_sim, X3_sim[2])
    ax3[1][1].plot(T3_sim, X3_sim[3])
    ax3[0][0].hlines(np.vstack((np.ones((len(T3_sim),1)),-np.ones((len(T3_sim),1))))*0.3,T3_sim[0], T3_sim[-1])
    ax3[0][0].set_xlabel(labels[0], fontsize = fontSize)
    ax3[0][1].set_xlabel(labels[1], fontsize = fontSize)
    ax3[1][0].set_xlabel(labels[2], fontsize = fontSize)
    ax3[1][1].set_xlabel(labels[3], fontsize = fontSize)

    # fig_test, ax_test = plt.subplots(1,1, figsize = (10, 8))
    # ax_test.plot(T_sim,U_sim.T)
    # ax_test.plot(T,U.T, linestyle = "--")
    # ax_test.hlines(np.vstack((np.ones((len(T),1)),-np.ones((len(T),1))))*sys.fl,T[0], T[-1])
    # ax_test.set_xlabel("u(force)", fontsize=18)

    V2 = X2_sim.T[-1].dot(S2).dot(X2_sim.T[-1])
    if V2 < rho2:
        succ2 += 1
    V1 = X1_sim.T[-1].dot(S1).dot(X1_sim.T[-1])
    if V1 < rho1:
        succ1 += 1
    V3 = X3_sim.T[-1].dot(S3).dot(X3_sim.T[-1])
    if V3 < rho3:
        succ3 += 1

print(f"Percentage of success DIRTRAN: {np.round((succ1/nVerifications)*100,2)}")
print(f"Percentage of success RTC: {np.round((succ2/nVerifications)*100,2)}")
print(f"Percentage of success RTCD: {np.round((succ3/nVerifications)*100,2)}")

# succ1 = 50
# succ2 = 60
# succ3 = 62

# Bar Plot
df = pandas.DataFrame(dict(graph=["DIRTRAN", "RTC", "RTCD"], #n=[succ1,succ2,succ3])) 
                           n =[np.round((succ1/nVerifications)*100,2), np.round((succ2/nVerifications)*100,2), np.round((succ3/nVerifications)*100,2)])) 
fig, ax = plt.subplots(figsize =(11, 6))
ind = np.arange(len(df))
width = 0.1
ax.barh(ind, df.n, width,  label=r"$\frac{successes}{simulations} \%$")
for s in ['top', 'bottom', 'left', 'right']: # Remove axes splines
    ax.spines[s].set_visible(False)
ax.xaxis.set_ticks_position('none') # Remove x, y Ticks
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad = 5) # Add padding between axes and labels
ax.yaxis.set_tick_params(pad = 10)
ax.grid(visible = True, color ='grey', # Add x, y gridlines
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
ax.invert_yaxis() # Show top values
for i in ax.patches: # Add annotation to bars
    plt.text(i.get_width()+0.2, i.get_y()+0.05,
            str(round((i.get_width()), 2)),
            fontsize = 18, fontweight ='bold',
            color ='grey')
ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
ax.legend(fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_xlim(0,100)
#ax.set_xlabel(r"$\frac{volume\ increment(\%)}{time}$")
plt.show() # Show Plot