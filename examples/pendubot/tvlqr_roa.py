import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('WebAgg')

from acrobot.model.symbolic_plant import SymbolicDoublePendulum
from acrobot.utils.csv_trajectory import load_trajectory
from acrobot.model.model_parameters import model_parameters
from acrobot.utils.csv_trajectory import save_trajectory, load_trajectory
from acrobot.utils.plotting import plot_timeseries
from acrobot.simulation.simulation import Simulator

from acrobot.roaEstimation.vis import saveFunnel, plotFunnel, TVrhoVerification, plotRhoEvolution
from acrobot.roaEstimation.obj_fcts import caprr_coopt_interface, tv_caprr_coopt_interface

robot = "pendubot"

## Model parameters from urdf and yaml
parameters = "pendubot_params"
yaml_path = "data/pendubot/designParams/"+parameters+".yml"
urdf_template = "data/pendubot/urdfs/design_A.0/model_1.0/pendubot.urdf"
urdf_path = "data/pendubot/urdfs/roaUrdfs/pendubot_"+parameters+".urdf"

## Trajectory from csv
csv_path_nominal_controller = "data/pendubot/dircol/simulatedTrajectory_Cut_"+parameters+".csv" 
csv_path_nominal_est = csv_path_nominal_controller
mpar = model_parameters(filepath=yaml_path)
design_params = {"m": mpar.m,
                 "l": mpar.l,
                 "lc": mpar.r,
                 "b": mpar.b,
                 "fc": mpar.cf,
                 "g": mpar.g,
                 "I": mpar.I,
                 "tau_max": mpar.tl}
par_dict = yaml.safe_load(open(yaml_path, 'r'))

## Controller initialization
Q = np.diag([par_dict['q11'], par_dict['q22'], par_dict['q33'], par_dict['q44']]) # IROS optimized parameters
R = np.eye(1)*0.01 # Best obtained stabilization for the trajectory obtained from direct collocation
Qf = np.copy(Q)

## Time-invariant rho estimation
# roa_backend = "sos_con"
# Rf = np.eye(2)*par_dict['r11'] # IROS optimized parameters
# roa_calc = caprr_coopt_interface(design_params=design_params,
#                                  Q=Qf,
#                                  R=Rf,
#                                  backend=roa_backend,
#                                  robot = robot)
# roa_calc._update_lqr(Q=Qf, R=Rf)
# vol, rho_f, S_f = roa_calc._estimate()
# print("rho upright: ", rho_f)
# print("volume upright: ", vol)
# print("----")
#Definition to avoid the previous computation and to speed-up the debugging, for IROS optimized params 
rho_f = 0.1
S_f = np.array([[105.86374092,  84.15606054,  24.90480954,  11.20809097],
                [ 84.15606054,  70.7625208 ,  20.1350422 ,   9.17428507],
                [ 24.90480954,  20.1350422 ,   5.9851045 ,   2.69143225],
                [ 11.20809097,   9.17428507,   2.69143225,   1.23535595]])

## Time-varying rho estimation
roaConf = { "robot": robot,                                 # type of robot considered
            "urdf": urdf_path,                              # urdf acrobot parameters
            "mpar": mpar,                                   # yaml acrobot parameters
            "Q": Q,                                         # tvlqr controller gains
            "R": R,
            "Qf": Qf,
            "traj_csv": csv_path_nominal_est,               # nominal trajectory
            "controller_csv": csv_path_nominal_controller,  # nominal trajectory controller
            "nSimulations": 200,                            # number of simulations in each interval for simulation-based
            "rho00":1000,                                   # big initial guess for rho for simulation-based
            "rho_f":rho_f,                                  # rho from top LQR RoA estimation
            "Sf":S_f,                                       # S from top LQR RoA estimation, not really used
            "max_dt": 0.01                                 # max dt for simulation-based
}

tv_roa_method =  "tv_sos" # use "tv_prob" for the simulation-based 
roa_calc = tv_caprr_coopt_interface(design_params, roaConf, backend=tv_roa_method)
vol_t, rho_t, S_t = roa_calc._estimate()
print("rho trajectory: ", rho_t)
print("volume trajectory: ", vol_t)
print("----")

# ## Plotting and Verification of the Funnel
T_nom, X_nom, U_nom = load_trajectory(csv_path= roaConf["controller_csv"], with_tau=True)
csv_funnel = "data/pendubot/funnels/tv_sosfunnel-60.csv"
#csv_funnel = saveFunnel(rho_t, S_t, T_nom, estMethod = tv_roa_method, robot = robot)
#csv_funnel = "data/"+robot+"/funnels/"+tv_roa_method+f"funnel-{len(T_nom)}.csv" # Usefull for visualization without computation
nVerifications = 100
verifiedIdx = 0
# choice of which projection of the state space we want to visualize
pos1_idx = 0 # pos1
pos2_idx = 1 #pos2
vel1_idx = 2 #vel1
vel2_idx = 3 #vel2
plot_idx0 = pos1_idx
plot_idx1 = vel1_idx
ax = plotFunnel(csv_funnel, roaConf["controller_csv"], plot_idx0,plot_idx1)
TVrhoVerification(csv_funnel,csv_path_nominal_est,nVerifications,verifiedIdx, roaConf, plot_idx0,plot_idx1, ax)
plt.show()