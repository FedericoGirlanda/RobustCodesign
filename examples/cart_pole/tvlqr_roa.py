import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

from cart_pole.model.parameters import Cartpole
from cart_pole.utilities.process_data import prepare_trajectory
from cart_pole.simulation.simulator import StepSimulator
from cart_pole.controllers.tvlqr.RoAest.PROBest import probTVROA
from cart_pole.controllers.tvlqr.RoAest.utils import storeFunnel, funnelVolume_convexHull
from cart_pole.controllers.tvlqr.RoAest.plot import plotFunnel, TVfunnelVerification, plotRhoEvolution
from cart_pole.controllers.lqr.RoAest.SOSest import bisect_and_verify

from pydrake.all import Linearize, \
                        LinearQuadraticRegulator, \
                        DiagramBuilder, \
                        AddMultibodyPlantSceneGraph, \
                        Parser

# System, trajectory and controller init
sys = Cartpole("short")
sys.fl = 8
urdf_path = "data/cart_pole/urdfs/cartpole.urdf"
xG = np.array([0,0,0,0])
traj_path = "data/cart_pole/dirtran/trajectory.csv"  # "data/cart_pole/dirtrel/trajectory.csv"   
trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
X = np.array([trajectory.T[1], trajectory.T[2], trajectory.T[3], trajectory.T[4]])
U = np.array([trajectory.T[5]])
T = np.array([trajectory.T[0]]).T 
traj_dict = prepare_trajectory(traj_path)

from time import time
start = time()

# Getting last rho from TI sos
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)
Parser(plant).AddModelFromFile(urdf_path)
plant.Finalize()
tilqr_context = plant.CreateDefaultContext()
input_i = plant.get_actuation_input_port().get_index()
output_i = plant.get_state_output_port().get_index()
plant.get_actuation_input_port().FixValue(tilqr_context, [0])
Q_tilqr = np.diag([10, 10, 1, 1])  
R_tilqr = np.eye(1) * .1
tilqr_context.SetContinuousState(xG)
linearized_cartpole = Linearize(plant, tilqr_context, input_i, output_i,
                                equilibrium_check_tolerance=1e-3) 
(Kf, Sf) = LinearQuadraticRegulator(linearized_cartpole.A(), linearized_cartpole.B(), Q_tilqr, R_tilqr)
hyperparams = {"taylor_deg": 3,
               "lambda_deg": 2}
rhof = bisect_and_verify(sys,Kf,Sf,hyperparams)
print("")
print("Last rho from SOS: ", rhof)
print("")

# Simulation of tvlqr stabilization
traj_x1 = traj_dict["des_cart_pos_list"]
traj_x2 = traj_dict["des_pend_pos_list"]
traj_x3 = traj_dict["des_cart_vel_list"]
traj_x4 = traj_dict["des_pend_vel_list"]
controller_options = {"T_nom": traj_dict["des_time_list"],
                        "U_nom": traj_dict["des_force_list"],
                        "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                        "Q": np.diag([10,10,1,1]),
                        "R": np.array([.1]),
                        "xG": xG}
cartpole = {"urdf": urdf_path,
            "sys": sys,
            "x_lim": 0.35}
dt_sim = 0.001
sim = StepSimulator(cartpole, controller_options, dt_sim)

# Probabilistic RoA est
roaConf = {'rho00': rhof,
           'rho_f': rhof,
           'nSimulations': 100}
estimator = probTVROA(roaConf,sim)
(rho, S) = estimator.doEstimate()
print("The estimated rho is: ", rho)
est_time = int(time()-start)

# Store the obtained funnel
funnel_path = "data/cart_pole/RoA/Probfunnel.csv"
storeFunnel(S,rho,T,funnel_path)
plot_indeces = (0,2)
#plotFunnel(funnel_path, traj_path, plot_indeces)

# Funnel volume calculation
print("Funnel volume: ",funnelVolume_convexHull(funnel_path, traj_path))
print("Seconds needed: ", est_time)

# Funnel Verification
traj_x1 = traj_dict["des_cart_pos_list"]
traj_x2 = traj_dict["des_pend_pos_list"]
traj_x3 = traj_dict["des_cart_vel_list"]
traj_x4 = traj_dict["des_pend_vel_list"]
controller_options = {"T_nom": traj_dict["des_time_list"],
                        "U_nom": traj_dict["des_force_list"],
                        "X_nom": np.vstack((traj_x1, traj_x2, traj_x3, traj_x4)),
                        "Q": np.diag([100,100,.1,.1]),
                        "R": np.array([1]),
                        "xG": np.array([0,0,0,0])}
cartpole = {"urdf": urdf_path,
            "sys": sys,
            "x_lim": 0.35}
dt_sim = 0.01
n_sim = 100
knot = 0
sim = StepSimulator(cartpole, controller_options, dt_sim)
TVfunnelVerification(sim, funnel_path, n_sim, knot)
plt.show()