from asyncio import format_helpers
import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('WebAgg')

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.utils.csv_trajectory import load_trajectory
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.simulation.simulation import Simulator

from roatools.vis import saveFunnel, plotFunnel, TVrhoVerification, plotRhoEvolution
from roatools.obj_fcts import caprr_coopt_interface, tv_caprr_coopt_interface

robot = "acrobot"

## Model parameters from urdf and yaml
parameters = "CMA-ES_design1st"
yaml_path = "data/acrobot/roaParameters/"+parameters+".yml"
urdf_path = "data/urdfs/roaUrdfs/acrobot_"+parameters+".urdf"

## loading the trajectory from csv
trajOpt_method = "dircol"
csv_path_nominal = "data/acrobot/" +trajOpt_method+"/"+trajOpt_method+"_roa/simulatedTrajectory_"+parameters+"_cut.csv" 
#csv_path_nominal = "data/acrobot/"+trajOpt_method+"/"+trajOpt_method+"_roa/nominalTrajectory_"+parameters+"_cut.csv" 
read_with = "numpy"  
T_nom, X_nom, U_nom = load_trajectory(csv_path_nominal)

## Model parameters from yaml
mpar = model_parameters(filepath=yaml_path)
design_params = {"m": mpar.m,
                 "l": mpar.l,
                 "lc": mpar.r,
                 "b": mpar.b,
                 "fc": mpar.cf,
                 "g": mpar.g,
                 "I": mpar.I,
                 "tau_max": mpar.tl}

## Controller initialization
par_dict = yaml.safe_load(open(yaml_path, 'r'))
Q = np.diag([par_dict['q11'], par_dict['q22'], par_dict['q33'], par_dict['q44']])
#R = np.eye(1)*par_dict['r11']
R = np.eye(1)*0.01
Qf = np.copy(Q)

## Time-invariant rho estimation
# roa_backend = "sos_con"
# Rf = np.eye(2)*par_dict['r11']
# roa_calc = caprr_coopt_interface(design_params=design_params,
#                                  Q=Qf,
#                                  R=Rf,
#                                  backend=roa_backend)
# roa_calc._update_lqr(Q=Qf, R=Rf)
# vol, rho_f, S_f = roa_calc._estimate()
# print("rho upright: ", rho_f)
# print("volume upright: ", vol)
# print("----")
rho_f = 1.1
S_f = np.array([[622.12901601, 228.08761529, 124.20241198,  62.84371692,],
                [228.08761529,  84.45599505,  45.62011666,  23.09090088,],
                [124.20241198,  45.62011666,  24.89428345,  12.59440845,],
                [ 62.84371692,  23.09090088,  12.59440845,   6.37901051]])

# Time-varying rho estimation
roaConf = { "robot": robot,                 # type of robot considered
            "urdf": urdf_path,              # urdf acrobot parameters
            "mpar": mpar,                   # yaml acrobot parameters
            "Q": Q,                         # tvlqr controller gains
            "R": R,
            "Qf": Qf,
            "traj_csv": csv_path_nominal,   # nominal trajectory
            "nSimulations":200,             # number of simulations in each interval
            "rho00":1000,                   # big initial guess for rho
            "rho_f":rho_f,                  # rho from top LQR RoA estimation
            "Sf":S_f,                      # S from top LQR RoA estimation
            "max_dt": 0.001                 # max dt for the simulation
}

# tv_roa_method =  "tv_sos" #"tv_prob"
# roa_calc = tv_caprr_coopt_interface(design_params, roaConf, backend=tv_roa_method)
# vol_t, rho_t, S_t = roa_calc._estimate()
# print("rho trajectory: ", rho_t)
# print("volume trajectory: ", vol_t)
# print("----")

# csv_funnel = saveFunnel(rho_t, S_t, T_nom, estMethod = tv_roa_method)
# #csv_funnel = "data/acrobot/tvlqr/roa/tv_probfunnel-60.csv" # Usefull for visualization without computation

# nVerifications = 100
# verifiedIdx = 0
# # choice of which projection of the state space we want to visualize
# pos1_idx = 0 # pos1
# pos2_idx = 1 #pos2
# vel1_idx = 2 #vel1
# vel2_idx = 3 #vel2
# plot_idx0 = pos1_idx
# plot_idx1 = vel1_idx
# ax = plotFunnel(csv_funnel, csv_path_nominal, plot_idx0,plot_idx1)
# TVrhoVerification(csv_funnel,csv_path_nominal,nVerifications,verifiedIdx, roaConf, plot_idx0,plot_idx1, ax)
# plt.show()
# assert False

from pydrake.all import ( DiagramBuilder, Simulator, LogVectorOutput, Saturation, AddMultibodyPlantSceneGraph, 
                            ExtractSimulatorConfig, ApplySimulatorConfig)
from pydrake.systems.controllers import (FiniteHorizonLinearQuadraticRegulatorOptions,
                                         FiniteHorizonLinearQuadraticRegulator,
                                         MakeFiniteHorizonLinearQuadraticRegulator)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.multibody.parsing import Parser
from pydrake.all import (MathematicalProgram, Solve, Variables, Variable)
from pydrake.symbolic import Polynomial as simb_poly
import pydrake.symbolic as sym

def Acrobot_fExplicit(params,x, u, lib):

    """
    Facility in order to deal with the Dynamics definition in the SOS estimation method.
    """
    
    ## Model parameters
    I1  = params["I"][0]
    I2  = params["I"][1]
    m1  = params["m"][0]
    m2  = params["m"][1]
    l1  = params["l"][0]
    # l2  = params["l"][1] # because use of r2
    r1  = params["lc"][0]
    r2  = params["lc"][1]
    b1  = params["b"][0] 
    b2  = params["b"][1] 
    fc1 = params["fc"][0] 
    fc2 = params["fc"][1]
    g   = params["g"]

    q    = x[0:2] #Change of coordinates for manipulator eqs (this is not in error coords)
    qd   = x[2:4]
    q1   = q[0]
    q2   = q[1]
    qd1  = qd[0]
    qd2  = qd[1]   

    m11 = I1 + I2 + m2*l1**2 + 2*m2*l1*r2*lib.cos(q2) # mass matrix
    m12 = I2 + m2 *l1 * r2 * lib.cos(q2)
    m21 = I2 + m2 *l1 * r2 * lib.cos(q2)
    m22 = I2
    # M   = np.array([[m11,m12],
    #                 [m21,m22]])
    det_M = m22*m11-m12*m21
    M_inv = (1/det_M) * np.array([  [m22,-m12],
                                    [-m21,m11]])

    c11 =  -2 * m2 * l1 * r2 * lib.sin(q2) * qd2 # coriolis matrix
    c12 = -m2 * l1 * r2 * lib.sin(q2) * qd2
    c21 =  m2 * l1 * r2 * lib.sin(q2) * qd1
    c22 = 0
    C   = np.array([[c11,c12],
                    [c21,c22]])

    sin12 = (lib.sin(q1)*lib.cos(q2)) + (lib.sin(q2)*lib.cos(q1)) # sen(q1+q2) = sen(q1)cos(q2) + sen(q2)cos(q1)
    g1 = -m1*g*r1*lib.sin(q1) - m2*g*(l1*lib.sin(q1) + r2*sin12) # gravity matrix
    g2 = -m2*g*r2*sin12
    G  = np.array([g1,g2])


    if lib == sym:
        f1 = b1*qd1 + fc1*lib.atan(100*qd1) # coloumb vector symbolic for taylor
        f2 = b2*qd2 + fc2*lib.atan(100*qd2)
        F = np.array([f1,f2])
    elif lib == np:
        f1 = b1*qd1 + fc1*lib.arctan(100*qd1) # coloumb vector nominal
        f2 = b2*qd2 + fc2*lib.arctan(100*qd2)
        F = np.array([f1,f2])

    B  = np.array([[0,0],[0,1]]) # b matrix acrobot

    f_res       = M_inv.dot(   B.dot(u) + G - C.dot(qd) - F ) # explicit dynamics

    return f_res

# Setup acrobot plant
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
file_name = roaConf["urdf"]
Parser(plant).AddModelFromFile(file_name)
plant.Finalize()
acrobot = plant
context = acrobot.CreateDefaultContext()

# nominal trajectory info
T, X, U = load_trajectory(csv_path= roaConf["traj_csv"], with_tau=True)
x0 = PiecewisePolynomial.CubicShapePreserving(T, X.T, zero_end_point_derivatives=True)
u0 = PiecewisePolynomial.FirstOrderHold(np.reshape(T,(T.shape[0],1)), np.reshape(U.T[1],(U.T[1].shape[0],1)).T)
# tvlqr construction with drake
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.x0 = x0
options.u0 = u0
options.Qf = Qf
options.input_port_index = acrobot.get_actuation_input_port().get_index()
# tvlqr construction with drake
controller = FiniteHorizonLinearQuadraticRegulator(acrobot,
                                                    context,
                                                    t0=options.u0.start_time(),
                                                    tf=options.u0.end_time(),
                                                    Q=Q,
                                                    R=R,
                                                    options=options)

hyper_params = {"taylor_deg": 3,
                "lambda_deg": 2,
                "mode": 2}

knot = len(T)-1

# number of knot points
N = len(T)

# Initial rho(t) definition (exponential init)
rho_t = 0.003*np.ones(N)
rho_t[-1] = rho_f

# Sampled constraints
t_iplus1 = T[knot]
t_i = T[knot-1]
dt = t_iplus1 - t_i

## Saturation parameters
torque_limit = design_params["tau_max"][1] 

# Opt. problem definition
prog = MathematicalProgram()
x_bar = prog.NewIndeterminates(4, "x") # shifted system state
gamma = prog.NewContinuousVariables(3)
prog.AddCost(gamma[0]+gamma[1]+gamma[2])
prog.AddConstraint(gamma[0] <= 0)
prog.AddConstraint(gamma[1] <= 0)
prog.AddConstraint(gamma[2] <= 0)

## Dynamics definition
K_i = controller.K.value(t_i)[0]
u0 = controller.u0.value(t_i)[0][0]
ubar = - K_i.dot(x_bar)
u = u0 + ubar
u_minus = - torque_limit
u_plus = torque_limit
u_bar_vec =  np.array([0, ubar])
u_nom_vec = np.array([0,u0])
u_vec = np.array([0,u])
u_minus_vec = np.array([0, u_minus]) # Saturation definition
u_plus_vec  = np.array([0, u_plus])
x_star      = controller.x0.value(t_i) # desired coordinates
x_star = np.reshape(x_star,(4,))
x = x_star+x_bar

f_x = Acrobot_fExplicit(design_params,x,u_vec, sym)
f_x_minus = Acrobot_fExplicit(design_params,x,np.array([0, -torque_limit]), sym)
f_x_plus = Acrobot_fExplicit(design_params,x,np.array([0, torque_limit]), sym)
f_star = Acrobot_fExplicit(design_params,x_star,u_nom_vec, np)
env = { x_bar[0]   : 0, # Taylor approximation of the dynamics
        x_bar[1]   : 0,
        x_bar[2]  : 0,
        x_bar[3]  : 0}
taylor_deg = hyper_params["taylor_deg"]
qdd1_approx        = sym.TaylorExpand(f=f_x[0]-f_star[0],       a=env,  order=taylor_deg)
qdd2_approx        = sym.TaylorExpand(f=f_x[1]-f_star[1],       a=env,  order=taylor_deg)
qdd1_approx_minus  = sym.TaylorExpand(f=f_x_minus[0]-f_star[0], a=env,  order=taylor_deg)
qdd2_approx_minus  = sym.TaylorExpand(f=f_x_minus[1]-f_star[1], a=env,  order=taylor_deg)
qdd1_approx_plus   = sym.TaylorExpand(f=f_x_plus[0]-f_star[0],  a=env,  order=taylor_deg)
qdd2_approx_plus   = sym.TaylorExpand(f=f_x_plus[1]-f_star[1],  a=env,  order=taylor_deg)

f_bar       = np.array([[x_bar[2]], # final nominal and saturated dynamics
                        [x_bar[3]],
                        [qdd1_approx],
                        [qdd2_approx]])
f_bar_minus = np.array([[x_bar[2]],
                        [x_bar[3]],
                        [qdd1_approx_minus],
                        [qdd2_approx_minus]])

f_bar_plus  = np.array([[x_bar[2]],
                        [x_bar[3]],
                        [qdd1_approx_plus],
                        [qdd2_approx_plus]]) 

f_bar = np.reshape(f_bar, (4,))
f_bar_minus = np.reshape(f_bar_minus, (4,))
f_bar_plus = np.reshape(f_bar_plus, (4,))

# Lyapunov function and its derivative
S0_t = controller.S
S0_i = S0_t.value(t_i)
if t_iplus1 == T[-1]:
    S0_iplus1 = roaConf["Sf"]
else:
    S0_iplus1 = S0_t.value(t_iplus1)
S0dot_i = (S0_iplus1-S0_i)/dt
V_i = (x_bar).dot(S0_i.dot(x_bar))
Vdot_i_x = V_i.Jacobian(x_bar).dot(f_bar)
Vdot_i_t = x_bar.dot(S0dot_i.dot(x_bar))
Vdot_i = Vdot_i_x + Vdot_i_t

# Boundaries due to the saturation 
Vdot_minus = Vdot_i_t + V_i.Jacobian(x_bar).dot(f_bar_minus)
Vdot_plus = Vdot_i_t + V_i.Jacobian(x_bar).dot(f_bar_plus)

# Multipliers definition
lambda_deg = hyper_params["lambda_deg"]
h1 = prog.NewFreePolynomial(Variables(x_bar), lambda_deg)
mu_ij1 = h1.ToExpression()
h2 = prog.NewFreePolynomial(Variables(x_bar), lambda_deg)
mu_ij2 = h2.ToExpression()
h3 = prog.NewFreePolynomial(Variables(x_bar), lambda_deg)
mu_ij3 = h3.ToExpression()
lambda_1 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
lambda_2 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
lambda_3 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
lambda_4 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()

# rho dot definition
rho_i = rho_t[knot-1]
rho_iplus1 = rho_t[knot]
rho_dot_i = (rho_iplus1 - rho_i)/dt

# Optimization constraints 
constr_minus = gamma[0] - (Vdot_minus) +rho_dot_i - mu_ij1*(V_i - rho_i) + lambda_1*(-u_minus_vec[1] +ubar)
constr = gamma[1] - (Vdot_i) + rho_dot_i - mu_ij2*(V_i - rho_i) #+ lambda_2*(u_minus_vec[1]-ubar) + lambda_3*(-u_plus_vec[1]+ubar)
constr_plus = gamma[2] - (Vdot_plus) +rho_dot_i - mu_ij3*(V_i - rho_i) + lambda_4*(u_plus_vec[1]-ubar)

for c in [constr]: #[constr_minus, constr, constr_plus]:
    prog.AddSosConstraint(c)

# Solve the problem and store the polynomials
result_mult = Solve(prog)  
h_maps = np.array([result_mult.GetSolution(h1).monomial_to_coefficient_map(),
                    result_mult.GetSolution(h2).monomial_to_coefficient_map(),
                    result_mult.GetSolution(h3).monomial_to_coefficient_map()])
eps = result_mult.get_optimal_cost()

# failing checker
fail = (not result_mult.is_success())
 
if fail:
    print("fail")