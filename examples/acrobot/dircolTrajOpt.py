import numpy as np
import matplotlib as mpl
mpl.use('WebAgg')

from pydrake.all import ( SnoptSolver, DiagramBuilder, Simulator, LogVectorOutput, Saturation, AddMultibodyPlantSceneGraph)
from pydrake.systems.trajectory_optimization import DirectCollocation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.controllers import (FiniteHorizonLinearQuadraticRegulatorOptions,
                                         MakeFiniteHorizonLinearQuadraticRegulator,
                                         FiniteHorizonLinearQuadraticRegulator)
from pydrake.multibody.parsing import Parser

from acrobot.utils.plotting import plot_timeseries
from acrobot.utils.csv_trajectory import save_trajectory
from acrobot.utils.urdfs import generate_urdf
from acrobot.trajOpt.dircol_utils import extract_data_from_polynomial
from acrobot.model.model_parameters import model_parameters
from acrobot.trajOpt.direct_collocation import dircol_calculator

robot = "acrobot"

parameters = "CMA-ES_design1st"
yaml_params_file = "data/acrobot/designParams/"+parameters+".yml"
urdf_template = "data/acrobot/urdfs/design_A.0/model_1.0/acrobot.urdf"
urdf_file = "data/acrobot/urdfs/roaUrdfs/acrobot_"+parameters+".urdf"
mpar = model_parameters(filepath=yaml_params_file)

save_dir = "data/acrobot/dircol/"+robot+"NominalTrajectory_"+parameters+".csv"
save_dir_sim = "data/acrobot/dircol/"+robot+"SimulatedTrajectory_"+parameters+".csv"

kNumTimeSamples = 500
kMinimumTimeStep = 0.01
kMaximumTimeStep = 0.1
torque_limit = 8.0  # N*m
speed_limit = 10
theta_limit = 2*np.pi
x0 = (0, 0, 0, 0)
xG = (np.pi, 0, 0, 0)
R = 0.1
timespan_init = [0., 10.]
time_penalization = 0.

dircol_calc = dircol_calculator(urdf_template, urdf_file, mpar, save_dir = save_dir, robot = robot)
dircol_calc.compute_trajectory(kNumTimeSamples,torque_limit,x0,xG,theta_limit,speed_limit,R,
                               time_penalization, timespan_init, kMinimumTimeStep,kMaximumTimeStep)
freq = 10
T,X,U = dircol_calc.get_trajectory(freq)

plot_timeseries(T,X,U)
save_trajectory(csv_path=save_dir,
                T=T, X=X, U=U)

x0_traj = PiecewisePolynomial.CubicShapePreserving(T,X.T, zero_end_point_derivatives=True)
u0_traj = PiecewisePolynomial.FirstOrderHold(T, np.reshape(U.T[1],(len(U.T[1]),1)).T)

# Setup acrobot plant
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
file_name = urdf_file
Parser(plant).AddModelFromFile(file_name)
plant.Finalize()
acrobot = plant
context = acrobot.CreateDefaultContext()

# Setup step controller with saturation block
Q = np.diag([2.08, 0.15, 0.99, 0.99])
R = np.eye(1)*0.01
Qf = np.copy(Q)
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.x0 = x0_traj
options.u0 = u0_traj
options.Qf = Qf
options.input_port_index = acrobot.get_actuation_input_port().get_index()
controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(acrobot,
                                                    context,
                                                    t0=options.u0.start_time(),
                                                    tf=options.u0.end_time(),
                                                    Q=Q,
                                                    R=R,
                                                    options=options)
controller_plant = builder.AddSystem(controller_sys)
saturation = builder.AddSystem(Saturation(min_value=[-torque_limit], max_value=[torque_limit]))

# Add blocks connections
builder.Connect(controller_plant.get_output_port(),
                saturation.get_input_port())
builder.Connect(saturation.get_output_port(),
                acrobot.get_actuation_input_port())
builder.Connect(acrobot.get_state_output_port(),
                controller_plant.get_input_port())     

# Setup a logger for the acrobot state
dt_log = 0.1
state_logger = LogVectorOutput(acrobot.get_state_output_port(), builder, dt_log)
input_logger = LogVectorOutput(saturation.get_output_port(), builder, dt_log)

# Build-up the diagram
diagram = builder.Build()

# Set up a simulator to run this diagram
simulator = Simulator(diagram)

print("simulating...")
simulator.Initialize()
context = simulator.get_mutable_context()
context.SetTime(options.u0.start_time())
simulator.AdvanceTo(options.u0.end_time())

x_sim = state_logger.FindLog(context).data()
u_sim = input_logger.FindLog(context).data()
t_sim = np.linspace(options.u0.start_time(),options.u0.end_time(),len(x_sim.T))
u2_sim = np.vstack((np.zeros(len(x_sim.T)),u_sim))

plot_timeseries(t_sim,x_sim.T,u2_sim.T, T_des = T, U_des = U, X_des = X)
t_save = np.reshape(t_sim.T,(len(t_sim.T),)).T
save_trajectory(csv_path=save_dir_sim,
                T=t_save.T, X=x_sim.T, U=u2_sim.T)

x0_traj = PiecewisePolynomial.CubicShapePreserving(t_sim.T,x_sim, zero_end_point_derivatives=True)
u0_traj = PiecewisePolynomial.FirstOrderHold(t_sim.T, u_sim)

# Setup acrobot plant
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
file_name = urdf_file
Parser(plant).AddModelFromFile(file_name)
plant.Finalize()
acrobot = plant

context = acrobot.CreateDefaultContext()

# Setup step controller with saturation block
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.x0 = x0_traj
options.u0 = u0_traj
options.Qf = Qf
options.input_port_index = acrobot.get_actuation_input_port().get_index()
controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(acrobot,
                                                    context,
                                                    t0=options.u0.start_time(),
                                                    tf=options.u0.end_time(),
                                                    Q=Q,
                                                    R=R,
                                                    options=options)
controller_plant = builder.AddSystem(controller_sys)
saturation = builder.AddSystem(Saturation(min_value=[-torque_limit], max_value=[torque_limit]))

# Add blocks connections
builder.Connect(controller_plant.get_output_port(),
                saturation.get_input_port())
builder.Connect(saturation.get_output_port(),
                acrobot.get_actuation_input_port())
builder.Connect(acrobot.get_state_output_port(),
                controller_plant.get_input_port())     

# Setup a logger for the acrobot state
dt_log = 0.1
state_logger = LogVectorOutput(acrobot.get_state_output_port(), builder, dt_log)
input_logger = LogVectorOutput(saturation.get_output_port(), builder, dt_log)

# Build-up the diagram
diagram = builder.Build()

# Set up a simulator to run this diagram
simulator = Simulator(diagram)

print("simulating...")
simulator.Initialize()
context = simulator.get_mutable_context()
context.SetTime(options.u0.start_time())
simulator.AdvanceTo(options.u0.end_time())

x_sim2 = state_logger.FindLog(context).data()
u_sim2 = input_logger.FindLog(context).data()
t_sim2 = np.linspace(options.u0.start_time(),options.u0.end_time(),len(x_sim2.T))
u2_sim2 = np.vstack((np.zeros(len(x_sim.T)),u_sim))

plot_timeseries(t_sim2,x_sim2.T,u2_sim2.T, T_des = t_sim.T, U_des = u2_sim.T, X_des = x_sim.T)