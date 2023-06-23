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

parameters = "CMA-ES_design1st"
yaml_params_file = "/home/federico/Documents/robust_codesign/data/acrobot/designParams/"+parameters+".yml"
urdf_template = "/home/federico/Documents/robust_codesign/data/acrobot/urdfs/design_A.0/model_1.0/acrobot.urdf"
urdf_file = "/home/federico/Documents/robust_codesign/data/acrobot/urdfs/roaUrdfs/acrobot_"+parameters+".urdf"
mpar = model_parameters(filepath=yaml_params_file)
generate_urdf(urdf_template,urdf_file, model_pars=mpar)

save_dir = "/home/federico/Documents/robust_codesign/data/acrobot/dircol/nominalTrajectory_"+parameters+".csv"
save_dir_sim = "/home/federico/Documents/robust_codesign/data/acrobot/dircol/simulatedTrajectory_"+parameters+".csv"

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
file_name = urdf_file
Parser(plant).AddModelFromFile(file_name)
plant.Finalize()
context = plant.CreateDefaultContext()

kNumTimeSamples = 100
kMinimumTimeStep = 0.05
kMaximumTimeStep = 0.2

dircol = DirectCollocation(
                plant,
                context,
                num_time_samples=kNumTimeSamples,
                minimum_timestep=kMinimumTimeStep,
                maximum_timestep=kMaximumTimeStep,
                input_port_index=plant.get_actuation_input_port().get_index())
prog = dircol.prog()

# Add equal time interval constraint
dircol.AddEqualTimeIntervalsConstraints()

# Add torque limits
u = dircol.input()[0]
#u_init = dircol.input(0)[0]
torque_limit = 8.0  # N*m
#prog.AddLinearConstraint(u_init == 0)
dircol.AddConstraintToAllKnotPoints(u <= torque_limit)
dircol.AddConstraintToAllKnotPoints(-torque_limit <= u)

# Add state limits
speed_limit = 10
state = dircol.state()
dircol.AddConstraintToAllKnotPoints(state[2] <= speed_limit)
dircol.AddConstraintToAllKnotPoints(-speed_limit <= state[2])
dircol.AddConstraintToAllKnotPoints(state[3] <= speed_limit)
dircol.AddConstraintToAllKnotPoints(-speed_limit <= state[3])

theta_limit = 2*np.pi
dircol.AddConstraintToAllKnotPoints(state[1] <= theta_limit)
dircol.AddConstraintToAllKnotPoints(-theta_limit <= state[1])
# dircol.AddConstraintToAllKnotPoints(state[0] <= theta_limit)
# dircol.AddConstraintToAllKnotPoints(-theta_limit <= state[0])

# Adding initial and final state constraints
x0 = [0, 0, 0, 0]
xG = [np.pi, 0, 0, 0]
prog.AddBoundingBoxConstraint(xG, xG, dircol.final_state())
prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())

# Adding cost on input "effort"
R = 0.1
dircol.AddRunningCost((R * u) * u)

# Setting an initial trajectory guess
timespan_init = 13
traj_init_t = np.column_stack((0, timespan_init)).T
traj_init_x = PiecewisePolynomial.FirstOrderHold(traj_init_t, np.column_stack((x0, xG)))
dircol.SetInitialTrajectory(PiecewisePolynomial(), traj_init_x)

print("The trajectory optimization has started...")
solver = SnoptSolver()
result = solver.Solve(prog)

assert result.is_success()
print("The trajectory optimization has succeded!")

x0_traj = dircol.ReconstructStateTrajectory(result)
u0_traj = dircol.ReconstructInputTrajectory(result)

freq = 50
x, t = extract_data_from_polynomial(x0_traj, freq)
u, _ = extract_data_from_polynomial(u0_traj, freq)
u2 = np.vstack((np.zeros(u.size),u))

plot_timeseries(t.T,x.T,u2.T)
t = np.reshape(t.T,(len(t.T),)).T
traj_file = save_dir
save_trajectory(csv_path=traj_file,
                T=t.T, X=x.T, U=u2.T)
# plot_timeseries(t,x.T,u2.T)
#assert False

x0_traj = PiecewisePolynomial.CubicShapePreserving(t.T,x, zero_end_point_derivatives=True)
u0_traj = PiecewisePolynomial.FirstOrderHold(t.T, u)

# Setup acrobot plant
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
file_name = urdf_file
Parser(plant).AddModelFromFile(file_name)
plant.Finalize()
acrobot = plant
context = acrobot.CreateDefaultContext()

# Setup step controller with saturation block
# Q = np.diag([10,10,1,1])
# R = np.eye(1)*2
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

t_save = np.reshape(t_sim.T,(len(t_sim.T),)).T
traj_file = save_dir_sim
save_trajectory(csv_path=traj_file,
                T=t_save.T, X=x_sim.T, U=u2_sim.T)

plot_timeseries(t_sim,x_sim.T,u2_sim.T, T_des = t.T, U_des = u2.T, X_des = x.T)

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
# Q = np.diag([10,10,1,1])
# R = np.eye(1)*2
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

x_sim2 = state_logger.FindLog(context).data()
u_sim2 = input_logger.FindLog(context).data()
t_sim2 = np.linspace(options.u0.start_time(),options.u0.end_time(),len(x_sim2.T))
u2_sim2 = np.vstack((np.zeros(len(x_sim.T)),u_sim))

plot_timeseries(t_sim2,x_sim2.T,u2_sim2.T, T_des = t_sim.T, U_des = u2_sim.T, X_des = x_sim.T)