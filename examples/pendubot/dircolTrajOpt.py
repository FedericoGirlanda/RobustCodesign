import numpy as np
import matplotlib as mpl
mpl.use('WebAgg')

from pydrake.all import ( SnoptSolver, DiagramBuilder, Simulator, LogVectorOutput, Saturation, AddMultibodyPlantSceneGraph, 
                        PiecewisePolynomial, Solve)
from pydrake.systems.trajectory_optimization import DirectCollocation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.controllers import (FiniteHorizonLinearQuadraticRegulatorOptions,
                                         MakeFiniteHorizonLinearQuadraticRegulator,
                                         FiniteHorizonLinearQuadraticRegulator)
from pydrake.multibody.parsing import Parser
from pydrake.solvers.snopt import SnoptSolver


from acrobot.trajOpt import dircol_utils
from acrobot.utils.urdfs import generate_urdf
from acrobot.utils.plotting import plot_timeseries
from acrobot.utils.csv_trajectory import save_trajectory, load_trajectory
from acrobot.utils.urdfs import generate_urdf
from acrobot.trajOpt.dircol_utils import extract_data_from_polynomial
from acrobot.model.model_parameters import model_parameters
from acrobot.trajOpt.direct_collocation import dircol_calculator

robot = "pendubot"

parameters = "pendubot_params"
yaml_params_file = "data/"+robot+"/designParams/"+parameters+".yml"
urdf_template = "data/"+robot+"/urdfs/design_A.0/model_1.0/"+robot+".urdf"
urdf_file = "data/"+robot+"/urdfs/roaUrdfs/"+robot+"_"+parameters+".urdf"
mpar = model_parameters(filepath=yaml_params_file)

save_dir = "data/"+robot+"/dircol/nominalTrajectory_"+parameters+".csv"
save_dir_sim = "data/"+robot+"/dircol/simulatedTrajectory_"+parameters+".csv"

kNumTimeSamples = 500
kMinimumTimeStep = 0.01
kMaximumTimeStep = 0.1
torque_limit = 8  # N*m
speed_limit = 10
theta_limit = 2*np.pi
x0 = (0, 0, 0, 0)
xG = (np.pi, 0, 0, 0)
R = 0.1
timespan_init = [0., 10.]
time_penalization = 0.

generate_urdf(urdf_template, urdf_file, model_pars=mpar)
plant, context, scene_graph = dircol_utils.create_plant_from_urdf(urdf_file)
dircol = DirectCollocation( plant,
                            context,
                            num_time_samples=kNumTimeSamples,
                            minimum_timestep=kMinimumTimeStep,
                            maximum_timestep=kMaximumTimeStep,
                            input_port_index=plant.get_actuation_input_port().get_index())

# Add equal time interval constraint
dircol.AddEqualTimeIntervalsConstraints()

# Add torque limit
u = dircol.input()[0]
# Cost on input "effort"
dircol.AddRunningCost(R * u ** 2)
dircol.AddConstraintToAllKnotPoints(-torque_limit <= u)
dircol.AddConstraintToAllKnotPoints(u <= torque_limit)

# Initial state constraint
dircol.prog().AddBoundingBoxConstraint( x0,
                                        x0,
                                        dircol.initial_state())

# Angular velocity constraints
state = dircol.state()
dircol.AddConstraintToAllKnotPoints(state[2] <= speed_limit)
dircol.AddConstraintToAllKnotPoints(-speed_limit <= state[2])
dircol.AddConstraintToAllKnotPoints(state[3] <= speed_limit)
dircol.AddConstraintToAllKnotPoints(-speed_limit <= state[3])

# Add constraint on elbow position
dircol.AddConstraintToAllKnotPoints(state[1] <= theta_limit)
dircol.AddConstraintToAllKnotPoints(-theta_limit <= state[1])

# Final state constraint
dircol.prog().AddBoundingBoxConstraint(xG, xG, dircol.final_state())

# Add a final cost equal to the total duration.
dircol.AddFinalCost(dircol.time() * time_penalization)
initial_x_trajectory = PiecewisePolynomial.FirstOrderHold( timespan_init, np.column_stack((x0, xG)))
dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)
solver = SnoptSolver()
result = solver.Solve(dircol.prog())
assert result.is_success()
        
(x_traj, acc_traj, jerk_traj, u_traj) = dircol_utils.construct_trajectories(dircol, result)
freq = 10
X, T = dircol_utils.extract_data_from_polynomial(x_traj, freq)
u1_traj, _ = dircol_utils.extract_data_from_polynomial(u_traj, freq)
u2_traj = np.zeros((u1_traj.size))
T = np.asarray(T).flatten()
X = np.asarray(X).T
U = np.asarray([u1_traj.flatten(), u2_traj.flatten()]).T

#T, X, U = load_trajectory(save_dir)

plot_timeseries(T,X,U)
save_trajectory(csv_path=save_dir,
                T=T, X=X, U=U)

x0_traj = PiecewisePolynomial.CubicShapePreserving(T,X.T, zero_end_point_derivatives=True)
u0_traj = PiecewisePolynomial.FirstOrderHold(T, np.reshape(U.T[0],(len(U.T[0]),1)).T)

# Setup acrobot plant
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
Parser(plant).AddModelFromFile(urdf_file)
plant.Finalize()
context = plant.CreateDefaultContext()

# Setup step controller with saturation block
# Q = np.diag([1,1,1,1])
# R = np.eye(1)
Q = np.diag([2.08, 0.15, 0.99, 0.99])
R = np.eye(1)*0.01
Qf = np.copy(Q)
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.x0 = x0_traj
options.u0 = u0_traj
options.Qf = Qf
options.input_port_index = plant.get_actuation_input_port().get_index()
controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(plant,
                                                    context,
                                                    t0=options.u0.start_time(),
                                                    tf=options.u0.end_time(),
                                                    Q=Q,
                                                    R=R,
                                                    options=options)
controller_plant = builder.AddSystem(controller_sys)
saturation = builder.AddSystem(Saturation(min_value=[-mpar.tl[0]], max_value=[mpar.tl[0]]))

# Add blocks connections
builder.Connect(controller_plant.get_output_port(),
                saturation.get_input_port())
builder.Connect(saturation.get_output_port(),
                plant.get_actuation_input_port())
builder.Connect(plant.get_state_output_port(),
                controller_plant.get_input_port())     

# Setup a logger for the acrobot state
dt_log = 0.1
state_logger = LogVectorOutput(plant.get_state_output_port(), builder, dt_log)
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
u2_sim = np.vstack((u_sim, np.zeros(len(x_sim.T))))

plot_timeseries(t_sim,x_sim.T,u2_sim.T, T_des = T, U_des = U, X_des = X)
t_save = np.reshape(t_sim.T,(len(t_sim.T),)).T
save_trajectory(csv_path=save_dir_sim,
                T=t_save.T, X=x_sim.T, U=u2_sim.T)

x0_traj = PiecewisePolynomial.CubicShapePreserving(t_sim.T,x_sim, zero_end_point_derivatives=True)
u0_traj = PiecewisePolynomial.FirstOrderHold(t_sim.T, u_sim)

# Setup acrobot plant
builder = DiagramBuilder()
pendubot, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
Parser(pendubot).AddModelFromFile(urdf_file)
pendubot.Finalize()
context = pendubot.CreateDefaultContext()

# Setup step controller with saturation block
options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.x0 = x0_traj
options.u0 = u0_traj
options.Qf = Qf
options.input_port_index = pendubot.get_actuation_input_port().get_index()
controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(pendubot,
                                                    context,
                                                    t0=options.u0.start_time(),
                                                    tf=options.u0.end_time(),
                                                    Q=Q,
                                                    R=R,
                                                    options=options)
controller_plant = builder.AddSystem(controller_sys)
saturation = builder.AddSystem(Saturation(min_value=[-mpar.tl[0]], max_value=[mpar.tl[0]]))

# Add blocks connections
builder.Connect(controller_plant.get_output_port(),
                saturation.get_input_port())
builder.Connect(saturation.get_output_port(),
                pendubot.get_actuation_input_port())
builder.Connect(pendubot.get_state_output_port(),
                controller_plant.get_input_port())     

# Setup a logger for the acrobot state
state_logger = LogVectorOutput(pendubot.get_state_output_port(), builder, dt_log)
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
t_sim2 = np.linspace(options.u0.start_time(),options.u0.end_time(),len(x_sim.T))
u2_sim2 = np.vstack((u_sim, np.zeros(len(x_sim.T))))

plot_timeseries(t_sim2,x_sim2.T,u2_sim2.T, T_des = t_sim.T, U_des = u2_sim.T, X_des = x_sim.T)