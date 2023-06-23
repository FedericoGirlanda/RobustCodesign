import numpy as np

from pydrake.all import ( DiagramBuilder, Simulator, LogVectorOutput, Saturation, AddMultibodyPlantSceneGraph, 
                            ExtractSimulatorConfig, ApplySimulatorConfig)
from pydrake.systems.controllers import (FiniteHorizonLinearQuadraticRegulatorOptions,
                                         MakeFiniteHorizonLinearQuadraticRegulator,
                                         FiniteHorizonLinearQuadraticRegulator)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.multibody.parsing import Parser

from acrobot.utils.csv_trajectory import load_trajectory

class DrakeStepSimulator():
    def __init__(self, csv_controller, Q,R,Qf, torque_limit, urdf, dt_sim, dt_log = 0.001, robot = "acrobot"):

        # type of robot
        self.robot = robot

        # load the nominal trajectory from csv file
        self.T_nom, self.X_nom, self.U_nom = load_trajectory(csv_path=csv_controller, with_tau=True)

        # controller parameters
        self.Q = Q
        self.R = R
        self.Qf = Qf

        # saturation and logging parameters
        self.torque_limit = torque_limit
        self.dt_log = dt_log

        # initial state
        self.x0 = np.array([None, None, None, None])

        # plant urdf
        self.urdf = urdf

        # simulation time step
        self.dt_sim = dt_sim

    def init_simulation(self):

        # Setup acrobot plant
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(plant).AddModelFromFile(self.urdf)
        plant.Finalize()
        context = plant.CreateDefaultContext()

        # Setup step controller with saturation block        
        controller_sys, self.tvlqr_S = TVLQRController(self.T_nom.T,self.X_nom,self.U_nom,self.Q,self.R,self.Qf,plant,context, self.robot)
        controller_plant = builder.AddSystem(controller_sys)

        if self.robot == "acrobot":
            saturation = builder.AddSystem(Saturation(min_value=[-self.torque_limit[1]], max_value=[self.torque_limit[1]]))
        elif self.robot == "pendubot":
            saturation = builder.AddSystem(Saturation(min_value=[-self.torque_limit[0]], max_value=[self.torque_limit[0]]))

        # Add blocks connections
        builder.Connect(controller_plant.get_output_port(),
                        saturation.get_input_port())
        builder.Connect(saturation.get_output_port(),
                        plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(),
                        controller_plant.get_input_port())     
        
        # Setup a logger for the acrobot state
        self.state_logger = LogVectorOutput(plant.get_state_output_port(), builder, self.dt_log)
        self.input_logger = LogVectorOutput(saturation.get_output_port(), builder, self.dt_log)

        # Build-up the diagram
        self.diagram = builder.Build()

    def simulate(self, x0, init_knot, final_knot = -1):
        # Define timings and states for the simulation
        t0 = self.T_nom[init_knot]
        tf = self.T_nom[final_knot]

        # Set up a simulator to run this diagram
        self.simulator = Simulator(self.diagram)

        config = ExtractSimulatorConfig(self.simulator)
        config.max_step_size = self.dt_sim
        ApplySimulatorConfig(self.simulator, config)

        context = self.simulator.get_mutable_context()

        # Set the initial conditions (theta1, theta2, theta1dot, theta2dot)
        context.SetContinuousState(x0)
        context.SetTime(t0)

        self.simulator.AdvanceTo(tf)

        x_sim = self.state_logger.FindLog(context).data()
        u_sim = self.input_logger.FindLog(context).data()
        t_sim = np.linspace(t0,tf,len(x_sim.T))

        if self.robot == "acrobot":
            u_sim = np.vstack((np.zeros(len(x_sim.T)), u_sim))
        if self.robot == "pendubot":
            u_sim = np.vstack((u_sim, np.zeros(len(x_sim.T))))

        return t_sim, x_sim, u_sim

def TVLQRController(T,X,U, Q, R, Qf, robot_model, context, robot = "acrobot"):

    x0 = PiecewisePolynomial.CubicShapePreserving(T, X.T, zero_end_point_derivatives=True)

    if robot == "acrobot":
        u0 = PiecewisePolynomial.FirstOrderHold(np.reshape(T,(T.shape[0],1)), np.reshape(U.T[1],(U.T[1].shape[0],1)).T)
    elif robot == "pendubot":
        u0 = PiecewisePolynomial.FirstOrderHold(np.reshape(T,(T.shape[0],1)), np.reshape(U.T[0],(U.T[0].shape[0],1)).T)

    # tvlqr plant construction with drake
    options = FiniteHorizonLinearQuadraticRegulatorOptions()
    options.x0 = x0
    options.u0 = u0
    options.Qf = Qf
    options.input_port_index = robot_model.get_actuation_input_port().get_index()
    tvlqr_sys = MakeFiniteHorizonLinearQuadraticRegulator(robot_model,
                                                        context,
                                                        t0=options.u0.start_time(),
                                                        tf=options.u0.end_time(),
                                                        Q=Q,
                                                        R=R,
                                                        options=options)

    # tvlqr function construction with drake
    tvlqr_funct = FiniteHorizonLinearQuadraticRegulator(robot_model,
                                                        context,
                                                        t0=options.u0.start_time(),
                                                        tf=options.u0.end_time(),
                                                        Q=Q,
                                                        R=R,
                                                        options=options)

    return tvlqr_sys, tvlqr_funct.S