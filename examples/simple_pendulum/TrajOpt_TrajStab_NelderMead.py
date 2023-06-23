from scipy.optimize import minimize
import numpy as np
import os
import time

from pydrake.examples.pendulum import PendulumPlant
from pydrake.all import FiniteHorizonLinearQuadraticRegulatorOptions, FiniteHorizonLinearQuadraticRegulator, \
                        Linearize, LinearQuadraticRegulator, PiecewisePolynomial

from simple_pendulum.controllers.tvlqr.roa.utils import funnelVolume
from simple_pendulum.controllers.tvlqr.roa.plot import plotFunnel, rhoComparison
from simple_pendulum.utilities.process_data import prepare_trajectory, saveFunnel
from simple_pendulum.controllers.tvlqr.roa.sos import TVsosRhoComputation
from simple_pendulum.controllers.lqr.roa.sos import SOSequalityConstrained
from simple_pendulum.trajectory_optimization.dirtrel.dirtrelTrajOpt import RobustDirtranTrajectoryOptimization

class pendulum():
    def __init__(self, m,l,b,g,tl):
        self.pendulum_plant = PendulumPlant()
        pendulum_context = self.pendulum_plant.CreateDefaultContext()
        pendulum_params = self.pendulum_plant.get_mutable_parameters(pendulum_context)
        pendulum_params[0] = m
        pendulum_params[1] = l
        pendulum_params[2] = b
        pendulum_params[3] = g

        self.m = m
        self.l = l
        self.b = b
        self.g = g
        self.torque_limit = tl

class lqr_controller():
    def __init__(self, pendulum_plant, Q, R):
        goal = [np.pi,0]
        tilqr_context = pendulum_plant.CreateDefaultContext()
        pendulum_plant.get_input_port(0).FixValue(tilqr_context, [0])
        tilqr_context.SetContinuousState(goal)
        linearized_pendulum = Linearize(pendulum_plant, tilqr_context)
        (self.K, self.S) = LinearQuadraticRegulator(linearized_pendulum.A(),
                                        linearized_pendulum.B(),
                                        Q,
                                        [R])

class tvlqr_controller():
    def __init__(self, pendulum_plant,X,U,T, Q, R, Sf):
        u0 = PiecewisePolynomial.FirstOrderHold(T, U)
        x0 = PiecewisePolynomial.CubicShapePreserving(
                                              T,
                                              X,
                                              zero_end_point_derivatives=True)
        pendulum_context = pendulum_plant.CreateDefaultContext()
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.x0 = x0
        options.u0 = u0
        options.Qf = Sf  
        self.tvlqr = FiniteHorizonLinearQuadraticRegulator(
                        pendulum_plant,
                        pendulum_context,
                        t0=options.u0.start_time(),
                        tf=options.u0.end_time(),
                        Q=Q,
                        R=[R],
                        options=options)

class NelderMeadOpt():
    def __init__(self, params, objective):
        ''' Assume params = dict{M,x_t,u_t,q11,q22,r}, M = [m,l], objective can be either "trajectory" or "design" '''
        self.funnel_volume_storage = []
        self.objective = objective
        if self.objective == "trajectory":
            self.pt = np.array([params["q11"],params["q22"],params["r"]])
            self.m = params["M"][0]
            self.l = params["M"][1]
        elif self.objective == "design":
            pass
 
    def objectiveFunction(self,params):
        if self.objective == "trajectory":
            damping = 0.35
            gravity = 9.81
            torque_limit = 3

            # pendulum parameters
            mpar = {"l": self.l,
                    "m": self.m,
                    "b": damping, 
                    "g": gravity,
                    "cf": 0.0,
                    "tl": torque_limit}

            # robust direct transcription parameters
            options = {"N": 51,
                        "R": params[2],
                        "Rl": .1,
                        "Q": np.diag([params[0],params[1]]),
                        "Ql": np.diag([10,1]),
                        "QN": np.eye(2)*100,
                        "QNl": np.eye(2)*100,
                        "D": 0.2*0.2, 
                        "E1": np.eye(2)*0.01, #np.zeros((2,2)),
                        "x0": [0.0,0.0],
                        "xG": [np.pi, 0.0],
                        "tf0": 3,
                        "speed_limit": 7,
                        "theta_limit": 2*np.pi,
                        "time_penalization": 0, 
                        "hBounds": [0.05, 0.05]}

            dirtrel = RobustDirtranTrajectoryOptimization(mpar, options)
            try:
                T, X, U = dirtrel.ComputeTrajectory()
                #funnel_volume = dirtrel.l_w
                log_dir = "data/simple_pendulum/dirtrel"
                traj_data = np.vstack((T, X[0], X[1], U)).T
                traj_path = os.path.join(log_dir, "trajectory_Nelder.csv" )
                np.savetxt(traj_path, traj_data, delimiter=',',
                            header="time,pos,vel,torque", comments="")
            except:
                funnel_volume = 0.001 # 100000
                self.funnel_volume_storage = np.append(self.funnel_volume_storage,funnel_volume)
                print("DIRTREL failure") 
                return funnel_volume
            data_dict = prepare_trajectory(traj_path)
            trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
            self.x_t = [trajectory.T[1].T, trajectory.T[2].T] 
            self.u_t = trajectory.T[3].T
            self.time = trajectory.T[0].T 

            drake_pendulum = pendulum(self.m, self.l, damping, gravity, torque_limit)
            ti_controller = lqr_controller(drake_pendulum.pendulum_plant, options["Q"], options["R"])
            # Taking the finals values of S and rho from the invariant case, SOS method has been chosen
            (rhof, Sf) = SOSequalityConstrained(drake_pendulum,ti_controller)
            tv_controller = tvlqr_controller(drake_pendulum.pendulum_plant, X, U, T, options["Q"], options["R"], Sf)
            N = len(self.time)
            (rho, S) = TVsosRhoComputation(drake_pendulum, tv_controller, self.time, N, rhof, verbose = False)
            funnel_path = f"data/simple_pendulum/funnels/Sosfunnel_Nelder.csv" #f"data/simple_pendulum/funnels/Sosfunnel_NelderMead.csv"
            saveFunnel(rho, S, self.time, funnel_path = funnel_path)
            funnel_volume = funnelVolume(funnel_path, traj_path)
        elif self.objective == "design":
            funnel_volume = None  
        self.funnel_volume_storage = np.append(self.funnel_volume_storage,funnel_volume)
        print("The new funnel volume is: ", funnel_volume) 
        return -funnel_volume #funnel_volume
    
    def solve(self,max_iter, q_bound, r_bound):
        # Define the optimization options and constraints
        options = {'maxiter': max_iter, 
                   'maxfev': max_iter,
                   'disp': False}
        bounds = [(0.1,q_bound),(0.1,q_bound),(0.1,r_bound)]
        # Perform the search
        start = time.time()
        result = minimize(self.objectiveFunction, self.pt, method='nelder-mead', options=options, bounds=bounds)
        optimization_time = int((time.time() - start)/60)
        # Print the result
        print('Status : %s' % result['message'])
        print('The process took %d minutes' % optimization_time)
        print('Total Evaluations: %d' % result['nfev'])
        solution = result['x']
        # evaluation = self.objectiveFunction(solution)
        # print('Solution: f(%s) = %.5f' % (solution, evaluation))
        return solution

if __name__ == "__main__":  
    import matplotlib as mpl
    mpl.use("WebAgg")
    import matplotlib.pyplot as plt

    objective = "trajectory"
    max_iter = 100 
    q_bound = 10
    r_bound = 1

    params = {"M": [0.67, 0.5], # design parameters [m,l]
                "x_t": [0,0], #TODO: for objective = design
                "u_t": 0, #TODO: for objective = design
                "time": 0, #TODO: for objective = design 
                "q11": 10,
                "q22": 1,
                "r": 0.1}

    nm = NelderMeadOpt(params, objective)
    solution = nm.solve(max_iter, q_bound, r_bound)
    Q_opt = np.diag([solution[0], solution[1]])
    R_opt = solution[2]
    print("The optimal Q is: ", Q_opt)
    print("The optimal R is: ", [R_opt])
    N_it = len(nm.funnel_volume_storage)
    plt.plot(range(N_it),nm.funnel_volume_storage)
    plt.show()
    # Q_opt = np.diag([10, 8.02222222]) Nelder1 65 min
    # R_opt = 1.
    # Q_opt = np.diag([10, 3.05233196]) Nelder2 66 min
    # R_opt = 1.0
    # Q_opt = np.diag([10, 5.03305898]) Nelder 61 min
    # R_opt = 1.0

    # pendulum parameters
    mpar = {"m": 0.67,
            "l": 0.5,
            "b": 0.35, 
            "g": 9.81,
            "cf": 0.0,
            "tl": 3}

    log_dir = "data/simple_pendulum/dirtrel"
    traj_path = os.path.join(log_dir, "trajectory_Nelder.csv" )
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    X = [trajectory.T[1].T, trajectory.T[2].T] 
    U = trajectory.T[3].T
    T = trajectory.T[0].T
    T = np.reshape(T, (T.shape[0], -1))
    U = np.reshape(U,(U.shape[0], -1)).T
    drake_pendulum = pendulum(params["M"][0], params["M"][1], mpar["b"], mpar["g"], mpar["tl"])
    ti_controller = lqr_controller(drake_pendulum.pendulum_plant, Q_opt, R_opt)
    # Taking the finals values of S and rho from the invariant case, SOS method has been chosen
    (rhof, Sf) = SOSequalityConstrained(drake_pendulum,ti_controller)
    tv_controller = tvlqr_controller(drake_pendulum.pendulum_plant, X, U, T, Q_opt, R_opt, Sf)
    N = len(trajectory.T[0].T)
    (rho, S) = TVsosRhoComputation(drake_pendulum, tv_controller, trajectory.T[0].T, N, rhof, verbose = False)
    funnel_path = f"data/simple_pendulum/funnels/Sosfunnel_Nelder.csv"
    saveFunnel(rho, S, T, funnel_path = funnel_path)

    dirtrel_funnel_path = "data/simple_pendulum/funnels/SosfunnelDIRTREL.csv"
    dirtrel_traj_path = "data/simple_pendulum/dirtrel/trajectory.csv"
    Q = np.diag([10,1])
    R = 0.1
    trajectory = np.loadtxt(dirtrel_traj_path, skiprows=1, delimiter=",")
    X = [trajectory.T[1].T, trajectory.T[2].T] 
    U = trajectory.T[3].T
    T = trajectory.T[0].T
    T = np.reshape(T, (T.shape[0], -1))
    U = np.reshape(U,(U.shape[0], -1)).T
    drake_pendulum = pendulum(params["M"][0], params["M"][1], mpar["b"], mpar["g"], mpar["tl"])
    ti_controller = lqr_controller(drake_pendulum.pendulum_plant, Q, R)
    # Taking the finals values of S and rho from the invariant case, SOS method has been chosen
    (rhof, Sf) = SOSequalityConstrained(drake_pendulum,ti_controller)
    tv_controller = tvlqr_controller(drake_pendulum.pendulum_plant, X, U, T, Q, R, Sf)
    N = len(trajectory.T[0].T)
    (rho, S) = TVsosRhoComputation(drake_pendulum, tv_controller, trajectory.T[0].T, N, rhof, verbose = False)
    saveFunnel(rho, S, T, funnel_path = dirtrel_funnel_path)

    ax1 = plotFunnel(dirtrel_funnel_path,dirtrel_traj_path)
    plotFunnel(funnel_path,traj_path, ax1)
    rhoComparison(funnel_path, dirtrel_funnel_path, label1="nelder", label2 = "dirtrel")
    print("Volume of Nelder funnel:", funnelVolume(funnel_path,traj_path))
    print("Volume of DIRTREL funnel:", funnelVolume(dirtrel_funnel_path,dirtrel_traj_path))
    plt.show()