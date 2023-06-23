import numpy as np

from pydrake.trajectories import PiecewisePolynomial
from pydrake.solvers.snopt import SnoptSolver
from pydrake.symbolic import sin, cos, atan
from pydrake.all import MathematicalProgram, MultibodyPlant, SceneGraph, Parser

class RobustDirtranTrajectoryOptimization():

    def __init__(self, params, options, urdf):
        self.prog = MathematicalProgram()
        self.options = options
        self.params = params
        
        # Setup of the Variables
        self.N = options["N"]
        self.h_vars = self.prog.NewContinuousVariables(self.N-1, "h")
        self.x_vars = np.array([self.prog.NewContinuousVariables(self.N, "x0"),
                                self.prog.NewContinuousVariables(self.N, "x1"),
                                self.prog.NewContinuousVariables(self.N, "x2"),
                                self.prog.NewContinuousVariables(self.N, "x3")])
        self.u_vars = self.prog.NewContinuousVariables(self.N, "u")
    
    def ComputeTrajectory(self):
        # Create constraints for dynamics and add them
        self.AddDynamicConstraints(self.options["x0"],self.options["xG"])

        # Add a linear constraint to a variable in all the knot points
        self.AddConstraintToAllKnotPoints(self.h_vars,self.options["hBounds"][0],self.options["hBounds"][1])
        self.AddConstraintToAllKnotPoints(self.u_vars,-self.params.tl[1],self.params.tl[1])
        self.AddConstraintToAllKnotPoints(self.x_vars[0],-self.options["theta_limit"],self.options["theta_limit"])
        self.AddConstraintToAllKnotPoints(self.x_vars[1],-self.options["theta_limit"],self.options["theta_limit"])
        self.AddConstraintToAllKnotPoints(self.x_vars[2],-self.options["speed_limit"],self.options["speed_limit"])
        self.AddConstraintToAllKnotPoints(self.x_vars[3],-self.options["speed_limit"],self.options["speed_limit"])

        # Add an initial guess for the resulting trajectory
        self.AddStateInitialGuess(self.options["x0"],self.options["xG"])

        # Add cost on the final state
        self.AddFinalStateCost(self.options["QN"])

        # Add integrative cost
        self.AddRunningCost(self.u_vars, self.options["R"])
        self.AddRunningCost(self.x_vars, self.options["Q"])

        # Add robust cost for dirtrel
        # self.robust_input_lowerBound = np.ones((3*self.options["N"],1))*-np.inf
        # self.robust_input_upperBound = np.ones((3*self.options["N"],1))*np.inf
        # self.prog.AddConstraint(self.robust_input_constr, self.robust_input_lowerBound, self.robust_input_upperBound, np.append(self.x_vars,self.u_vars))
        # self.robust_state_lowerBound = np.ones((3*self.options["N"],1))*-np.inf
        # self.robust_state_upperBound = np.ones((3*self.options["N"],1))*np.inf
        # self.prog.AddConstraint(self.robust_state_constr, self.robust_state_lowerBound, self.robust_state_upperBound, np.append(self.x_vars,self.u_vars))
        #self.prog.AddCost(self.robust_cost,np.append(self.x_vars,self.u_vars))

        # Solve the Mathematical program
        solver = SnoptSolver()
        result = solver.Solve(self.prog)
        times, states, inputs = self.GetResultingTrajectory(result)

        return times, states, inputs 

    def AddDynamicConstraints(self, x0, xG):    
        self.prog.AddConstraint(self.x_vars[0][0] == x0[0])
        self.prog.AddConstraint(self.x_vars[1][0] == x0[1])
        self.prog.AddConstraint(self.x_vars[2][0] == x0[2])
        self.prog.AddConstraint(self.x_vars[3][0] == x0[3])
        for i in range(self.N-1):
            x_n = [self.x_vars[0][i],self.x_vars[1][i],self.x_vars[2][i],self.x_vars[3][i]]
            u_n = self.u_vars[i]
            h_n = self.h_vars[i]
            x_nplus1 = self.dynamics_integration(x_n, u_n, h_n)
            self.prog.AddConstraint(self.x_vars[0][i+1] == x_nplus1[0])
            self.prog.AddConstraint(self.x_vars[1][i+1] == x_nplus1[1])
            self.prog.AddConstraint(self.x_vars[2][i+1] == x_nplus1[2])
            self.prog.AddConstraint(self.x_vars[3][i+1] == x_nplus1[3])
        self.prog.AddConstraint(self.x_vars[0][-1] == xG[0])
        self.prog.AddConstraint(self.x_vars[1][-1] == xG[1])
        self.prog.AddConstraint(self.x_vars[2][-1] == xG[2])
        self.prog.AddConstraint(self.x_vars[3][-1] == xG[3])
    
    def AddConstraintToAllKnotPoints(self, traj_vars, lb, ub):
        lb_vec = np.ones(len(traj_vars))*lb
        ub_vec = np.ones(len(traj_vars))*ub
        self.prog.AddLinearConstraint(traj_vars, lb_vec, ub_vec)

    def AddStateInitialGuess(self, init_, end_):
        init_guess0 = np.linspace(init_[0], end_[0],self.N)
        init_guess1 = np.linspace(init_[1], end_[1],self.N)
        init_guess2 = np.linspace(init_[2], end_[2],self.N)
        init_guess3 = np.linspace(init_[3], end_[3],self.N)
        for i in range(self.N):
            self.prog.SetInitialGuess(self.x_vars[0][i], init_guess0[i])
            self.prog.SetInitialGuess(self.x_vars[1][i], init_guess1[i])
            self.prog.SetInitialGuess(self.x_vars[2][i], init_guess2[i])
            self.prog.SetInitialGuess(self.x_vars[3][i], init_guess3[i])

    def AddFinalStateCost(self, QN):
        x_final = self.x_vars.T[-1]
        self.prog.AddCost(x_final.T.dot(QN.dot(x_final)))

    def AddRunningCost(self, traj_vars, cost_matrix):
        cost = 0
        for i in range(len(traj_vars)):
            if not isinstance(cost_matrix, (list, np.ndarray)):
                cost = cost + (cost_matrix * traj_vars[i]**2)
            else:
                cost = cost + (traj_vars.T[i].T.dot(cost_matrix.dot(traj_vars.T[i])))
        self.prog.AddCost(cost)

    def GetResultingTrajectory(self, result):
        timeSteps = result.GetSolution(self.h_vars)
        t_prev = 0
        time_traj = [t_prev]
        for h_i in timeSteps:
            time_traj = np.append(time_traj,[t_prev + h_i])
            t_prev = t_prev + h_i
        state_traj = result.GetSolution(self.x_vars)
        input_traj = np.reshape(result.GetSolution(self.u_vars),(1,self.N))

        return time_traj,state_traj,input_traj 

    def dynamics_integration(self,x_n,u_n, h_n):
        # EULER
        f_n = self.dynamics_f(x_n,u_n)
        x_nplus1 = np.array(x_n) + h_n*np.array(f_n)

        # # RK4 #TODO: why it is some how blocking for the solution of the MathProg?
        # x_n = np.array(x_n)
        # k1 = np.array(self.dynamics_f(x_n,u_n))
        # k2 = np.array(self.dynamics_f(x_n+(h_n*k1/2),u_n))
        # k3 = np.array(self.dynamics_f(x_n+(h_n*k2/2),u_n))
        # k4 = np.array(self.dynamics_f(x_n+h_n*k3,u_n))
        # x_nplus1 = x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*h_n        
        return x_nplus1
    
    def dynamics_f(self, x_n,u_n):
        ## Model parameters
        I1  = self.params.I[0]
        I2  = self.params.I[1]
        m1  = self.params.m[0]
        m2  = self.params.m[1]
        l1  = self.params.l[0]
        # l2  = self.params.l[1] # because use of r2
        r1  = self.params.r[0]
        r2  = self.params.r[1]
        b1  = self.params.b[0] 
        b2  = self.params.b[1] 
        fc1 = self.params.cf[0] 
        fc2 = self.params.cf[1]
        g   = self.params.g

        q    = x_n[0:2] 
        qd   = x_n[2:4]
        q1   = q[0]
        q2   = q[1]
        qd1  = qd[0]
        qd2  = qd[1]   

        m11 = I1 + I2 + m2*l1**2 + 2*m2*l1*r2*cos(q2) # mass matrix
        m12 = I2 + m2 *l1 * r2 * cos(q2)
        m21 = I2 + m2 *l1 * r2 * cos(q2)
        m22 = I2
        # M   = np.array([[m11,m12],
        #                 [m21,m22]])
        det_M = m22*m11-m12*m21
        M_inv = (1/det_M) * np.array([  [m22,-m12],
                                        [-m21,m11]])

        c11 =  -2 * m2 * l1 * r2 * sin(q2) * qd2 # coriolis matrix
        c12 = -m2 * l1 * r2 * sin(q2) * qd2
        c21 =  m2 * l1 * r2 * sin(q2) * qd1
        c22 = 0
        C   = np.array([[c11,c12],
                        [c21,c22]])

        sin12 = sin(q1+q2) #(lib.sin(q1)*lib.cos(q2)) + (lib.sin(q2)*lib.cos(q1)) # sen(q1+q2) = sen(q1)cos(q2) + sen(q2)cos(q1)
        g1 = -m1*g*r1*sin(q1) - m2*g*(l1*sin(q1) + r2*sin12) # gravity matrix
        g2 = -m2*g*r2*sin12
        G  = np.array([[g1],[g2]])

        f1 = b1*qd1 + fc1*atan(100*qd1) # coloumb vector symbolic for taylor
        f2 = b2*qd2 + fc2*atan(100*qd2)
        F = np.array([[f1],[f2]])

        B  = np.array([[0],[1]]) # b matrix acrobot

        qd = np.array(qd).reshape((2,1))  
        qdd = M_inv.dot(   B.dot(u_n) + G - C.dot(qd) - F )
        f_n = np.array([1*qd[0][0],1*qd[1][0],qdd[0][0],qdd[1][0]])
        return f_n
    
    def robust_cost(self, traj_vars):
        N = self.options["N"]
        Rl = self.options["Rl"]
        Ql = self.options["Ql"]
        QNl = self.options["QNl"]
        
        K, self.E = self.deltaLQR(traj_vars)
            
        # Cost computation
        c = 0
        for k in range(N-1):
            c = c + np.trace((Ql + (K[:,:,k].T*Rl).dot(K[:,:,k])).dot(self.E[:,:,k]))
        c = c + np.trace(QNl.dot(self.E[:,:,N-1]))
        return c

    def deltaLQR(self, traj_vars):
        N = self.options["N"]
        D = self.options["D"]
        E1 = self.options["E1"]
        Q = self.options["Q"]
        QN = self.options["QN"]
        R = self.options["R"]

        x_vars = np.array([traj_vars[[i for i in range(N)]],traj_vars[[N+i for i in range(N)]],
                           traj_vars[[(2*N)+i for i in range(N)]],traj_vars[[(3*N)+i for i in range(N)]]])
        u_vars = np.array(traj_vars[[((4*N) + i) for i in range(N)]])

        # Dynamics linearization
        A = np.zeros((4,4,N-1), dtype='O')
        B = np.zeros((4,1,N-1), dtype='O')
        G = np.zeros((4,1,N-1), dtype='O')
        for k in range(N-1):
            x_k = [x_vars[0][k],x_vars[1][k],x_vars[2][k],x_vars[3][k]]
            u_k = u_vars[k]
            f_k, df_k = self.dynamics_f(x_k,u_k)
            A[:,:,k] = np.array(df_k[1])
            B[:,:,k] = np.array([df_k[2]]).T
            G[:,:,k] = np.array([df_k[3]]).T

        # K tvlqr computation
        P = QN
        K = np.zeros((1,4,N-1), dtype='O')
        for k in np.flip(range(N-1)):
            K[:,:,k] = (B[:,:,k].T.dot(P.dot(A[:,:,k])))/(B[:,:,k].T.dot(P.dot(B[:,:,k]))+R)
            P = Q + (K[:,:,k].T*R).dot(K[:,:,k]) + (A[:,:,k] - B[:,:,k].dot(K[:,:,k])).T.dot(P.dot(A[:,:,k] - B[:,:,k].dot(K[:,:,k])))

        # E and H calculation
        E = np.zeros((4,4,N), dtype='O')
        E[:,:,0] = E1
        H = np.zeros((4,1), dtype='O')
        for k in range(N-1):
            E[:,:,k+1] = (A[:,:,k] - B[:,:,k].dot(K[:,:,k])).dot(E[:,:,k].dot((A[:,:,k] - B[:,:,k].dot(K[:,:,k])).T)) + (A[:,:,k] - B[:,:,k].dot(K[:,:,k])).dot(H.dot(G[:,:,k].T)) + G[:,:,k].dot(H.T.dot((A[:,:,k] - B[:,:,k].dot(K[:,:,k])).T)) + G[:,:,k].dot(D*(G[:,:,k].T))
            H = (A[:,:,k] - B[:,:,k].dot(K[:,:,k])).dot(H) + G[:,:,k]*D

        return K, E

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use("WebAgg")
    from pydrake.all import FiniteHorizonLinearQuadraticRegulatorOptions, \
                            DiagramBuilder, Saturation, LogVectorOutput,\
                            MakeFiniteHorizonLinearQuadraticRegulator, Simulator
    from acrobot.utils.plotting import plot_timeseries
    from acrobot.utils.csv_trajectory import save_trajectory, load_trajectory
    from acrobot.model.model_parameters import model_parameters
    from acrobot.trajOpt.dircol_utils import extract_data_from_polynomial

    robot = "acrobot"

    parameters = "CMA-ES_design1st"
    yaml_params_file = "data/acrobot/designParams/"+parameters+".yml"
    mpar = model_parameters(filepath=yaml_params_file)
    urdf_path = "data/acrobot/urdfs/roaUrdfs/acrobot_"+parameters+".urdf"

    save_dir = "data/acrobot/dirtran/"+robot+"NominalTrajectory_"+parameters+".csv"
    save_dir_sim = "data/acrobot/dirtran/"+robot+"SimulatedTrajectory_"+parameters+".csv"

    # direct transcription parameters
    options = {"N": 100,
            "R": 0.1,
            "Rl": 0.1,
            "Q": np.diag([2.08, 0.15, 0.99, 0.99]),
            "Ql": np.diag([10,1,1,1]),
            "QN": np.diag([2.08, 0.15, 0.99, 0.99]),
            "QNl": np.eye(4)*100,
            "D": 0.2*0.2,
            "E1": np.zeros((4,4)),
            "x0": [0.0,0.0,0.0,0.0],
            "xG": [np.pi,0.0,0.0,0.0],
            "tf0": 10.,
            "speed_limit": 10,
            "theta_limit": 2*np.pi,
            "time_penalization": 0,
            "fixed_dt": 0.01, 
            "hBounds": [0.001, 0.1]} 

    traj_file = "my_trajectory.csv"
    second_sim = True
    dirtran = RobustDirtranTrajectoryOptimization(mpar, options, urdf_path)
    T,X,U = dirtran.ComputeTrajectory()
    plt.figure()
    plt.plot(T,X[0], linestyle= "dashed", label = "theta1", color = "blue")
    plt.plot(T,X[1], linestyle= "dashed", label = "theta2", color = "yellow")
    plt.plot(T,X[2], linestyle= "dashed", label = "thetaDot1", color = "green")
    plt.plot(T,X[3], linestyle= "dashed", label = "thetaDot1", color = "red")
    plt.plot(T,U[0], linestyle= "dashed", label = "u1", color = "purple")
    plt.legend()
    plt.show()

    assert False

    # Saving the obtained trajectory
    U0 = np.zeros(np.array(U).size)
    log_dir = "data/"+robot+"/dirtran"
    traj_path = os.path.join(log_dir, traj_file )
    save_trajectory(csv_path=traj_path,T=np.asarray(T).flatten(), X=np.asarray(X).T, U=np.asarray([U0.flatten(), U.flatten()]).T)

    # T, X, U = load_trajectory(csv_path= traj_path, with_tau=True)
    # U = np.reshape(U.T[1],(len(U.T[1]),1)).T
    # X = X.T

    # plt.plot(T,X[0], linestyle= "dashed")
    # plt.plot(T,X[1], linestyle= "dashed")
    # plt.plot(T,X[2], linestyle= "dashed")
    # plt.plot(T,X[3], linestyle= "dashed")
    # plt.plot(T,U[0], linestyle= "dashed")
    # plt.show()

    # Setup Options and Create TVLQR
    u0 = PiecewisePolynomial.FirstOrderHold(T, U)
    x0 = PiecewisePolynomial.CubicShapePreserving(T,
                                                  X,
                                                  zero_end_point_derivatives=True)
    # freq = 100
    # X,T = extract_data_from_polynomial(x0, freq)
    # U,_ = extract_data_from_polynomial(u0, freq)
    # T = np.reshape(T,(np.array(T).shape[1],))
    # u0 = PiecewisePolynomial.FirstOrderHold(T, U)
    # x0 = PiecewisePolynomial.CubicShapePreserving(T,
    #                                               X,
    #                                               zero_end_point_derivatives=True)


    tvlqr_options = FiniteHorizonLinearQuadraticRegulatorOptions()
    tvlqr_options.x0 = x0
    tvlqr_options.u0 = u0
    tvlqr_options.Qf = options["QN"] 
    builder = DiagramBuilder()
    A = builder.AddSystem(dirtran.plant)
    context = A.CreateDefaultContext()
    tvlqr_options.input_port_index = A.get_actuation_input_port().get_index()
    controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(
                            A,
                            context,
                            t0=tvlqr_options.u0.start_time(),
                            tf=tvlqr_options.u0.end_time(),
                            Q=options["Q"],
                            R=[options["R"]*0.1],
                            options=tvlqr_options)

    C = builder.AddSystem(controller_sys)
    saturation = builder.AddSystem(Saturation(min_value=[-mpar.tl[1]], max_value=[mpar.tl[1]]))
    builder.Connect(C.get_output_port(),
                    saturation.get_input_port())
    builder.Connect(saturation.get_output_port(),
                    A.get_actuation_input_port())
    builder.Connect(A.get_state_output_port(),
                    C.get_input_port())     
    state_logger = LogVectorOutput(A.get_state_output_port(), builder, 0.01)
    input_logger = LogVectorOutput(saturation.get_output_port(), builder, 0.01)
    diagram = builder.Build()
    sim = Simulator(diagram)
    context = sim.get_mutable_context()
    sim.AdvanceTo(T[-1])

    x_sim = state_logger.FindLog(context).data()
    u_sim = input_logger.FindLog(context).data()
    t_sim = np.linspace(T[0],T[-1],len(x_sim.T))
    plt.plot(t_sim,x_sim[0], color = "blue")
    plt.plot(t_sim,x_sim[1], color = "yellow")
    plt.plot(t_sim,x_sim[2], color = "green")
    plt.plot(t_sim,x_sim[3], color = "red")
    plt.plot(t_sim,u_sim[0], color = "purple")
    
    if not second_sim:
        plt.show()
    else:
        # Plant initialization for simulation
        builder = DiagramBuilder()
        plant = MultibodyPlant(time_step=0.0)
        scene_graph = SceneGraph()
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        parser = Parser(plant)
        parser.AddModelFromFile(urdf_path)
        plant.Finalize()
        A = builder.AddSystem(plant)
        context = A.CreateDefaultContext()

        # Setup Options and Create TVLQR
        u0 = PiecewisePolynomial.FirstOrderHold(t_sim, u_sim)
        x0 = PiecewisePolynomial.CubicShapePreserving(t_sim,
                                                    x_sim,
                                                    zero_end_point_derivatives=True)
        tvlqr_options = FiniteHorizonLinearQuadraticRegulatorOptions()
        tvlqr_options.x0 = x0
        tvlqr_options.u0 = u0
        tvlqr_options.Qf = options["QN"] 
        tvlqr_options.input_port_index = A.get_actuation_input_port().get_index()
        controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(
                                A,
                                context,
                                t0=tvlqr_options.u0.start_time(),
                                tf=tvlqr_options.u0.end_time(),
                                Q=options["Q"],
                                R=[options["R"]*0.1],
                                options=tvlqr_options)

        C = builder.AddSystem(controller_sys)
        saturation = builder.AddSystem(Saturation(min_value=[-mpar.tl[1]], max_value=[mpar.tl[1]]))
        builder.Connect(C.get_output_port(),
                        saturation.get_input_port())
        builder.Connect(saturation.get_output_port(),
                        A.get_actuation_input_port())
        builder.Connect(A.get_state_output_port(),
                        C.get_input_port())     
        state_logger = LogVectorOutput(A.get_state_output_port(), builder, 0.01)
        input_logger = LogVectorOutput(saturation.get_output_port(), builder, 0.01)
        diagram = builder.Build()
        sim = Simulator(diagram)
        context = sim.get_mutable_context()
        sim.AdvanceTo(t_sim[-1])

        x_sim2 = state_logger.FindLog(context).data()
        u_sim2 = input_logger.FindLog(context).data()
        t_sim2 = np.linspace(t_sim[0],t_sim[-1],len(x_sim2.T))

        U0 = np.zeros(np.array(U).size)
        traj_file = "my_simulated_trajectory.csv"
        traj_path = os.path.join(log_dir, traj_file )
        save_trajectory(csv_path=traj_path,T=np.asarray(t_sim).flatten(), X=np.asarray(x_sim).T, U=np.asarray([U0.flatten(), u_sim.flatten()]).T)

        plt.figure()
        plt.plot(t_sim,x_sim[0], linestyle = "dashed", color = "blue")
        plt.plot(t_sim,x_sim[1], linestyle = "dashed", color = "yellow")
        plt.plot(t_sim,x_sim[2], linestyle = "dashed", color = "green")
        plt.plot(t_sim,x_sim[3], linestyle = "dashed", color = "red")
        plt.plot(t_sim,u_sim[0], linestyle = "dashed", color = "purple")
        plt.plot(t_sim2,x_sim2[0], color = "blue")
        plt.plot(t_sim2,x_sim2[1], color = "yellow")
        plt.plot(t_sim2,x_sim2[2], color = "green")
        plt.plot(t_sim2,x_sim2[3], color = "red")
        plt.plot(t_sim2,u_sim2[0], color = "purple")
        plt.show()