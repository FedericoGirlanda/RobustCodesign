import numpy as np
import sympy as smp
import math

# drake imports
from pydrake.all import (MathematicalProgram, Solve, Variables, Variable, MonomialBasis)
from pydrake.symbolic import Polynomial as simb_poly
import pydrake.symbolic as sym
from pydrake.examples import AcrobotPlant
from sympy.utilities import lambdify
#from pydrake.solvers import mosek

def verify_double_pendulum_rho(rho, params, S, K, robot, taylor_deg=3,
                               lambda_deg=4, mode=2, verbose=False,
                               x_bar_eval=[np.pi, 0, 0, 0]):
    """
    params      --> parameters of pendulum and controller
    taylor_deg  --> degree of the taylor approximation
    lamda_deg   --> degree of SOS lagrange multipliers

    It solves the feasibility SOS problem in one of the three modes:
    0: completely unconstrained (no actuation limits)
        -->     fastest, it will theoretically overestimate the actual RoA
    1(TODO): check only where controls are not saturated
        -->     mid    , will underestimate actual RoA.
        We could actually use a for this.
        Does it make sense to maximize the region in which no saturation
        occurs, i.e. the linearization is valid and the params are well?
    2: also check for saturated dynamics
        -->     slowest, but best estimate. Still much more fast wrt the najafi method.
    """

    # LQR parameters
    S = S  # params["S"]
    K = K  # params["K"]

    # Saturation parameters
    u_plus_vec = np.array(params["tau_max"])
    u_minus_vec = - np.array(params["tau_max"])

    # Opt. Problem definition (Indeterminates in error coordinates)
    prog = MathematicalProgram()
    x_bar = prog.NewIndeterminates(4, "x_bar")
    x_bar_1 = x_bar[0]
    x_bar_2 = x_bar[1]
    xd_bar_1 = x_bar[2]
    xd_bar_2 = x_bar[3]

    # Dynamics definition
    x_star      = np.array([np.pi,0,0,0]) # desired state in physical coordinates
    x           = x_star+x_bar # state in physical coordinates
    u = -K.dot(x_bar) # control input

    f_exp_acc,f_exp_acc_minus,f_exp_acc_plus = SosDoublePendulumDynamics(params,x, np.reshape(u, (2,1)), np.reshape(u_minus_vec, (2,1)), np.reshape(u_plus_vec, (2,1)), sym, robot)  
    q = x[0:2] # Manipulator eqs (this is not in error coords)
    qd = x[2:4]
    q1 = q[0]
    q2 = q[1]
    qd1 = qd[0]
    qd2 = qd[1]

    # Taylor approximation
    env = {x_bar_1: 0,
           x_bar_2: 0,
           xd_bar_1: 0,
           xd_bar_2: 0}
    qdd1_approx = sym.TaylorExpand(f=f_exp_acc[0][0], a=env, order=taylor_deg)
    qdd2_approx = sym.TaylorExpand(f=f_exp_acc[1][0], a=env, order=taylor_deg)
    if mode == 2:
        qdd1_approx_minus = sym.TaylorExpand(f=f_exp_acc_minus[0][0],
                                             a=env,
                                             order=taylor_deg)
        qdd2_approx_minus = sym.TaylorExpand(f=f_exp_acc_minus[1][0],
                                             a=env,
                                             order=taylor_deg)
        qdd1_approx_plus = sym.TaylorExpand(f=f_exp_acc_plus[0][0],
                                            a=env,
                                            order=taylor_deg)
        qdd2_approx_plus = sym.TaylorExpand(f=f_exp_acc_plus[1][0],
                                            a=env,
                                            order=taylor_deg)
    f = np.array([[qd1],
                  [qd2],
                  [qdd1_approx],
                  [qdd2_approx]])

    if mode == 2:
        f_minus = np.array([[qd1],
                            [qd2],
                            [qdd1_approx_minus],
                            [qdd2_approx_minus]])

        f_plus = np.array([[qd1],
                           [qd2],
                           [qdd1_approx_plus],
                           [qdd2_approx_plus]])

    # Definition of the Lyapunov function and of its derivative
    V = x_bar.dot(S.dot(x_bar))
    Vdot = (V.Jacobian(x_bar).dot(f))[0]
    if mode == 2:
        Vdot_minus = (V.Jacobian(x_bar).dot(f_minus))[0]
        Vdot_plus = (V.Jacobian(x_bar).dot(f_plus))[0]

    # Vdot_check = 2*np.dot(x_bar, np.dot(S, f.flatten())) # Checking the effect of the approximation on Vdot
    # f_true = np.array([qd1, qd2, f_exp_acc[0], f_exp_acc[1]])
    # Vdot_true = 2*np.dot(x_bar, np.dot(S, f_true.flatten()))

    # Multipliers definition
    lambda_b = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
    if mode == 2:
        # u in linear range
        lambda_2 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        lambda_3 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        # uplus
        lambda_c = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        lambda_4 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        # uminus
        lambda_a = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        lambda_1 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()

    # Completely unconstrained dynamics
    epsilon = 10e-20
    if mode == 0:
        prog.AddSosConstraint(-Vdot + lambda_b*(V-rho) - epsilon*x_bar.dot(x_bar))

    # Considering the input saturation in the constraints
    if mode == 2:
        if robot == "acrobot":
            nom1 = (+ K.dot(x_bar) + u_minus_vec)[1] # where both nom1 and nom2 are < 0, the nominal dynamics have to be fullfilled
            nom2 = (- K.dot(x_bar) - u_plus_vec)[1]  
            neg = (- K.dot(x_bar) - u_minus_vec)[1]  # where this is < 0, the negative saturated dynamics have to be fullfilled
            pos = (+ K.dot(x_bar) + u_plus_vec)[1]   # where this is < 0, the positive saturated dynamics have to be fullfilled
        if robot == "pendubot":
            nom1 = (+ K.dot(x_bar) + u_minus_vec)[0] # where both nom1 and nom2 are < 0, the nominal dynamics have to be fullfilled
            nom2 = (- K.dot(x_bar) - u_plus_vec)[0]  
            neg = (- K.dot(x_bar) - u_minus_vec)[0]  # where this is < 0, the negative saturated dynamics have to be fullfilled
            pos = (+ K.dot(x_bar) + u_plus_vec)[0]   # where this is < 0, the positive saturated dynamics have to be fullfilled

        prog.AddSosConstraint(-Vdot + lambda_b*(V-rho) + lambda_2*nom1 + lambda_3*nom2 - epsilon*x_bar.dot(x_bar))
        # neg saturation
        prog.AddSosConstraint(-Vdot_minus + lambda_a*(V - rho) + lambda_1*neg - epsilon*x_bar.dot(x_bar))
        # pos saturation
        prog.AddSosConstraint(-Vdot_plus + lambda_c*(V - rho) + lambda_4*pos - epsilon*x_bar.dot(x_bar))

    # Problem solution
    result = Solve(prog)
    # print(prog) # usefull for debugging

    if verbose:
        env = {x_bar_1: x_bar_eval[0],
               x_bar_2: x_bar_eval[1],
               xd_bar_1: x_bar_eval[2],
               xd_bar_2: x_bar_eval[3]}

        print("-K(xBar): ")
        print(-K.dot(x_bar)[1].Evaluate(env))
        print("xBar: ")
        print(sym.Expression(x_bar[0]).Evaluate(env))
        print(sym.Expression(x_bar[1]).Evaluate(env))
        print(sym.Expression(x_bar[2]).Evaluate(env))
        print(sym.Expression(x_bar[3]).Evaluate(env))
        print("dotX (approximated) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(qdd1_approx.Evaluate(env))
        print(qdd2_approx.Evaluate(env))
        print("dotX (true) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(f_exp_acc[0].Evaluate(env))
        print(f_exp_acc[1].Evaluate(env))
        print("V")
        print(V.Evaluate(env))
        print("Vdot (approximated)")
        print(Vdot.Evaluate(env))
        # print("Vdot check (approximated)")
        # print(Vdot_check.Evaluate(env))
        # print("Vdot true")
        # print(Vdot_true.Evaluate(env))
        print("S")
        print(S)

    return result.is_success()


def bisect_and_verify(params, S, K, robot, hyper_params, rho_min=1e-10,
                      rho_max=5, maxiter=15, verbose=False):
    """
    Simple bisection root finding for finding the RoA using the feasibility
    problem.
    The default values have been choosen after multiple trials.
    """
    for i in range(maxiter):
        # np.random.uniform(rho_min,rho_max)
        rho_probe = rho_min+(rho_max-rho_min)/2
        res = verify_double_pendulum_rho(rho_probe,
                                         params,
                                         S,
                                         K,
                                         robot,
                                         taylor_deg=hyper_params["taylor_deg"],
                                         lambda_deg=hyper_params["lambda_deg"],
                                         mode=hyper_params["mode"])
        if verbose:
            print("---")
            print("rho_min:   "+str(rho_min))
            print("rho_probe: "+str(rho_probe)+" verified: "+str(res))
            print("rho_max:   "+str(rho_max))
            print("---")
        if res:
            rho_min = rho_probe
        else:
            rho_max = rho_probe

    return rho_min

def rho_equalityConstrained(params, S, K, robot, taylor_deg=3,
                            lambda_deg=2, verbose=False,
                            x_bar_eval=[np.pi, 0, 0, 0]):
    """
    params      --> parameters of pendulum and controller
    taylor_deg  --> degree of the taylor approximation
    lamda_deg   --> degree of SOS lagrange multipliers

    It solves the equality constrained formulation of the SOS problem just for the completely unconstrained (no actuation limits) case.
    Surprisingly it usually just slightly overestimate the actual RoA.
    The computational time is the worst between the SOS-based method but it is still very convenient wrt the najafi one.
    On the other hand, a bad closed-loop dynamics makes the estimation to drammatically decrease.
    """

    # LQR parameters
    S = S  # params["S"]
    K = K  # params["K"]

    # Saturation parameters
    u_plus_vec = np.array(params["tau_max"])
    u_minus_vec = - np.array(params["tau_max"])

    # Opt. Problem definition (Indeterminates in error coordinates)
    prog = MathematicalProgram()
    x_bar = prog.NewIndeterminates(4, "x_bar")
    x_bar_1 = x_bar[0]
    x_bar_2 = x_bar[1]
    xd_bar_1 = x_bar[2]
    xd_bar_2 = x_bar[3]

    rho = prog.NewContinuousVariables(1, "rho")[0]
    prog.AddCost(-rho) # Aiming to maximize rho

    # Dynamics definition
    x_star      = np.array([np.pi,0,0,0]) # desired state in physical coordinates
    x           = x_star+x_bar # state in physical coordinates
    u = -K.dot(x_bar) # control input

    f_exp_acc,f_exp_acc_minus,f_exp_acc_plus = SosDoublePendulumDynamics(params,x, np.reshape(u, (2,1)), np.reshape(u_minus_vec, (2,1)), np.reshape(u_plus_vec, (2,1)), sym, robot)    
    q = x[0:2] # Manipulator eqs (this is not in error coords)
    qd = x[2:4]
    q1 = q[0]
    q2 = q[1]
    qd1 = qd[0]
    qd2 = qd[1]

    # Taylor approximation
    env = {x_bar_1: 0,
           x_bar_2: 0,
           xd_bar_1: 0,
           xd_bar_2: 0}
    qdd1_approx = sym.TaylorExpand(f=f_exp_acc[0][0], a=env, order=taylor_deg)
    qdd2_approx = sym.TaylorExpand(f=f_exp_acc[1][0], a=env, order=taylor_deg)

    f = np.array([[qd1],
                  [qd2],
                  [qdd1_approx],
                  [qdd2_approx]])

    # Definition of the Lyapunov function and of its derivative
    V = x_bar.dot(S.dot(x_bar))
    Vdot = (V.Jacobian(x_bar).dot(f))[0]

    Vdot_check = 2*np.dot(x_bar, np.dot(S, f.flatten())) # Checking the effect of the approximation on Vdot
    f_true = np.array([qd1, qd2, f_exp_acc[0], f_exp_acc[1]])
    Vdot_true = 2*np.dot(x_bar, np.dot(S, f_true.flatten()))

    # Multipliers definition
    lambda_b = prog.NewFreePolynomial(Variables(x_bar), lambda_deg).ToExpression()

    # Completely unconstrained dynamics
    prog.AddSosConstraint(((x_bar.T).dot(x_bar)**2)*(V - rho) + lambda_b*(Vdot))

    # Problem solution
    result = Solve(prog)
    # print(prog) # usefull for debugging

    if verbose:
        env = {x_bar_1: x_bar_eval[0],
               x_bar_2: x_bar_eval[1],
               xd_bar_1: x_bar_eval[2],
               xd_bar_2: x_bar_eval[3]}

        print("-K(xBar): ")
        print(-K.dot(x_bar)[1].Evaluate(env))
        print("xBar: ")
        print(sym.Expression(x_bar[0]).Evaluate(env))
        print(sym.Expression(x_bar[1]).Evaluate(env))
        print(sym.Expression(x_bar[2]).Evaluate(env))
        print(sym.Expression(x_bar[3]).Evaluate(env))
        print("dotX (approximated) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(qdd1_approx.Evaluate(env))
        print(qdd2_approx.Evaluate(env))
        print("dotX (true) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(f_exp_acc[0].Evaluate(env))
        print(f_exp_acc[1].Evaluate(env))
        print("V")
        print(V.Evaluate(env))
        print("Vdot (approximated)")
        print(Vdot.Evaluate(env))
        print("Vdot check (approximated)")
        print(Vdot_check.Evaluate(env))
        print("Vdot true")
        print(Vdot_true.Evaluate(env))
        print("S")
        print(S)

    return result.GetSolution(rho)

#########################################################################################################################################
# Time-variyng Region of Attraction estimation
#########################################################################################################################################

def TVsosRhoComputation(params, controller, time , rhof, verification_hyper_params, robot = "acrobot"):
    """
    Bilinear alternationturn used for the SOS funnel estimation.

    Parameters
    ----------
    pendulum: simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller: simple_pendulum.controllers.tvlqr.tvlqr
        configured tvlqr controller object
    time: np.array
        time array related to the nominal trajectory
    N: int
        number of considered knot points
    rhof: float
        final rho value, from the time-invariant RoA estimation

    Returns
    -------
    rho_t : np.array
        array that contains the estimated rho value for all the knot points
    S: np.array
        array that contains the S matrix in each knot point
    """
    # number of knot points
    N = len(time)

    # Initial rho(t) definition
    t_min = 0.5*time[-1]
    rho_min = 0.005
    c_left = (1/t_min)*math.log(rho_min/rhof)
    rho_t_left = rhof*np.exp(c_left*time)
    c_right = (1/(t_min-time[-1]))*math.log(rho_min/rhof)
    A_right = rho_min*np.exp(-c_right*t_min)
    rho_t_right = A_right*np.exp(c_right*time)
    rho_t = np.where(time<=t_min,rho_t_left,rho_t_right)

    # c = (1/time[-1])*math.log(rhof/0.001)
    # rho_t = 0.001*np.exp(c*time)

    # rho_t = 0.1*np.ones((N,))
    # rho_t[-1] = rhof

    # Bilinear SOS alternation for improving the first guess
    cost_prev = np.inf
    convergence = False
    while(not convergence):
        # gamma_min = 0
        for knot in np.flip([i for i in range(1,round(N))]):
            print("---------------")
            print(f"Multiplier step in knot {knot-1}:")

            fail = True
            while(fail):
                # Search for a multiplier, fixing rho
                (fail, h_maps, gammas) = TVmultSearch(params, controller, knot, time, rho_t, verification_hyper_params, robot = robot)
                if fail:
                    rho_t[knot-1] = 0.75*rho_t[knot-1]
                else:
                    print(f"The feasible rho is {rho_t[knot-1]}") 
                    if knot == N-1:
                        h_maps_storage = h_maps
                        gamma_max = np.max(gammas)
                    else:
                        h_maps_storage = np.vstack((h_maps,h_maps_storage))
                        gamma_max =  np.vstack((np.max(gammas),gamma_max))

        for knot in np.flip([i for i in range(1,round(N))]):
            print("---------------")
            print(f"V step in knot {knot-1}:")

            fail = True
            while(fail):
                # Search for rho, fixing the multiplier       
                (fail, rho_opt) = TVrhoSearch(params, controller, knot, time, h_maps_storage[knot-1], rho_t, verification_hyper_params, gamma_max[knot-1], robot = robot)
                if fail:
                    rho_t[knot-1] = 0.75*rho_t[knot-1]
                    gamma_max[knot-1] = 1.25*gamma_max[knot-1]
                else:
                    print("RHO IMPROVES!! rho is ", rho_opt)
                    rho_t[knot-1] = rho_opt
        
        # Check for convergence
        eps = 0.1
        if((cost_prev - np.sum(rho_t))/cost_prev < eps): 
            convergence = True  
        cost_prev = np.sum(rho_t)
    
    print("---------------")
    return (rho_t, controller.S)

# def TVsosRhoComputation(params, controller, time , rhof, verification_hyper_params, robot = "acrobot"):
#     """
#     Bilinear alternationturn used for the SOS funnel estimation.

#     Parameters
#     ----------
#     pendulum: simple_pendulum.model.pendulum_plant
#         configured pendulum plant object
#     controller: simple_pendulum.controllers.tvlqr.tvlqr
#         configured tvlqr controller object
#     time: np.array
#         time array related to the nominal trajectory
#     N: int
#         number of considered knot points
#     rhof: float
#         final rho value, from the time-invariant RoA estimation

#     Returns
#     -------
#     rho_t : np.array
#         array that contains the estimated rho value for all the knot points
#     S: np.array
#         array that contains the S matrix in each knot point
#     """
#     # number of knot points
#     N = len(time)

#     # Initial rho(t) definition (composition of exponential init)
#     # t_min = 0.5*time[-1]
#     # rho_min = 0.005
#     # c_left = (1/t_min)*math.log(rho_min/rhof)
#     # rho_t_left = rhof*np.exp(c_left*time)
#     # c_right = (1/(t_min-time[-1]))*math.log(rho_min/rhof)
#     # A_right = rho_min*np.exp(-c_right*t_min)
#     # rho_t_right = A_right*np.exp(c_right*time)
#     # rho_t = np.where(time<=t_min,rho_t_left,rho_t_right)

#     c = (1/time[-1])*math.log(rhof/0.001)
#     rho_t = 0.001*np.exp(c*time)

#     # rho_t = 0.1*np.ones((N,))
#     # rho_t[-1] = rhof

#     # Bilinear SOS alternation for improving the first guess
#     cost_prev = np.inf
#     convergence = False
#     while(not convergence):
#         # gamma_min = 0
#         for knot in np.flip([i for i in range(1,round(N))]):
#             print("---------------")

#             print(f"Multiplier step in knot {knot-1}:")
#             fail = True
#             while(fail):
#                 # Search for a multiplier, fixing rho
#                 (fail, h_maps, gamma) = TVmultSearch(params, controller, knot, time, rho_t, verification_hyper_params, robot = robot)
#                 if fail:
#                     rho_t[knot-1] = 0.75*rho_t[knot-1]
#                 else:
#                     print(f"The feasible rho is {rho_t[knot-1]}") 

#             print(f"V step in knot {knot-1}:")
#             fail = True
#             while(fail):
#                 # Search for rho, fixing the multiplier       
#                 (fail, rho_opt) = TVrhoSearch(params, controller, knot, time, h_maps, rho_t, verification_hyper_params, gamma, robot = robot)
#                 if fail:
#                     rho_t[knot-1] = 0.75*rho_t[knot-1]
#                     gamma = 1.5*gamma
#                 else:
#                     print("RHO IMPROVES!! rho is ", rho_opt)
#             rho_t[knot-1] = rho_opt
        
#         # Check for convergence
#         eps = 0.1
#         if((cost_prev - np.sum(rho_t))/cost_prev < eps): 
#             convergence = True  
#         cost_prev = np.sum(rho_t)
    
#     print("---------------")
#     return (rho_t, controller.S)

def TVrhoSearch(params, controller, knot, time, h_maps, rho_t, hyper_params, gamma_max, robot = "acrobot"):
    """
    V step of the bilinear alternationturn used in the SOS funnel estimation.

    Parameters
    ----------
    pendulum: simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller: simple_pendulum.controllers.tvlqr.tvlqr
        configured tvlqr controller object
    knot: int
        number of considered knot point
    time: np.array
        time array related to the nominal trajectory
    h_map: Dict[pydrake.symbolic.Monomial, pydrake.symbolic.Expression]
        map of the coefficients of the multiplier obtained from the multiplier step
    rho_t: np.array
        array that contains the evolving estimation of the rho values for each knot point

    Returns
    -------
    fail : boolean
        gives info about the correctness of the optimization problem
    rho_opt: float
        optimized rho value for this knot point
    """

    # Sampled constraints
    t_iplus1 = time[knot]
    t_i = time[knot-1]
    dt = t_iplus1 - t_i

    ## Saturation parameters
    if robot == "acrobot":
        torque_limit = params["tau_max"][1] 
    elif robot == "pendubot":
        torque_limit = params["tau_max"][0] 

    # Opt. problem definition
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(4, "x") # not shifted system state
    x_star      = controller.x0.value(t_i) # desired coordinates
    x_star = np.reshape(x_star,(4,))
    x_bar = x - x_star
    rho_i = prog.NewContinuousVariables(1)[0]
    prog.AddCost(-rho_i)
    prog.AddConstraint(rho_i >= 0.75*rho_t[knot-1])
    prog.SetInitialGuess(rho_i, rho_t[knot-1])
    rho_dot_i = (rho_t[knot]-rho_i)/dt

    ## Dynamics definition
    K_i = controller.K.value(t_i)[0]
    u0 = controller.u0.value(t_i)[0][0]
    ubar = - K_i.dot(x_bar)
    u_minus = - torque_limit -u0
    u_plus = torque_limit -u0
    u = ubar + u0
    if robot == "acrobot":
        u_vec = np.array([[0], [u]])
        u_minus_vec = np.array([[0], [u_minus]]) # Saturation definition
        u_plus_vec  = np.array([[0], [u_plus]])
    elif robot == "pendubot":
        u_vec = np.array([[u], [0]])
        u_minus_vec = np.array([[u_minus], [0]]) # Saturation definition
        u_plus_vec  = np.array([[u_plus], [0]])

    f_x,f_x_minus,f_x_plus = SosDoublePendulumDynamics(params,x,u_vec, u_minus_vec,u_plus_vec, sym, robot = robot)
    env = { x[0]   : x_star[0], # Taylor approximation of the dynamics
            x[1]   : x_star[1],
            x[2]  : x_star[2],
            x[3]  : x_star[3]}
    taylor_deg = hyper_params["taylor_deg"]
    qdd1_approx        = sym.TaylorExpand(f=f_x[0][0],       a=env,  order= taylor_deg) - f_x[0][0].Evaluate(env)
    qdd2_approx        = sym.TaylorExpand(f=f_x[1][0],       a=env,  order= taylor_deg) - f_x[1][0].Evaluate(env)
    qdd1_approx_minus  = sym.TaylorExpand(f=f_x_minus[0][0], a=env,  order=taylor_deg) - f_x_minus[0][0].Evaluate(env)
    qdd2_approx_minus  = sym.TaylorExpand(f=f_x_minus[1][0], a=env,  order=taylor_deg) - f_x_minus[1][0].Evaluate(env)
    qdd1_approx_plus   = sym.TaylorExpand(f=f_x_plus[0][0],  a=env,  order=taylor_deg) - f_x_plus[0][0].Evaluate(env)
    qdd2_approx_plus   = sym.TaylorExpand(f=f_x_plus[1][0],  a=env,  order=taylor_deg) - f_x_plus[1][0].Evaluate(env)

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
    u_minus_vec = np.reshape(u_minus_vec, (2,))
    u_plus_vec = np.reshape(u_plus_vec, (2,))

    # Lyapunov function and its derivative
    S_i = controller.S.value(t_i)
    S_iplus1 = controller.S.value(t_iplus1)
    Sdot_i = (S_iplus1-S_i)/dt
    V_i = (x_bar).dot(S_i.dot(x_bar))
    Vdot_i_x = (2*x_bar).dot(S_i.dot(f_bar)) #V_i.Jacobian(x_bar).dot(f_bar)
    Vdot_i_t = x_bar.dot(Sdot_i.dot(x_bar))
    Vdot_i = Vdot_i_x + Vdot_i_t
    Vdot_minus = Vdot_i_t + (2*x_bar).dot(S_i.dot(f_bar_minus)) #V_i.Jacobian(x_bar).dot(f_bar_minus)
    Vdot_plus = Vdot_i_t + (2*x_bar).dot(S_i.dot(f_bar_plus)) #V_i.Jacobian(x_bar).dot(f_bar_plus)

    # Multipliers definition
    lambda_deg = hyper_params["lambda_deg"]
    h = prog.NewFreePolynomial(Variables(x), lambda_deg)
    ordered_basis = list(h.monomial_to_coefficient_map().keys())
    zip_iterator = zip(ordered_basis, list(h_maps[0].values()))
    h_dict = dict(zip_iterator)
    h = simb_poly(h_dict)
    mu_ij = h.ToExpression()
    # New sos multipliers 
    # lambda_1 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0].ToExpression()
    # lambda_4 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0].ToExpression()
    # Retriving the sos multiplier result from the previous problem
    # l1 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0]
    # ordered_basis = list(l1.monomial_to_coefficient_map().keys())
    # zip_iterator = zip(ordered_basis, list(h_maps[1].values()))
    # l1_dict = dict(zip_iterator)
    # l1 = simb_poly(l1_dict).ToExpression()
    # lambda_2 = l1
    # l2 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0]
    # ordered_basis = list(l2.monomial_to_coefficient_map().keys())
    # zip_iterator = zip(ordered_basis, list(h_maps[2].values()))
    # l2_dict = dict(zip_iterator)
    # l2 = simb_poly(l2_dict).ToExpression()
    # lambda_3 = l2

    # Optimization constraints 
    eps = -gamma_max[0] 
    #constr_minus = eps - (Vdot_minus) +rho_dot_i - mu_ij2*(V_i - rho_i) + lambda_1*(-u_minus +ubar)
    constr = eps - (Vdot_i) + rho_dot_i - mu_ij*(V_i - rho_i) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar)
    #constr_plus = eps - (Vdot_plus) +rho_dot_i - mu_ij2*(V_i - rho_i) + lambda_4*(u_plus-ubar)

    for c in [constr]: #[constr_minus, constr, constr_plus]:
        prog.AddSosConstraint(c)

    # Solve the problem
    # solver = mosek.MosekSolver()
    # result = solver.Solve(prog)
    result = Solve(prog)

    # failing checker
    fail = not result.is_success()
    if fail:
        print("rho step Error")
        rho_opt = None
    else:
        rho_opt = result.GetSolution(rho_i)

    return fail, rho_opt

def TVmultSearch(params, controller, knot, time, rho_t, hyper_params, robot = "acrobot"):
    """
    Multiplier step of the bilinear alternationturn used in the SOS funnel estimation.

    Parameters
    ----------
    pendulum: simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller: simple_pendulum.controllers.tvlqr.tvlqr
        configured tvlqr controller object
    knot: int
        number of considered knot point
    time: np.array
        time array related to the nominal trajectory
    rho_t: np.array
        array that contains the evolving estimation of the rho values for each knot point

    Returns
    -------
    fail : boolean
        gives info about the correctness of the optimization problem
    h_map: Dict[pydrake.symbolic.Monomial, pydrake.symbolic.Expression]
        map of the coefficients of the multiplier obtained from the multiplier step
    eps: float
        optimal cost of the optimization problem that can be useful in the V step
    """

    # Sampled constraints
    t_iplus1 = time[knot]
    t_i = time[knot-1]
    dt = t_iplus1 - t_i

    ## Saturation parameters
    if robot == "acrobot":
        torque_limit = params["tau_max"][1] 
    elif robot == "pendubot":
        torque_limit = params["tau_max"][0] 

    # Opt. problem definition
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(4, "x") # not shifted system state
    x_star      = controller.x0.value(t_i) # desired coordinates
    x_star = np.reshape(x_star,(4,))
    x_bar = x - x_star
    #x_bar = prog.NewIndeterminates(4, "xbar") # shifted system state
    #x_star      = controller.x0.value(t_i) # desired coordinates
    # x_star = np.reshape(x_star,(4,))
    # x = x_star+x_bar
    gamma = prog.NewContinuousVariables(1)[0]
    prog.AddCost(-gamma)
    prog.AddConstraint(gamma >= 0)

    ## Dynamics definition
    K_i = controller.K.value(t_i)[0]
    u0 = controller.u0.value(t_i)[0][0]
    ubar = - K_i.dot(x_bar)
    u_minus = - torque_limit -u0
    u_plus = torque_limit -u0
    u = ubar + u0
    if robot == "acrobot":
        u_vec = np.array([[0], [u]])
        u_minus_vec = np.array([[0], [u_minus]]) # Saturation definition
        u_plus_vec  = np.array([[0], [u_plus]])
    elif robot == "pendubot":
        u_vec = np.array([[u], [0]])
        u_minus_vec = np.array([[u_minus], [0]]) # Saturation definition
        u_plus_vec  = np.array([[u_plus], [0]])

    f_x,f_x_minus,f_x_plus = SosDoublePendulumDynamics(params,x,u_vec, u_minus_vec,u_plus_vec, sym, robot = robot)
    env = { x[0]   : x_star[0], # Taylor approximation of the dynamics
            x[1]   : x_star[1],
            x[2]  : x_star[2],
            x[3]  : x_star[3]}
    taylor_deg = hyper_params["taylor_deg"]
    qdd1_approx        = sym.TaylorExpand(f=f_x[0][0],       a=env,  order= taylor_deg) - f_x[0][0].Evaluate(env)
    qdd2_approx        = sym.TaylorExpand(f=f_x[1][0],       a=env,  order= taylor_deg) - f_x[1][0].Evaluate(env)
    qdd1_approx_minus  = sym.TaylorExpand(f=f_x_minus[0][0], a=env,  order=taylor_deg) - f_x_minus[0][0].Evaluate(env)
    qdd2_approx_minus  = sym.TaylorExpand(f=f_x_minus[1][0], a=env,  order=taylor_deg) - f_x_minus[1][0].Evaluate(env)
    qdd1_approx_plus   = sym.TaylorExpand(f=f_x_plus[0][0],  a=env,  order=taylor_deg) - f_x_plus[0][0].Evaluate(env)
    qdd2_approx_plus   = sym.TaylorExpand(f=f_x_plus[1][0],  a=env,  order=taylor_deg) - f_x_plus[1][0].Evaluate(env)

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
    u_minus_vec = np.reshape(u_minus_vec, (2,))
    u_plus_vec = np.reshape(u_plus_vec, (2,))

    # Lyapunov function and its derivative
    S_i = controller.S.value(t_i)
    S_iplus1 = controller.S.value(t_iplus1)
    Sdot_i = (S_iplus1-S_i)/dt
    V_i = (x_bar).dot(S_i.dot(x_bar))
    Vdot_i_x = (2*x_bar).dot(S_i.dot(f_bar)) #V_i.Jacobian(x_bar).dot(f_bar)
    Vdot_i_t = x_bar.dot(Sdot_i.dot(x_bar))
    Vdot_i = Vdot_i_x + Vdot_i_t
    Vdot_minus = Vdot_i_t + (2*x_bar).dot(S_i.dot(f_bar_minus)) #V_i.Jacobian(x_bar).dot(f_bar_minus)
    Vdot_plus = Vdot_i_t + (2*x_bar).dot(S_i.dot(f_bar_plus)) #V_i.Jacobian(x_bar).dot(f_bar_plus)

    # Multipliers definition
    lambda_deg = hyper_params["lambda_deg"]
    h = prog.NewFreePolynomial(Variables(x), lambda_deg) 
    #h = prog.NewFreePolynomial(Variables(x_bar), lambda_deg)
    mu_ij = h.ToExpression()

    hl_2 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0]
    lambda_2 = hl_2.ToExpression()
    hl_3 = prog.NewSosPolynomial(Variables(x), lambda_deg)[0]
    lambda_3 = hl_3.ToExpression()      

    # rho dot definition
    rho_i = rho_t[knot-1]
    rho_iplus1 = rho_t[knot]
    rho_dot_i = (rho_iplus1 - rho_i)/dt

    # Optimization constraints     
    #constr_minus = -gamma - (Vdot_minus) +rho_dot_i - mu_ij*(V_i - rho_i) + lambda_1*(-u_minus +ubar)
    constr = -gamma - (Vdot_i) + rho_dot_i - mu_ij*(V_i - rho_i) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar)
    #constr_plus = -gamma - (Vdot_plus) +rho_dot_i - mu_ij*(V_i - rho_i) + lambda_4*(u_plus-ubar)

    for c in [constr]:  #[constr_minus, constr, constr_plus]:
        prog.AddSosConstraint(c)

    #prog.AddConstraint(mu_ij.Evaluate(env),0,0, x ) #mult(x des) == 0, not working

    # Solve the problem and store the polynomials
    # solver = mosek.MosekSolver()
    # result_mult = solver.Solve(prog) 
    result_mult = Solve(prog) 

    # failing checker
    fail = (not result_mult.is_success())

    if not fail:
        h_maps = np.array([result_mult.GetSolution(h).monomial_to_coefficient_map(), #None, None])
                         result_mult.GetSolution(hl_2).monomial_to_coefficient_map(),
                         result_mult.GetSolution(hl_3).monomial_to_coefficient_map()])
        gammas = np.array([-result_mult.get_optimal_cost()])
        # prog_backoff = MathematicalProgram()
        # x_backoff = prog_backoff.NewIndeterminates(4, "x_b")
        # gamma_backoff = prog_backoff.NewContinuousVariables(1)[0]
        # eps = 1e-05
        # prog_backoff.AddConstraint(-gamma_backoff <= result_mult.get_optimal_cost() + eps)

        # h = prog_backoff.NewFreePolynomial(Variables(x_backoff), lambda_deg)
        # ordered_basis = list(h.monomial_to_coefficient_map().keys())
        # zip_iterator = zip(ordered_basis, list(result_mult.GetSolution(h).monomial_to_coefficient_map().values()))
        # h_dict = dict(zip_iterator)
        # h = simb_poly(h_dict)
        # mu_ij = h.ToExpression()

        # succ_constr = -gamma_backoff - (Vdot_i) + rho_dot_i - mu_ij*(V_i - rho_i)
        # backoff_env = {x[0]   : x_backoff[0], 
        #                 x[1]   : x_backoff[1],
        #                 x[2]  : x_backoff[2],
        #                 x[3]  : x_backoff[3]}
        # constr_backoff = succ_constr.Substitute(backoff_env)
        # prog_backoff.AddSosConstraint(constr_backoff)
        # result_mult_b = Solve(prog_backoff)

        # h_maps[0] = result_mult_b.GetSolution(h).monomial_to_coefficient_map()
        # gammas = np.array([result_mult_b.GetSolution(gamma_backoff)])
        ##################################################################
        # print(Vdot_i.Evaluate({ x[0]   : x_star[0] + 0.1, 
        #                         x[1]   : x_star[1] + 0.1,
        #                         x[2]  : x_star[2] + 0.1,
        #                         x[3]  : x_star[3] + 0.1}))
        # print(V_i.Evaluate({ x[0]   : x_star[0] + 0.1, 
        #                         x[1]   : x_star[1] + 0.1,
        #                         x[2]  : x_star[2] + 0.1,
        #                         x[3]  : x_star[3] + 0.1}))
        # print(-result_mult.get_optimal_cost())
        # print(rho_i)
        # print(rho_dot_i)
        # print(result_mult.GetSolution(h2).ToExpression().Evaluate({ x[0]   : x_star[0] + 0.1, 
        #                                                         x[1]   : x_star[1] + 0.1,
        #                                                         x[2]  : x_star[2] + 0.1,
        #                                                         x[3]  : x_star[3] + 0.1}))
        # succ_constr = result_mult.get_optimal_cost() - (Vdot_i) + rho_dot_i - result_mult.GetSolution(h2).ToExpression()*(V_i - rho_i)
        # print(succ_constr.Evaluate({ x[0]   : x_star[0] + 0.1, 
        #                         x[1]   : x_star[1] + 0.1,
        #                         x[2]  : x_star[2] + 0.1,
        #                         x[3]  : x_star[3] + 0.1}))
        #######################################################################
        # prog_V = MathematicalProgram()
        # x_V = prog_V.NewIndeterminates(4, "xV") # not shifted system state
        # x_bar_V = x_V - x_star
        # ubar_V = - K_i.dot(x_bar_V)
        # rho_i = prog_V.NewContinuousVariables(1)[0]
        # prog_V.AddCost(-rho_i)
        # prog_V.AddConstraint(rho_i >= 0) # .75*rho_t[knot-1])
        # prog_V.SetInitialGuess(rho_i, rho_t[knot-1])
        # rho_dot_i = (rho_t[knot]-rho_i)/dt

        # h_V = prog_V.NewFreePolynomial(Variables(x_V), lambda_deg)
        # ordered_basis = list(h_V.monomial_to_coefficient_map().keys())
        # zip_iterator = zip(ordered_basis, list(result_mult.GetSolution(h).monomial_to_coefficient_map().values()))
        # h_dict = dict(zip_iterator)
        # h_V = simb_poly(h_dict)
        # mu_ij = h_V.ToExpression()
        # #print(mu_ij)

        # # h_old = result_mult.GetSolution(h)
        # # old_vars = h_old.indeterminates()
        # # h_new = h_old.ToExperssion().Substitute()


        # # hl_2 = prog_V.NewSosPolynomial(Variables(x_V), lambda_deg)[0]
        # # lambda_2 = hl_2.ToExpression()
        # # hl_3 = prog_V.NewSosPolynomial(Variables(x_V), lambda_deg)[0]
        # # lambda_3 = hl_3.ToExpression()  

        # V_env = {x[0]   : x_V[0], 
        #                 x[1]   : x_V[1],
        #                 x[2]  : x_V[2],
        #                 x[3]  : x_V[3]}
        # hl_2 = result_mult.GetSolution(hl_2)
        # lambda_2 = hl_2.ToExpression().Substitute(V_env)
        # hl_3 =  result_mult.GetSolution(hl_3)
        # lambda_3 = hl_3.ToExpression().Substitute(V_env) 
        # Vdot_i = Vdot_i.Substitute(V_env) 
        # V_i = V_i.Substitute(V_env)

        # constr_V = result_mult.get_optimal_cost() - (Vdot_i) + rho_dot_i - mu_ij*(V_i -rho_i) + lambda_2*(u_minus-ubar_V) + lambda_3*(-u_plus+ubar_V)
        # prog_V.AddSosConstraint(constr_V)
        # result_mult_V = Solve(prog_V)
        # if result_mult_V.is_success():
        #     print(result_mult_V.GetSolution(rho_i))
        # else:
        #     print("NOOO!")

    else:
        h_maps = None
        gammas = None

    return fail, h_maps, gammas

def SosDoublePendulumDynamics(params,x, u, u_minus_vec, u_plus_vec, lib, robot = "acrobot"):

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

    sin12 = lib.sin(q1+q2) #(lib.sin(q1)*lib.cos(q2)) + (lib.sin(q2)*lib.cos(q1)) # sen(q1+q2) = sen(q1)cos(q2) + sen(q2)cos(q1)
    g1 = -m1*g*r1*lib.sin(q1) - m2*g*(l1*lib.sin(q1) + r2*sin12) # gravity matrix
    g2 = -m2*g*r2*sin12
    G  = np.array([[g1],[g2]])


    if lib == sym:
        f1 = b1*qd1 + fc1*lib.atan(100*qd1) # coloumb vector symbolic for taylor
        f2 = b2*qd2 + fc2*lib.atan(100*qd2)
        F = np.array([[f1],[f2]])
    elif lib == np:
        f1 = b1*qd1 + fc1*lib.arctan(100*qd1) # coloumb vector nominal
        f2 = b2*qd2 + fc2*lib.arctan(100*qd2)
        F = np.array([[f1],[f2]])
    else:
        F1 = b1*qd1 + fc1*lib.atan(100*qd1)
        F2 = b2*qd2 + fc2*lib.atan(100*qd2)
        F = np.array([[F1], [F2]])

    if robot == "acrobot":
        B  = np.array([[0,0],[0,1]]) # b matrix acrobot
    elif robot == "pendubot":
        B  = np.array([[1,0],[0,0]]) # b matrix pendubot

    qd = np.array(qd).reshape((2,1))
    
    f_exp_acc       = M_inv.dot(   B.dot(u) + G - C.dot(qd) - F ) # nominal and saturated explicit dynamics
    f_exp_acc_minus = M_inv.dot(   B.dot( u_minus_vec   ) + G - C.dot(qd) - F )
    f_exp_acc_plus  = M_inv.dot(   B.dot( u_plus_vec    ) + G - C.dot(qd) - F )

    return f_exp_acc, f_exp_acc_minus, f_exp_acc_plus

def AcrobotHandMadeDynamics(params,q, u2, lib = smp):

    """
    Facility in order to deal with the Dynamics definition in the SOS estimation method.
    """
    
    ## Model parameters
    I1  = params["I"][0]
    I2  = params["I"][1]
    m1  = params["m"][0]
    m2  = params["m"][1]
    l1  = params["l"][0]
    l2  = params["l"][1]
    r1  = params["lc"][0]
    r2  = params["lc"][1]
    b1  = params["b"][0] 
    b2  = params["b"][1] 
    fc1 = params["fc"][0] 
    fc2 = params["fc"][1]
    g   = params["g"]

    q1   = q[0]
    q2   = q[1]
    q3  = q[2]
    q4  = q[3]   
    c2 = lib.cos(q2)
    s2 = lib.sin(q2)
    s1 = lib.sin(q1)
    s2 = lib.sin(q2)
    s12 = lib.sin(q1+q2)

    det_M = I2*(I1 + m2*l1**2 -m2*l1**2*c2**2)
    a1 = 2*m2*l1*l2*s2*q3*q4 +m2*l1*l2*s2*q4**2 -g*(m1+m2)*l1*s1 -g*m2*l2*s12 -b1*q3
    a2 = m2*l1*l2*(2*q3 + q4)*s2*q3 -g*m2*l2*s12 +u2 -b2*q4
    f1 = a1*I2 -a2*I2 -a2*m2*l1*l2*c2
    f2 = -a1*m2*l1*l2*c2 +a2*I1 +a2*m2*l1**2 +a2*m2*l1*l2*c2 -f1

    f_q = (1/det_M)*np.array([f1, f2])

    return f_q

def DrakeDoublePendulumDynamics(params,x_eval, u, u_minus_vec, u_plus_vec, robot = "acrobot"):

    """
    Facility in order to deal with the Dynamics definition in the SOS estimation method.
    """

    I1  = params["I"][0]
    I2  = params["I"][1]
    m1  = params["m"][0]
    m2  = params["m"][1]
    l1  = params["l"][0]
    r1  = params["lc"][0]
    r2  = params["lc"][1]
    b1  = params["b"][0] 
    b2  = params["b"][1] 
    fc1 = params["fc"][0] 
    fc2 = params["fc"][1]
    g   = params["g"]
    
    drake_plant = AcrobotPlant()
    context = drake_plant.CreateDefaultContext()
    context.SetContinuousState(x_eval)
    params = drake_plant.get_parameters(context)
    params.set_b1(b1)
    params.set_b2(b2)
    params.set_m1(m1)
    params.set_m2(m2)
    params.set_l1(l1)
    params.set_lc1(r1)
    params.set_lc2(r2)
    params.set_Ic1(I1)
    params.set_Ic2(I2)
    bias = drake_plant.DynamicsBiasTerm(context)
    M = drake_plant.MassMatrix(context)

    if robot == "acrobot":
        B = np.array([0, 1])
    elif robot == "pendubot":
        B = np.array([1, 0])

    u1, u2 = smp.symbols("u1 u2")
    u = smp.Matrix([u1, u2])
    drake_eom = (np.linalg.inv(M).dot(B.dot(u) - bias))
    drake_eom_la = [lambdify(smp.Matrix(u), drake_eom[0]),lambdify(smp.Matrix(u), drake_eom[1])]
    
    f_exp_acc       = [drake_eom_la[0](u[0], u[1]), drake_eom_la[1](u[0], u[1])] # nominal and saturated explicit dynamics
    f_exp_acc_minus = [drake_eom_la[0](u_minus_vec[0], u_minus_vec[1]), drake_eom_la[1](u_minus_vec[0], u_minus_vec[1])]
    f_exp_acc_plus  = [drake_eom_la[0](u_plus_vec[0], u_plus_vec[1]), drake_eom_la[1](u_plus_vec[0], u_plus_vec[1])]

    return f_exp_acc, f_exp_acc_minus, f_exp_acc_plus

if __name__ == "__main__":
    from acrobot.model.symbolic_plant import SymbolicDoublePendulum
    from acrobot.utils.model_parameters import model_parameters
    import sympy as smp
    from sympy.utilities import lambdify
    from pydrake.examples import AcrobotPlant

    x_eval = np.array([0,0,0,0])

    # Model parameters yaml
    parameters = "CMA-ES_design1st"
    yaml_path = "data/acrobot/designParams/"+parameters+".yml"
    mpar = model_parameters(filepath=yaml_path)
    design_params = {"m": mpar.m,
                    "l": mpar.l,
                    "lc": mpar.r,
                    "b": mpar.b,
                    "fc": mpar.cf,
                    "g": mpar.g,
                    "I": mpar.I,
                    "tau_max": mpar.tl}

    q1, q2, qd1, qd2, qdd1, qdd2 = smp.symbols(
            "q1 q2 \dot{q}_1 \dot{q}_2 \ddot{q}_1 \ddot{q}_2")
    u1, u2 = smp.symbols("u1 u2")
    u = smp.Matrix([u1, u2])
    x_lambdify = smp.Matrix([q1, q2, qd1, qd2, u1, u2])

    plant = SymbolicDoublePendulum(model_pars=mpar)
    eom = plant.equation_of_motion()
    eom = plant.replace_parameters(eom)[0]
    eom_la = lambdify(x_lambdify,eom)

    x = smp.Matrix([q1, q2, qd1, qd2])
    mine_eom, f1, f2 = SosDoublePendulumDynamics(design_params,x,u, np.array([0,0]),np.array([0,0]), smp)
    mine_eom_la = lambdify(x_lambdify,mine_eom[0][0])

    hand_eom = AcrobotHandMadeDynamics(design_params, x, u2)
    hand_eom_la = lambdify(smp.Matrix([q1, q2, qd1, qd2, u2]),hand_eom[0])

    drake_plant = AcrobotPlant()
    context = drake_plant.CreateDefaultContext()
    context.SetContinuousState(x_eval)
    params = drake_plant.get_parameters(context)
    params.set_b1(mpar.b[0])
    params.set_b2(mpar.b[1])
    params.set_m1(mpar.m[0])
    params.set_m2(mpar.m[1])
    params.set_l1(mpar.l[0])
    params.set_lc1(mpar.l[0])
    params.set_lc2(mpar.l[1])
    params.set_Ic1(mpar.I[0])
    params.set_Ic2(mpar.I[1])
    params[3] = mpar.l[1]
    bias = drake_plant.DynamicsBiasTerm(context)
    M = drake_plant.MassMatrix(context)
    B = np.array([0, 1])
    drake_eom = (np.linalg.inv(M).dot(B.dot(u) - bias))[1]
    drake_eom_la = lambdify(smp.Matrix(u), drake_eom)

    print(np.linalg.inv(M))

    m11 = mpar.I[0] + mpar.I[1] + mpar.m[1]*mpar.l[0]**2 + 2*mpar.m[1]*mpar.l[0]*mpar.r[1] * np.cos(x_eval[1]) # mass matrix
    m12 = mpar.I[1] + mpar.m[1] * mpar.l[0] * mpar.r[1] * np.cos(x_eval[1])
    m21 = mpar.I[1] + mpar.m[1] *mpar.l[0] * mpar.r[1] * np.cos(x_eval[1])
    m22 = mpar.I[1]
    M   = np.array([[m11,m12],
                    [m21,m22]])
    print(np.linalg.inv(M))

    assert False

    # x_err = smp.Matrix([x_eval[0]-np.pi,x_eval[1],x_eval[2],x_eval[3]]) # closed loop lqr
    # K = np.array([-98.54889065, -34.61472624, -20.05616404,  -8.72258856])
    # u2 = -K.dot(x_err)
    u2 = 0 #open loop
    u1 = 0
    print(eom_la(x_eval[0],x_eval[1],x_eval[2],x_eval[3],u1, u2))
    # print("------------------")
    # print(mine_eom_la(x_eval[0],x_eval[1],x_eval[2],x_eval[3],u1, u2))
    # print("------------------")
    # print(hand_eom_la(x_eval[0],x_eval[1],x_eval[2],x_eval[3], u2))
    print("------------------")
    print(drake_eom_la(u1,u2))

    